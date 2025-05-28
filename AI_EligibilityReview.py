import os 
import pandas as pd
import numpy as np
import random
import re
import threading # To run Model Updates separately to not slow down labelling
from bs4 import BeautifulSoup
from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib # To save model parameters (for when closing programming and comming back)

# ------------------------------------------------------------------------------------
# Global Parameters
# ------------------------------------------------------------------------------------

# Global variables for tracking batches and threading
batch_size = 20  # Minimum batch size
current_batch = []  # Rows selected for the current batch
rows_labeled_in_batch = 0  # Counter for rows labeled in the current batch
background_thread = None  # Thread for background model predictions

current_row_index = None

# Hyperparameters - CNN + RNN
MAX_NUM_WORDS = 10000  # Max vocabulary size
MAX_SEQUENCE_LENGTH = 500  # Max sequence length (padded/truncated)
EMBEDDING_DIM = 100  # Dimension of word embeddings

# Initialize global variables
tokenizer_fitted = False
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# Minimum Manual Review
reviewed_count = 0
min_manual_reviews = 300
model_can_take_over = False
accuracy_value = 0.975 # Once Accruacy > model auto applies labels

# ------------------------------------------------------------------------------------
# Import Data
# ------------------------------------------------------------------------------------

# Function to clean HTML tags and punctuation
def clean_text(text):
    if pd.isna(text):  # Handle missing values
        return '' 
    # Remove HTML tags and punctuation as before
    if isinstance(text, str):
        if text.strip().lower().endswith(('.html', '.htm')) or '/' in text or '\\' in text:
            return text  # Return the text as is, assuming it's not HTML
        try:
            soup = BeautifulSoup(text, "html.parser")
            text_cleaned = soup.get_text()  # Extract text from any valid HTML
        except Exception as e:
            print(f"Error parsing text with BeautifulSoup: {e}")
            return text  # Fallback: return the original text if parsing fails

        # Further cleaning (e.g., remove punctuation, lowercasing, etc.)
        text_cleaned = re.sub(r'[^\w\s]', '', text_cleaned)  # Remove punctuation
        text_cleaned = text.lower().strip()  # Lowercase and remove extra spaces

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words] 

    # Stemming or Lemmatization (choose one or both)
    #stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    #words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]

    # Handle special characters (e.g., emojis)
    words = [word for word in words if re.match(r'[a-zA-Z0-9]+', word)]
    # Join the words back into a string
    normalized_text = ' '.join(words)
    return normalized_text

def select_columns(columns, root):
    """Display a dialog with buttons for each column, arranged dynamically in multiple columns."""
    selected_columns = []

    def toggle_column(column, button):
        """Toggle the selection of a column."""
        if column in selected_columns:
            selected_columns.remove(column)
            button.config(relief=RAISED)
        else:
            selected_columns.append(column)
            button.config(relief=SUNKEN)

    def confirm_selection():
        """Close the selection dialog."""
        dialog.destroy()

    # Create a new Toplevel window
    dialog = Toplevel(root)
    dialog.title("Select Columns")
    Label(dialog, text="Select the columns to use:").pack(pady=10)

    # Determine the number of grid columns based on the total number of items
    num_grid_columns = max(1, min(len(columns) // 10 + 1, 5))  # Dynamic: 1-5 columns

    # Create a frame for the grid layout
    grid_frame = Frame(dialog)
    grid_frame.pack(padx=10, pady=10)

    # Add buttons to the grid
    for i, column in enumerate(columns):
        btn = Button(grid_frame, text=column, relief=RAISED, width=20)
        # Explicitly pass the button and column as arguments to the lambda
        btn.config(command=lambda col=column, b=btn: toggle_column(col, b))
        row, col = divmod(i, num_grid_columns)  # Compute row and column for grid placement
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

    # Add a confirmation button
    Button(dialog, text="Confirm", command=confirm_selection).pack(pady=10)

    # Wait for the user to finish
    dialog.grab_set()
    root.wait_window(dialog)

    return selected_columns

def load_excel():
    global df, eligible_df, excel_file_path, reviewed_count, current_row_index, selected_columns, root, main_pane

    # Create the root window first
    root = Tk()
    root.title("Eligibility Review")

    # Prompt user for Excel file location
    # file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx;*.xls")])

    # Prompt user for Excel file location
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx;*.xls")],
        initialdir="/app/data"  # Default directory inside the container
    )

    if not file_path:
        messagebox.showwarning("File Not Selected", "Please select an Excel file.")
        return

    excel_file_path = file_path
    df = pd.read_excel(file_path)

    df['Eligibility'] = df.get('Eligibility', None)  # Create Eligibility column if not already present
    df['Eligibility'] = df['Eligibility'].astype(object)  # Ensure the column can hold strings

    # Clean the columns based on user selection
    columns = df.columns.tolist()

    # Pass the root window to select_columns
    selected_columns = select_columns(columns, root)  # Store selected columns

    if not selected_columns:
        messagebox.showwarning("No Columns Selected", "Please select at least one column.")
        return

    # Apply cleaning to the selected columns
    for column in selected_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    df['combined_text'] = df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Calculate reviewed count (rows that already have an eligibility status)
    reviewed_count = df['Eligibility'].notnull().sum()
    print(f"Initial reviewed rows: {reviewed_count}")  # Print count of reviewed rows

    # Filter rows without an eligibility status
    eligible_df = df[df['Eligibility'].isnull()].copy()
    eligible_df["original_index"] = eligible_df.index  # Preserve original index

    initialize_model()
    uncertainty_sampling()  # Select the first batch of rows
   
    # Ensure the model is initialized before predictions
    if model is None:
        messagebox.showerror("Model Error", "The model could not be initialized. Please check your setup.")
        return

    create_display_frame(selected_columns)

    # Button Section
    button_frame = Frame(root)
    Button(button_frame, text="Eligible", command=lambda: record_eligibility(True)).pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
    Button(button_frame, text="Not Eligible", command=lambda: record_eligibility(False)).pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
    button_frame.pack(pady=10)

    # Ensure `current_row_index` is valid
    if current_batch:
        current_row_index = current_batch[0]  # First row in the batch
        print(f"Starting Row Index: {current_row_index}")
        update_display_frame(current_row_index, selected_columns)
    else:
        current_row_index = None
        print("No rows available for initial labeling.")
        messagebox.showinfo("No Rows", "No rows available for initial labeling.")

    root.mainloop()

# Function to save the Excel file after each action
def save_excel(file_path):
    df.to_excel(file_path, index=False)
    #messagebox.showinfo("Saved", f"Progress saved to {file_path}") ## Pop-up window to show save message

# ------------------------------------------------------------------------------------
# Using Neural Networks (Convolutional & Recurrent - Hybrid Model)
# ------------------------------------------------------------------------------------

def build_model():
    """Creates and returns a new model."""
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # Binary classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    global tokenizer_fitted, val_accuracy, model
    labeled_df = df[df['Eligibility'].notnull()]
    if len(labeled_df) > 10:
        # Tokenize and pad sequences
        if not tokenizer_fitted:
            tokenizer.fit_on_texts(labeled_df['combined_text'])
            tokenizer_fitted = True
        
        X = tokenizer.texts_to_sequences(labeled_df['combined_text'])
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        y = labeled_df['Eligibility'].apply(lambda x: 1 if x == "Eligible" else 0).values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build and compile the model if it's not initialized
        if model is None:
            model = build_model()

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        # Evaluate the model
        loss, val_accuracy = model.evaluate(X_val, y_val)
        print("Reviewed rows:", len(labeled_df), end=", ")
        print(f"Validation Accuracy: {val_accuracy:.3f}")

        save_model(model, tokenizer)

# Store Model if Program Closed
def save_model(model, tokenizer, model_file='eligibility_model.keras', tokenizer_file='tokenizer.joblib'):
    # Save the model in the new Keras format
    model.save(model_file)
    # Save the tokenizer using joblib
    joblib.dump(tokenizer, tokenizer_file)

# Function to load the saved model from disk
def load_model(model_file='eligibility_model.keras', tokenizer_file='tokenizer.joblib'):
    # Load the model from the new Keras format
    model = tf.keras.models.load_model(model_file)
    # Load the tokenizer
    tokenizer = joblib.load(tokenizer_file)
    return model, tokenizer

# Initialize the model when the program starts
def initialize_model():
    global model, tokenizer, tokenizer_fitted
    try:
        # Attempt to load the model and tokenizer
        model, tokenizer = load_model()
        tokenizer_fitted = True
    except (FileNotFoundError, ValueError) as e:
        # Handle missing files or invalid files
        print(f"Model or tokenizer file not found. Initializing a new model and tokenizer. Error: {e}")
        model = build_model()
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer_fitted = False

def uncertainty_sampling():
    global current_batch, eligible_df, batch_size, current_row_index

    print(f"Eligible DataFrame Size: {len(eligible_df)}")
    if eligible_df.empty:
        print("No eligible rows remaining.")
        current_batch = []
        current_row_index = None
        return

    # Select batch: larger of 1% or 20 rows
    batch_size = max(len(eligible_df) // 100, 20)
    print(f"Batch Size: {batch_size}")

    # Sample rows safely
    try:
        sampled_rows = eligible_df.sample(n=min(batch_size, len(eligible_df))).index.tolist()
        current_batch = sampled_rows  # Use reindexed indices
        print(f"Selected Batch Indices: {current_batch}")
    except ValueError as e:
        print(f"Batch Sampling Error: {e}")
        current_batch = []

    # Set current_row_index to the first row in the batch
    if current_batch:
        current_row_index = current_batch[0]  # First valid row in the batch
        print(f"Starting Row Index: {current_row_index}")
    else:
        current_row_index = None
        print("Error: No rows available for batch selection.")

def run_model_in_background():
    global background_thread

    if background_thread and background_thread.is_alive():
        print("Background model update already in progress.")
        return

    # Define the thread's target function
    def update_model():
        print("Starting background model predictions...")
        train_model()  # Update the model with the latest labeled data
        print("Model updated successfully in the background.")

    # Start a new thread for model updates
    background_thread = threading.Thread(target=update_model)
    background_thread.start()

# ------------------------------------------------------------------------------------
# Prediction Steps after Modelling
# ------------------------------------------------------------------------------------

def auto_apply_eligibility():
    global eligible_df
    remaining_df = df[df['Eligibility'].isnull()].copy()
    #X_remaining = vectorizer.transform(remaining_df['combined_text'])
    sequences = tokenizer.texts_to_sequences(remaining_df['combined_text'])
    X_remaining = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predicted_probs = model.predict(X_remaining)
    
    # Assign eligibility: 1 if prob >= 0.5, otherwise 0
    remaining_df['Probabilities'] = predicted_probs[:,1] # Probability for 0 and 1 output (so take second column)
    remaining_df['Eligibility'] = (predicted_probs[:,1] >= 0.5).astype(int)

    df.update(remaining_df)

    # Save the updated DataFrame
    save_excel(excel_file_path)
    messagebox.showinfo("Completed", "The model has finished applying eligibility to all remaining rows.")
    review_uncertain_rows() # Now re-review assigned to see performance

def review_uncertain_rows():
    global df, uncertain_df
    labeled_df = df[df['Eligibility'].notnull()]
    if labeled_df.empty:
        messagebox.showinfo("No Labeled Data", "No labeled data available for uncertain row review.")
        return
    # Combine text fields for vectorization
    #X_unlabeled = vectorizer.transform(labeled_df['combined_text'])  # Vectorize all rows
    #probs = model.predict_proba(X_unlabeled)[:, 1]  # Get probabilities of being eligible
    sequences = tokenizer.texts_to_sequences(labeled_df['combined_text'])
    X_unlabeled = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    probs = model.predict(X_unlabeled)[:,  1]  # Assuming the output is the probability of class 1 (Eligible)
    uncertainty_mask = (probs >= 0.4) & (probs <= 0.6)
    uncertain_df = df[uncertainty_mask].copy()
    if not uncertain_df.empty:
        uncertainty_sampling(uncertain_df)  # Start reviewing uncertain rows
    else:
        messagebox.showinfo("Finished", "All rows have been labelled and no uncertain rows remain.")

# ------------------------------------------------------------------------------------
# Functionality of App Window
# ------------------------------------------------------------------------------------

# Global list to keep track of dynamically created widgets
dynamic_widgets = {}  # Global dictionary to hold references to ScrolledText widgets
dynamic_frame = None 

def create_display_frame(selected_columns):
    """
    Create the layout of widgets (labels and text areas) for the selected columns.
    This function is called only once after the columns are selected.
    """
    global dynamic_frame, dynamic_widgets

    # Create the dynamic frame only once
    if dynamic_frame is None:
        dynamic_frame = Frame(root)
        dynamic_frame.pack(padx=10, pady=10, fill=BOTH, expand=True)

    # Clear any existing widgets in the frame (if needed)
    for widget in dynamic_frame.winfo_children():
        widget.destroy()

    dynamic_widgets.clear()  # Clear any previous references

    # Create labels and ScrolledText widgets for each selected column
    for column in selected_columns:
        # Create a label for the column
        label = Label(dynamic_frame, text=column, font=("Arial", 12, "bold"))
        label.pack(side=TOP, fill=X, padx=10, pady=5)

        # Create a ScrolledText widget to display column data
        text_area = ScrolledText(dynamic_frame, wrap=WORD, height=5, width=50)
        text_area.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Store the widget in the dictionary for later updates
        dynamic_widgets[column] = text_area

def update_display_frame(row_index, selected_columns):
    """
    Update the content of the existing widgets with data from the current row.
    This function is called each time a new row is displayed.
    """
    for column in selected_columns:
        # Get the widget for the column
        text_area = dynamic_widgets.get(column)
        if text_area:
            # Clear the previous content
            text_area.delete(1.0, END)

            # Safely get the new content for the column
            column_value = eligible_df.iloc[row_index][column]
            if pd.isna(column_value):
                column_value = "No data available"

            # Insert the new content into the text area
            text_area.insert(END, column_value)


# Function to handle button click and update Excel immediately
val_accuracy = 0.0

def record_eligibility(is_eligible):
    global reviewed_count, current_row_index, rows_labeled_in_batch, current_batch

    print(f"Current Row Index: {current_row_index}")
    print(f"Current Batch: {current_batch}")

    # Validate `current_row_index`
    if current_row_index is None or current_row_index not in current_batch:
        print("Invalid row or row not in the current batch.")
        return

    # Record eligibility
    original_index = eligible_df.loc[current_row_index, 'original_index']
    df.loc[original_index, 'Eligibility'] = "1" if is_eligible else "0"
    reviewed_count += 1
    rows_labeled_in_batch += 1
    save_excel(excel_file_path)

    # Remove the labeled row from the batch and `eligible_df`
    print(f"Removing Row: {current_row_index}")
    current_batch.remove(current_row_index)
    eligible_df.drop(index=current_row_index, inplace=True)
    eligible_df.reset_index(drop=True, inplace=True)

    print(f"Remaining Eligible Rows: {len(eligible_df)}")
    print(f"Updated Batch: {current_batch}")

    # Select the next row or a new batch
    if current_batch:
        current_row_index = current_batch[0]  # Set to the next row in the batch
        print(f"Next Row Index: {current_row_index}")
    else:
        print("Batch completed. Selecting a new batch...")
        rows_labeled_in_batch = 0
        uncertainty_sampling()

    # Start background model update after half the batch
    #if rows_labeled_in_batch == batch_size // 2:
    if rows_labeled_in_batch % 10 == 0 and rows_labeled_in_batch != 0:
        run_model_in_background()

    # Update display
    if current_row_index is not None:
        update_display_frame(current_row_index, selected_columns)
    else:
        print("No rows left to label.")

# Load the Excel file
folder_path = r"C:\Users\je116\OneDrive - Imperial College London\PhD-wpca-je116\9. Additional Projects\Funding Awards\09FEB2024 - Imperial BRC Digital Health Trials\3. Survey\Advertising"
filename = "PYTHON_ALL_TRIAL_Results.xlsx"
excel_file_path = os.path.join(folder_path, filename)

print("Loading excel file....")
load_excel()


