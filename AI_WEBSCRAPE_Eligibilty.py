import os 
import pandas as pd
import numpy as np
import random
import re
from bs4 import BeautifulSoup
from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
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

# Initialize global variables (Logistic Regression Model)
#vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') # If combining into one long string (otherwise separate vectorizers)
#vectorizer_fitted = False  # New flag to track if vectorizer is fitted
#title_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#summary_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#intervention_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

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
        # Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        text_cleaned = soup.get_text()  # Extract text from any HTML

        # Further cleaning (e.g., remove punctuation, lowercasing, etc.)
        text_cleaned = re.sub(r'[^\w\s]', '', text_cleaned)  # Remove punctuation
        text_cleaned = text.lower().strip()  # Lowercase and remove extra spaces

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words] 

    # Stemming or Lemmatization (choose one or both)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    #words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]

    # Handle special characters (e.g., emojis)
    words = [word for word in words if re.match(r'[a-zA-Z0-9]+', word)]
    # Join the words back into a string
    normalized_text = ' '.join(words)
    return normalized_text

# Function to load Excel file and shuffle rows without eligibility status
def load_excel(file_path):
    global df, eligible_df, excel_file_path, reviewed_count, current_row_index
    excel_file_path = file_path
    df = pd.read_excel(file_path)

    df['Eligibility'] = df.get('Eligibility', None)  # Create Eligibility column if not already present
    df['Eligibility'] = df['Eligibility'].astype(object)  # Ensure the column can hold strings

    # Apply cleaning to relevant columns
    df['Study Title'] = df['Study Title'].apply(clean_text)
    df['Brief Summary'] = df['Brief Summary'].apply(clean_text)
    df['Intervention Details'] = df['Intervention Details'].apply(clean_text)
    df['combined_text'] = df[['Study Title', 'Brief Summary', 'Intervention Details']].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Calculate reviewed count (rows that already have an eligibility status)
    reviewed_count = df['Eligibility'].notnull().sum()
    print(f"Initial reviewed rows: {reviewed_count}")  # Print count of reviewed rows

    # Filter out rows that don't have an eligibility status
    eligible_df = df[df['Eligibility'].isnull()].copy()  # Select only rows without eligibility status
    eligible_df.loc[:, 'original_index'] = eligible_df.index  # Store the original index

    initialize_model()

    # Check if there are eligible rows to review
    if not eligible_df.empty:
        current_row_index = random.choice(eligible_df.index) # Select a random row index from eligible_df
        train_model()
        display_row(current_row_index)  # Display the randomly selected row
    else:
        current_row_index = None  # Set to None when there are no eligible rows
        messagebox.showinfo("Completed", "All rows have an eligibility status.")
        review_uncertain_rows()  # Optionally review uncertain rows

# Function to save the Excel file after each action
def save_excel(file_path):
    df.to_excel(file_path, index=False)
    #messagebox.showinfo("Saved", f"Progress saved to {file_path}") ## Pop-up window to show save message

# ------------------------------------------------------------------------------------
# Train Model (Logistic Regression)
# ------------------------------------------------------------------------------------

# def train_model():   
#     global vectorizer, vectorizer_fitted, val_accuracy  
#     labeled_df = df[df['Eligibility'].notnull()]
#     print("Reviewed rows:", len(labeled_df), end=", ")
#     if len(labeled_df) > 0:
#         if not vectorizer_fitted: # Only fit Vectorizer once, else reuse.
#             # Vectorize each text component separately
#             #title_vector = title_vectorizer.fit_transform(labeled_df['Study Title'].fillna(''))
#             #summary_vector = summary_vectorizer.fit_transform(labeled_df['Brief Summary'].fillna(''))
#             #intervention_vector = intervention_vectorizer.fit_transform(labeled_df['Intervention Details'].fillna(''))
#             #X = np.hstack((title_vector.toarray(), summary_vector.toarray(), intervention_vector.toarray()))
#             X = vectorizer.fit_transform(labeled_df['combined_text'])
#             vectorizer_fitted = True
#         else:
#             X = vectorizer.transform(labeled_df['combined_text'])
#         y = labeled_df['Eligibility'].apply(lambda x: 1 if x == "Eligible" else 0)

#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.3, random_state=42
#         )
#         model.fit(X_train, y_train)
        
#         # Evaluate on training data
#         y_train_pred = model.predict(X_train)
#         train_accuracy = (y_train_pred == y_train).mean()
#         print(f"Training Accuracy: {train_accuracy:.3f}", end=", ")
        
#         # Evaluate on validation data
#         y_val_pred = model.predict(X_val)
#         val_accuracy = (y_val_pred == y_val).mean()
#         print(f"Validation Accuracy: {val_accuracy:.3f}")

#         # Train the model
#         #model.fit(X, y)
#         save_model(model, vectorizer)
#         #print(f"Training data shape: {X.shape}")
#         #print(f"Training labels distribution:\n{y.value_counts()}")
#         #print(f"Class distribution:\n{y.value_counts()}")

# Function to save the trained model to disk
# def save_model(model, vectorizer, model_file='eligibility_model.joblib', vectorizer_file='vectorizer.joblib'):
#     joblib.dump(model, model_file)
#     joblib.dump(vectorizer, vectorizer_file)
#     #print(f"Model and vectorizer saved to {model_file} and {vectorizer_file}")

# # Function to load the saved model from disk
# def load_model(model_file='eligibility_model.joblib', vectorizer_file='vectorizer.joblib'):
#     model = joblib.load(model_file)
#     vectorizer = joblib.load(vectorizer_file)
#     print(f"Model and vectorizer loaded from {model_file} and {vectorizer_file}")
#     return model, vectorizer

# # Load the model when the program starts
# def initialize_model():
#     global model, vectorizer, vectorizer_fitted
#     try:
#         model, vectorizer = load_model()
#         vectorizer_fitted = True
#     except FileNotFoundError:
#         model = LogisticRegression()
#         vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#         vectorizer_fitted = False
#         print("No saved model found. A new model and vectorizer will be created.")

# Sample rows with high uncertainty (i.e. close to 0.5)
# def uncertainty_sampling(dataset):
#     global uncertain_df, current_row_index
#     if not dataset.empty:
#         # Prepare data for uncertainty sampling
#         #title_vector = title_vectorizer.transform(dataset['Study Title'].fillna(''))
#         #summary_vector = summary_vectorizer.transform(dataset['Brief Summary'].fillna(''))
#         #intervention_vector = intervention_vectorizer.transform(dataset['Intervention Details'].fillna(''))

#         # Combine vectors for prediction
#         X_unlabeled = vectorizer.transform(dataset['combined_text'])
#         probs = model.predict_proba(X_unlabeled)[:, 1]  # Get probabilities of being eligible
#         uncertainty = np.abs(probs - 0.5)  # Uncertainty is highest when probability is close to 0.5

#         # Select the rows with the highest uncertainty
#         uncertain_indices = uncertainty.argsort()[-10:]  # Select top 10 uncertain rows
#         uncertain_rows = dataset.iloc[uncertain_indices]
#         uncertain_rows = uncertain_rows[uncertain_rows['Eligibility'].isnull()]

#         if not uncertain_rows.empty:
#             # Randomly sample one row from the uncertain rows
#             selected_row = uncertain_rows.sample(n=1)
#             current_row_index = selected_row.index[0]
#             display_row(current_row_index)
#         else:
#             messagebox.showinfo("Finished", "All uncertain rows have been reviewed.")
#     else:
#         messagebox.showinfo("Finished", "All uncertain rows have been reviewed.")

# ------------------------------------------------------------------------------------
# Using Neural Networks (Convolutional & Recurrent - Hybrid Model)
# ------------------------------------------------------------------------------------

def train_model():
    global tokenizer_fitted, val_accuracy, model
    labeled_df = df[df['Eligibility'].notnull()]
    if len(labeled_df) > 0:
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
            model = Sequential()
            model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
            model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
            model.add(MaxPooling1D(pool_size=4))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))  # Binary classification (eligible or not)

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
        model, tokenizer = load_model()
        tokenizer_fitted = True
    except FileNotFoundError:
        model = None  # Model will be initialized during training if not found
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer_fitted = False
        print("No saved model found. A new model and tokenizer will be created.")

def uncertainty_sampling(dataset):
    global uncertain_df, current_row_index
    if not dataset.empty:
        # Prepare data for uncertainty sampling

        # Tokenize and pad the sequences for the input text
        sequences = tokenizer.texts_to_sequences(dataset['combined_text'])
        X_unlabeled = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # Get the predicted probabilities from the model
        probs = model.predict(X_unlabeled)[:, 0]  # Assuming the output is the probability of class 1 (Eligible)

        # Uncertainty is highest when the probability is close to 0.5
        uncertainty = np.abs(probs - 0.5)

        # Select the rows with the highest uncertainty
        uncertain_indices = uncertainty.argsort()[-10:]  # Select top 10 uncertain rows
        uncertain_rows = dataset.iloc[uncertain_indices]
        uncertain_rows = uncertain_rows[uncertain_rows['Eligibility'].isnull()]

        if not uncertain_rows.empty:
            # Randomly sample one row from the uncertain rows
            selected_row = uncertain_rows.sample(n=1)
            current_row_index = selected_row.index[0]
            display_row(current_row_index)
        else:
            messagebox.showinfo("Finished", "All uncertain rows have been reviewed.")
    else:
        messagebox.showinfo("Finished", "All uncertain rows have been reviewed.")

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

# Function to display the current random row's data
def display_row(row_index):
    study_title_text.delete(1.0, END)
    summary_text.delete(1.0, END)
    intervention_text.delete(1.0, END)
    study_title_text.insert(END, eligible_df.loc[row_index, 'Study Title'])
    summary_text.insert(END, eligible_df.loc[row_index, 'Brief Summary'])
    intervention_text.insert(END, eligible_df.loc[row_index, 'Intervention Details'])

# Function to handle button click and update Excel immediately
def record_eligibility(is_eligible):
    global reviewed_count, model_can_take_over, val_accuracy  
    # Ensure current_row is an index in eligible_df
    if current_row_index is None or current_row_index >= len(eligible_df):
        print("Current Row Index:", current_row_index)
        print("Eligibile Dataset Length:", len(eligible_df))
        messagebox.showinfo("Error", "No current row to record eligibility.")
        return
    original_df_index = eligible_df.loc[current_row_index, 'original_index']
    df.loc[original_df_index, 'Eligibility'] = "Eligible" if is_eligible else "Not Eligible"
    reviewed_count += 1
    #print(f"Reviewed count after this action: {reviewed_count}")  # Debugging line
    save_excel(excel_file_path)

    #eligible_df.drop(eligible_df.index[current_row_index], inplace=True)
    eligible_df.drop(current_row_index, inplace=True)
    eligible_df.reset_index(drop=True, inplace=True)
    
    # Check if model can take over
    if (reviewed_count >= min_manual_reviews) & (val_accuracy >= accuracy_value):
        model_can_take_over = True
        auto_apply_eligibility()    
        messagebox.showinfo("Finished", "The model has applied predictions to the remaining rows.")
    else:  # Move to the next row
        if not eligible_df.empty:
            train_model()
            uncertainty_sampling(eligible_df)  # Call uncertainty_sampling instead of resetting to 0
        else:
            messagebox.showinfo("End", "You have reviewed all eligible rows.")
            review_uncertain_rows() 

# Create the GUI
root = Tk()
root.title("Eligibility Review")

# Create a PanedWindow widget that will hold the adjustable sections
main_pane = PanedWindow(root, orient=VERTICAL)
main_pane.pack(fill=BOTH, expand=True)

# Study Title Section
title_pane = PanedWindow(main_pane, orient=VERTICAL)
Label(title_pane, text="Study Title:").pack(side=TOP, fill=X, padx=10, pady=5)
study_title_text = ScrolledText(title_pane, wrap=WORD, height=5)
study_title_text.pack(fill=BOTH, expand=True, padx=10, pady=5)
main_pane.add(title_pane)

# Brief Summary Section
summary_pane = PanedWindow(main_pane, orient=VERTICAL)
Label(summary_pane, text="Brief Summary:").pack(side=TOP, fill=X, padx=10, pady=5)
summary_text = ScrolledText(summary_pane, wrap=WORD, height=10)
summary_text.pack(fill=BOTH, expand=True, padx=10, pady=5)
main_pane.add(summary_pane)

# Intervention Details Section
intervention_pane = PanedWindow(main_pane, orient=VERTICAL)
Label(intervention_pane, text="Intervention Details:").pack(side=TOP, fill=X, padx=10, pady=5)
intervention_text = ScrolledText(intervention_pane, wrap=WORD, height=10)
intervention_text.pack(fill=BOTH, expand=True, padx=10, pady=5)
main_pane.add(intervention_pane)

# Button Section
button_frame = Frame(root)
Button(button_frame, text="Eligible", command=lambda: record_eligibility(True)).pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
Button(button_frame, text="Not Eligible", command=lambda: record_eligibility(False)).pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
main_pane.add(button_frame)

# Load the Excel file
folder_path = r"C:\Users\je116\OneDrive - Imperial College London\PhD-wpca-je116\9. Additional Projects\Funding Awards\09FEB2024 - Imperial BRC Digital Health Trials\3. Survey\Advertising"
filename = "PYTHON_ALL_TRIAL_Results.xlsx"
excel_file_path = os.path.join(folder_path, filename)
load_excel(excel_file_path)

# Start the Tkinter main loop
root.mainloop()