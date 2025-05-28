# 🧠 AI Eligibility Review Tool

**AI_EligibilityReview.py** is an intelligent, semi-automated GUI application for reviewing eligibility of entries in clinical trials or similar tabular datasets. It uses an active learning approach with a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) model to assist with human labelling and gradually take over as model confidence increases.

---

## ✨ Features

- ✅ **Manual Eligibility Labelling** with GUI support via Tkinter  
- 🤖 **Active Learning**: Auto-suggests rows for review based on model uncertainty  
- 🧠 **CNN + Bi-LSTM Model**: Learns from text to predict eligibility  
- 🧼 **Preprocessing**: Cleans and normalizes text (stopwords, punctuation, lemmatization)  
- 🗃️ **Column Selector**: Dynamically select relevant columns for model input  
- 💾 **Progress Save**: Updates Excel sheet live with reviewed eligibility  
- 🔄 **Background Training**: Trains model in a separate thread to keep the UI responsive  
- 📊 **Auto-Apply Mode**: When accuracy exceeds threshold, labels remaining rows automatically  
- 📈 **Uncertainty Re-Review**: Identify and manually re-review uncertain predictions  

---

## 📂 Requirements

- Python 3.x  
- Required packages:
  - `tensorflow`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `nltk`
  - `bs4` (BeautifulSoup4)
  - `joblib`
  - `tkinter`

You can install the required packages using:

```bash
pip install tensorflow pandas numpy scikit-learn nltk beautifulsoup4 joblib
```

## Additional Python Packages Required
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## 🙌 Credits

- Developed by Jack Elkes.
- Built with insights and coding support from [ChatGPT](https://openai.com/chatgpt).
