# AI-Sentiment-Review-Classifier
AI Sentiment Review Classifier is a machine learning project that analyzes product reviews and classifies them as Positive, Negative, or Neutral.
It leverages classical ML and pretrained NLP models to deliver accurate and insightful sentiment predictions.

## 📌 Objective

To build a sentiment analysis pipeline that:
- Preprocesses customer reviews,
- Trains ML models (Naive Bayes, Logistic Regression) using TF-IDF,
- Applies pretrained sentiment analyzers (VADER, TextBlob, Flair),
- Evaluates and compares the performance of all approaches.

---

## 📁 Dataset

**Source**: Datafiniti Amazon Product Reviews  
- Columns used: `reviews.text`, `reviews.rating`
## 🛠️ Pipeline

1. **Text Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering, lemmatization using NLTK

2. **TF-IDF Vectorization**
   - Transforms cleaned reviews into numeric feature vectors

3. **Label Creation**
   - Labels generated from review ratings:
     - `>= 4` → positive
     - `== 3` → neutral
     - `<= 2` → negative

4. **Model Training**
   - **Naive Bayes**
   - **Logistic Regression**
  
   - ## Setup Instructions

### 1. Clone or Download the Repository
Place all files in a single directory. Ensure the following files are present:
- `app_streamlit.py` (the Streamlit app)
- `requirements.txt` (Python dependencies)
- `1429_1.csv/1429_1.csv` (the product reviews CSV file)

### 2. Install Python Dependencies
Open a terminal in the project directory and run:

```
pip install -r requirements.txt
```

If you encounter issues running `streamlit`, you may need to use the full path to the executable, e.g.:

### 3. Run the App
Start the Streamlit app with:

```
streamlit run app_streamlit.py
```

### 4. Use the App
- Open the local URL shown in your terminal (usually http://localhost:8501).
- Enter a product review in the text area.
- Select a model (VADER, TextBlob, or Naive Bayes).
- Click "Analyze Sentiment" to see the result.

## Notes
- The Naive Bayes model is trained on startup for demo purposes. For production, consider saving/loading the model and vectorizer.
- The app uses NLTK and TextBlob for additional sentiment analysis options.
- If you see warnings about NLTK data, it will be downloaded automatically.

## Example Reviews
- "This product is amazing! It exceeded my expectations."
- "The quality is okay, but the delivery was slow."
- "I am very disappointed with this purchase. It broke after one use."

## License
This project is for educational/demo purposes. 
