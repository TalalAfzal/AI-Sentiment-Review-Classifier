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
