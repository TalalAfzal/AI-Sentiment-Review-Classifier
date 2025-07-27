import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import joblib
import numpy as np

# Download NLTK resources if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if text is None:
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Load trained model and vectorizer ---
# For demo, we retrain on startup (in production, use joblib to load pre-trained model/vectorizer)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load a small sample for quick demo (in production, use full data and save model/vectorizer)
df = pd.read_csv('1429_1.csv/1429_1.csv')
if 'reviews.text' in df.columns:
    df['reviews.text'] = df['reviews.text'].fillna('')
    df['cleaned_text'] = df['reviews.text'].apply(clean_text)
else:
    st.error('Column reviews.text not found in DataFrame')
    st.stop()

def label_sentiment(rating):
    try:
        rating = float(rating)
    except:
        return 'neutral'
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df_filtered = df[df['cleaned_text'].str.strip() != ''].copy()
df_filtered['label'] = df_filtered['reviews.rating'].apply(label_sentiment)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_filtered['cleaned_text'])
y = df_filtered['label']
nb = MultinomialNB()
nb.fit(X, y)

# --- Streamlit UI ---
st.title('Product Review Sentiment Analysis Demo')
st.write('Enter a product review below and select a sentiment analysis model:')

review = st.text_area('Product Review', height=150)
model_choice = st.selectbox('Choose Model', ['VADER', 'TextBlob', 'Naive Bayes'])

if st.button('Analyze Sentiment'):
    if not review.strip():
        st.warning('Please enter a review.')
    else:
        if model_choice == 'VADER':
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(review)['compound']
            if score >= 0.05:
                sentiment = 'positive'
            elif score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            st.info(f'VADER Sentiment: **{sentiment}** (score: {score:.2f})')
        elif model_choice == 'TextBlob':
            polarity = TextBlob(review).sentiment.polarity
            if polarity > 0.05:
                sentiment = 'positive'
            elif polarity < -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            st.info(f'TextBlob Sentiment: **{sentiment}** (polarity: {polarity:.2f})')
        elif model_choice == 'Naive Bayes':
            cleaned = clean_text(review)
            X_input = vectorizer.transform([cleaned])
            pred = nb.predict(X_input)[0]
            st.info(f'Naive Bayes Sentiment: **{pred}**')
        else:
            st.error('Unknown model selected.')

st.markdown('---')
st.write('Demo by [Your Name].') 