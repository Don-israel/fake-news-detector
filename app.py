import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Article Text:")

if st.button("Check News"):
    transformed_text = vectorizer.transform([user_input])
    prediction = model.predict(transformed_text)[0]
    result = "✅ REAL" if prediction == 1 else "❌ FAKE"
    st.subheader(f"Prediction: {result}")

import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

from newspaper import Article

def fetch_latest_news(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Example: Fetch news from CNN
news_text = fetch_latest_news("https://edition.cnn.com/some-news-article")
print(news_text)

from difflib import SequenceMatcher

def is_similar(news1, news2):
    similarity = SequenceMatcher(None, news1, news2).ratio()
    return similarity > 0.7  # 70% similarity threshold

# Example: Compare user's news with CNN news
user_news = "Breaking: New planet discovered with alien life!"
trusted_news = fetch_latest_news("https://cnn.com/some-news")

if is_similar(user_news, trusted_news):
    print("This news is likely REAL")
else:
    print("This news might be FAKE")
