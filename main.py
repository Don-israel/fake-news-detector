import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import nltk  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score 

df = pd.read_csv("Fake.csv")
df.head()
df.dropna(inplace=True)  
df = df[['title', 'text', 'label']]  # Select important columns
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})  # Convert labels to 0 and 1
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import string  

nltk.download('stopwords')  
nltk.download('punkt')  

def preprocess(text):  
    text = text.lower()  # Convert to lowercase  
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)  # Tokenization  
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)  

df['text'] = df['text'].apply(preprocess)

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import string  

nltk.download('stopwords')  
nltk.download('punkt')  

def preprocess(text):  
    text = text.lower()  # Convert to lowercase  
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation  
    words = word_tokenize(text)  # Tokenization  
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords  
    return ' '.join(words)  

df['text'] = df['text'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(df['text'])  
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

model = MultinomialNB()  
model.fit(X_train, y_train)  

y_pred = model.predict(X_test)  

accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy * 100:.2f}%")  

def predict_news(news_text):  
    processed_text = preprocess(news_text)  
    transformed_text = vectorizer.transform([processed_text])  
    prediction = model.predict(transformed_text)[0]  
    return "REAL" if prediction == 1 else "FAKE"  

# Example  
news = "Breaking news: Scientists discover a new planet that can sustain life!"
print(predict_news(news))
