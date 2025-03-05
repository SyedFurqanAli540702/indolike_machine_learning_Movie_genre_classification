
# Movie Genre Classification Using NLP

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Data Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load Dataset
dataset = pd.read_csv('dataset/genre_classification.csv')
dataset['plot'] = dataset['plot'].astype(str).apply(preprocess_text)

# Encode Genre Labels
mlb = MultiLabelBinarizer()
dataset['genres'] = dataset['genres'].apply(lambda x: x.split('|'))
y = mlb.fit_transform(dataset['genres'])

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['plot'])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
logistic = LogisticRegression(max_iter=1000)
classifier = MultiOutputClassifier(logistic)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
