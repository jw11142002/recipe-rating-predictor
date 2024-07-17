### CSE 158 Assignment 2: Bag-of-Words Model v2
import csv
from collections import defaultdict
import math
#import scipy.optimize
#from sklearn import svm
#import numpy
import string
import random
import string
import  pandas as pd
#from sklearn import linear_model
from io import StringIO
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def analyze_sentiment(review_text):
    blob = TextBlob(review_text)
    sentiment_polarity = blob.sentiment.polarity
    
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    return sentiment_polarity, sentiment_label

# Load training data
train_file_path = 'clean_train.csv'
train_interactions = []

with open(train_file_path, newline='', encoding='utf-8') as train_file:
    reader = csv.reader(train_file)
    header = next(reader)  # Assuming the CSV file has a header

    for row in reader:
        interaction = {header[i]: row[i] for i in range(len(header))}
        interaction['sentiment_polarity'], interaction['sentiment_label'] = analyze_sentiment(interaction['review_text'])
        train_interactions.append(interaction)

# Load validation data
valid_file_path = 'clean_valid.csv'
valid_interactions = []

with open(valid_file_path, newline='', encoding='utf-8') as valid_file:
    reader = csv.reader(valid_file)
    header = next(reader)  # Assuming the CSV file has a header

    for row in reader:
        interaction = {header[i]: row[i] for i in range(len(header))}
        interaction['sentiment_polarity'], interaction['sentiment_label'] = analyze_sentiment(interaction['review_text'])
        valid_interactions.append(interaction)

# Extract features (TF-IDF) and target variable for training set
X_train = [interaction['review_text'] for interaction in train_interactions]
y_train = [1 if int(interaction['is_5']) == 1 else 0 for interaction in train_interactions]

# Extract features (TF-IDF) and target variable for validation set
X_valid = [interaction['review_text'] for interaction in valid_interactions]
y_valid = [1 if int(interaction['is_5']) == 1 else 0 for interaction in valid_interactions]

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the reviews to create the TF-IDF representation for training set
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the reviews to create the TF-IDF representation for validation set
X_valid_tfidf = vectorizer.transform(X_valid)

# Train a logistic regression classifier
# classifier = LogisticRegression(max_iter=300)
# classifier.fit(X_train_tfidf, y_train)

# Make predictions on the validation set
# predictions = classifier.predict(X_valid_tfidf)

# # Evaluate the accuracy
# accuracy = accuracy_score(y_valid, predictions)

ridge_classifier = Ridge()
ridge_classifier.fit(X_train_tfidf, y_train)
predictions = ridge_classifier.predict(X_valid_tfidf)
binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
accuracy = accuracy_score(y_valid, binary_predictions)
print(f"Accuracy on the validation set using TF-IDF: {accuracy:.4f}")