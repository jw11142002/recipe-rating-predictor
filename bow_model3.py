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
from sklearn.model_selection import GridSearchCV


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

# Load test data
test_file_path = 'clean_test.csv'
test_interactions = []

with open(test_file_path, newline='', encoding='utf-8') as test_file:
    reader = csv.reader(test_file)
    header = next(reader)  # Assuming the CSV file has a header

    for row in reader:
        interaction = {header[i]: row[i] for i in range(len(header))}
        interaction['sentiment_polarity'], interaction['sentiment_label'] = analyze_sentiment(interaction['review_text'])
        test_interactions.append(interaction)

# Extract features (TF-IDF) and target variable for training set
X_train = [interaction['review_text'] for interaction in train_interactions]
y_train = [1 if int(interaction['is_5']) == 1 else 0 for interaction in train_interactions]

# Extract features (TF-IDF) and target variable for validation set
X_valid = [interaction['review_text'] for interaction in valid_interactions]
y_valid = [1 if int(interaction['is_5']) == 1 else 0 for interaction in valid_interactions]

# Extract features (TF-IDF) for test set
X_test = [interaction['review_text'] for interaction in test_interactions]
y_test = [1 if int(interaction['is_5']) == 1 else 0 for interaction in test_interactions]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the reviews to create the TF-IDF representation for training set
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the reviews to create the TF-IDF representation for validation set
X_valid_tfidf = vectorizer.transform(X_valid)

# Transform the reviews to create the TF-IDF representation for test set
X_test_tfidf = vectorizer.transform(X_test)

# Train a Ridge regression model with hyperparameter tuning using GridSearchCV
ridge_classifier = Ridge()
param_grid = {'alpha': [0.1, 1, 10]}  # Example hyperparameter values

grid_search = GridSearchCV(ridge_classifier, param_grid, cv=3)
grid_search.fit(X_valid_tfidf, y_valid)

# Get the best hyperparameters from the grid search
best_alpha = grid_search.best_params_['alpha']

# Train a Ridge regression model with the best hyperparameters using the entire training set
final_ridge_classifier = Ridge(alpha=best_alpha)
final_ridge_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
test_predictions = final_ridge_classifier.predict(X_test_tfidf)

# Convert predicted probabilities to binary predictions
binary_test_predictions = [1 if pred >= 0.5 else 0 for pred in test_predictions]

# Evaluate the accuracy on the test set
accuracy_test = accuracy_score(y_test, binary_test_predictions)
print(f"Accuracy on the test set: {accuracy_test:.5f}")