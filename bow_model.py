### CSE 158 Assignment 2: Bag-of-Words Model
import csv
from collections import defaultdict
import math
#import scipy.optimize
#from sklearn import svm
#import numpy
import string
import random
import string
#from sklearn import linear_model
from io import StringIO
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


train_file_path = 'clean_train.csv'
valid_file_path = 'clean_valid.csv'
test_file_path = 'clean_test.csv'

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
        # Combine multiline content for the 'review' field
        row[3] = row[3] + '\n'.join(row[4:])
        del row[4:]  # Remove extra columns if any

        # Convert 'rating' to int if needed
        row[2] = int(row[2])

        # Create a dictionary for each interaction and add it to the list
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
        # Combine multiline content for the 'review' field
        row[3] = row[3] + '\n'.join(row[4:])
        del row[4:]  # Remove extra columns if any

        # Convert 'rating' to int if needed
        row[2] = int(row[2])

        # Create a dictionary for each interaction and add it to the list
        interaction = {header[i]: row[i] for i in range(len(header))}
        interaction['sentiment_polarity'], interaction['sentiment_label'] = analyze_sentiment(interaction['review_text'])
        valid_interactions.append(interaction)

# Extract features (bag-of-words) and target variable for training set
X_train = [interaction['review_text'] for interaction in train_interactions]
y_train = [1 if int(interaction['is_5']) == 1 else 0 for interaction in train_interactions]

# Extract features (bag-of-words) and target variable for validation set
X_valid = [interaction['review_text'] for interaction in valid_interactions]
y_valid = [1 if int(interaction['is_5']) == 1 else 0 for interaction in valid_interactions]

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the reviews to create the bag-of-words representation for training set
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the reviews to create the bag-of-words representation for validation set
X_valid_bow = vectorizer.transform(X_valid)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_bow, y_train)

# Make predictions on the validation set
predictions = classifier.predict(X_valid_bow)

# Evaluate the accuracy
accuracy = accuracy_score(y_valid, predictions)
print(f"Accuracy on the validation set: {accuracy:.5f}")

test_file_path = 'clean_test.csv'
test_interactions = []

with open(test_file_path, newline='', encoding='utf-8') as test_file:
    reader = csv.reader(test_file)
    header = next(reader)  # Assuming the CSV file has a header

    for row in reader:
        # Combine multiline content for the 'review' field
        row[3] = row[3] + '\n'.join(row[4:])
        del row[4:]  # Remove extra columns if any

        # Convert 'rating' to int if needed
        row[2] = int(row[2])

        # Create a dictionary for each interaction and add it to the list
        interaction = {header[i]: row[i] for i in range(len(header))}
        interaction['sentiment_polarity'], interaction['sentiment_label'] = analyze_sentiment(interaction['review_text'])
        test_interactions.append(interaction)

# Extract features (bag-of-words) and target variable for test set
X_test = [interaction['review_text'] for interaction in test_interactions]
y_test = [1 if int(interaction['is_5']) == 1 else 0 for interaction in test_interactions]

# Transform the reviews to create the bag-of-words representation for test set
X_test_bow = vectorizer.transform(X_test)

# Make predictions on the test set
test_predictions = classifier.predict(X_test_bow)

# Evaluate the accuracy on the test set
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy on the test set: {test_accuracy:.5f}")