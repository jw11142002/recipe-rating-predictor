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


csv_file_path = 'RAW_interactions.csv'

# Initialize the list to store all interactions
allInteractions = []

# Parse CSV data from the file
with open(csv_file_path, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header

    for row in reader:
        # Combine multiline content for the 'review' field
        row[4] = row[4] + '\n'.join(row[5:])
        del row[5:]  # Remove extra columns if any

        # Convert 'rating' to int if needed
        row[3] = int(row[3])

        # Create a dictionary for each interaction and add it to the list
        interaction = {header[i]: row[i] for i in range(len(header))}
        allInteractions.append(interaction)

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

# Iterate through all interactions and add sentiment-related information
for interaction in allInteractions:
    # Combine multiline content for the 'review' field
    review = interaction['review']
    
    # Perform sentiment analysis
    sentiment_polarity, sentiment_label = analyze_sentiment(review)
    
    # Add sentiment-related information to the interaction dictionary
    interaction['sentiment_polarity'] = sentiment_polarity
    interaction['sentiment_label'] = sentiment_label

# Print the sentiment-related information for the first few interactions
for interaction in allInteractions[:5]:
    print(f"Review: {interaction['review']}")
    print(f"Sentiment Polarity: {interaction['sentiment_polarity']}")
    print(f"Sentiment Label: {interaction['sentiment_label']}")
    print("------")