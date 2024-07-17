### CSE 158 Assignment 2: Sentiment Analysis Model
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


csv_file_path = 'clean_train.csv'

allInteractions = []

# Parse CSV data from the file
with open(csv_file_path, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header

    for row in reader:
        # Combine multiline content for the 'review' field
        row[3] = row[3] + '\n'.join(row[4:])
        del row[4:]  # Remove extra columns if any

        # Convert 'rating' to int if needed
        row[2] = int(row[2])

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

medianSentiment = 0.2
sentiments = []
correct = 0
total = 0
for interaction in allInteractions:
    # Combine multiline content for the 'review' field
    review = interaction['review_text']
    
    # Perform sentiment analysis
    sentiment_polarity, sentiment_label = analyze_sentiment(review)
    
    #Add sentiment-related information to the interaction dictionary
    interaction['sentiment_polarity'] = sentiment_polarity
    interaction['sentiment_label'] = sentiment_label
    #sentiments.append(sentiment_polarity)
    if sentiment_polarity >= medianSentiment and interaction['is_5'] == 1:
        correct += 1
    elif sentiment_polarity < medianSentiment and interaction['is_5'] == 0:
        correct += 1


print("Correct: " + str(correct))
print("Total: " + str(len(allInteractions)))
print("Accuracy on Validation Set: " + str(correct / len(allInteractions)))

# print("Correct: " + str(correct))
# print("Total: " + str(total))
# print("Accuracy on Validation Set: " + str(correct / total))

# sentiments.sort(reverse=True)
# medianSentiment = sentiments[math.floor(len(sentiments) * (1 - 0.7209))]

#print(medianSentiment)