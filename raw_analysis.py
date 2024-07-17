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

count = 0
for i in allInteractions:
    if i['rating'] == 5:
        count += 1

print(count / len(allInteractions))