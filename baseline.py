### CSE 158 Assignment 2: Baseline Model

## Setup

import pandas as pd

int_train_df = pd.read_csv("interactions_train.csv")
int_test_df = pd.read_csv("interactions_test.csv")

## Model

# creates binary attribute to indicate if 5-star review

int_cat_df = int_train_df.copy()
int_cat_df['is_5'] = int_cat_df['rating'].apply(lambda x: 1 if x == 5 else 0)

# finds proportion of 5-star reviews from training set

category_counts = int_cat_df.groupby('is_5').count().get('user_id')
not5_count = category_counts.iloc[0]
is5_count = category_counts.iloc[1]
total = category_counts.sum()
not5_prop = not5_count / total
is5_prop = is5_count / total

# finds model performance on test set

prediction_df = int_test_df.copy()
prediction_df['is_5'] = prediction_df['rating'].apply(lambda x: 1 if x == 5 else 0)
prediction_df['prediction'] = 1
prediction_df['correct'] = prediction_df.apply(lambda x: x['is_5'] == x['prediction'], axis=1)

results = prediction_df.groupby('correct').size() / len(prediction_df)