### CSE 158 Assignment 2: Recipe Feature EDA

## Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generate_clean_sets import split_dfs

clean_train_df = split_dfs[0]
recipes_raw_df = pd.read_csv('RAW_recipes.csv')

# proportion of 5-star ratings by recipe_id

rid_props = clean_train_df.groupby('recipe_id')['is_5'].mean()
rid_props_df = clean_train_df.groupby('recipe_id').mean()[['is_5']]

# exploring minutes feature

rid_minutes_df = recipes_raw_df[['id', 'minutes']].copy().set_index('id')
rid_minutes_df = rid_minutes_df.merge(rid_props_df, right_index = True, left_index= True)

# plotting recipes within 24 hours wrt average 5-star proportion
day_recipes_df = rid_minutes_df[rid_minutes_df['minutes'] <= 1440]
day_recipes_df['bin_cat'] = pd.cut(day_recipes_df['minutes'], bins = list(range(0, 1441, 30)), labels = False, right = False)
day_bin_avgs = day_recipes_df.groupby('bin_cat').mean()[['is_5']]
day_bin_avgs.reset_index().plot(x = 'bin_cat', y = 'is_5')

# plotting recipes within 2 hours wrt average 5-star proportion

two_hour_recipes_df = rid_minutes_df[rid_minutes_df['minutes'] <= 120]
two_hour_recipes_df['bin_cat'] = pd.cut(two_hour_recipes_df['minutes'], bins = list(range(0, 121, 12)), labels = False, right = False)
two_hour_bin_avgs = two_hour_recipes_df.groupby('bin_cat').mean()[['is_5']]
two_hour_bin_avgs.reset_index().plot(x = 'bin_cat', y = 'is_5')

# exploring steps feature

rid_steps_df = recipes_raw_df[['id', 'n_steps']].copy().set_index('id')
rid_steps_df = rid_steps_df.merge(rid_props_df, right_index = True, left_index= True)

# plotting n_steps wrt average 5-star proportion

step_avgs = rid_steps_df.groupby('n_steps').mean()
step_avgs.reset_index().plot(x = 'n_steps', y = 'is_5')

# exploring ingredients feature

rid_ingredients_df = recipes_raw_df[['id', 'n_ingredients']].copy().set_index('id')
rid_ingredients_df = rid_ingredients_df.merge(rid_props_df, right_index = True, left_index= True)

# plotting n_ingredients wrt average 5-star proportion

ingredient_avgs = rid_ingredients_df.groupby('n_ingredients').mean()
ingredient_avgs.reset_index().plot(x = 'n_ingredients', y = 'is_5')

# exploring calories feature

rid_calories_df = recipes_raw_df[['id', 'nutrition']].copy().set_index('id')
rid_calories_df['calories'] = rid_calories_df['nutrition'].apply(lambda x: float(x.split(",")[0][1:]))
rid_calories_df = rid_calories_df.merge(rid_props_df, right_index = True, left_index= True)

# plotting calories within 1000 calories wrt average 5-star proportion

thousand_cal_df = rid_calories_df[rid_calories_df['calories'] <= 1000]
thousand_cal_df['Calories (units of 100 calories)'] = pd.cut(two_hour_recipes_df['minutes'], bins = list(range(0, 1001, 10)), labels = False, right = False)
thousand_cal_bin_avgs = thousand_cal_df.groupby('Calories (units of 100 calories)').mean()[['is_5']]
thousand_cal_bin_avgs.reset_index().plot(x = 'Calories (units of 100 calories)', y = 'is_5')