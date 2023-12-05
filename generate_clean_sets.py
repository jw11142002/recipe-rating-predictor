import pandas as pd

interactions_raw_df = pd.read_csv("RAW_interactions.csv")

def produce_clean_csv(filepath, new_filename):
    # reads in csv, assumes is one of the splits
    set_df = pd.read_csv(filepath)

    # creates new column for category feature (binary), prunes unused columns
    set_df['is_5'] = set_df['rating'].apply(lambda x: 1 if x == 5 else 0)
    set_df = set_df.drop(columns = ['date', 'rating', 'u', 'i'])

    # creates new column for review text feature (str)
    set_df['review_text'] = interactions_raw_df.set_index(['user_id', 'recipe_id']) \
        .loc[list(zip(set_df['user_id'], set_df['recipe_id']))]['review'].values
    
    # produces new file with specified name
    set_df.to_csv(new_filename, index=False)
    return

splits_dict = {
    'train': "interactions_train.csv",
    'valid': "interactions_validation.csv",
    'test': "interactions_test.csv"
}

for filename, filepath in splits_dict.items():
    produce_clean_csv(filepath, f"clean_{filename}.csv")
