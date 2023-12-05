import pandas as pd

from generate_clean_sets import split_dfs

combined_valid_test_df = pd.concat(split_dfs[1], split_dfs[2], ignore_index = True)

def generate_random_set(n, filename):
    # max n is 19478 (combined rows of valid + test)
    # produces csv file, returns DataFrame
    random_set_df = combined_valid_test_df.sample(n)
    random_set_df.to_csv(filename, index=False)
    return random_set_df