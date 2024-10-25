import pandas as pd

# Ganti 'dataset.csv' dengan nama file dataset kamu
df = pd.read_csv('csv/data_tweet.csv')


sample_df = df.sample(n=30, random_state=42)  # random_state untuk reproducibility


sample_df.to_csv('csv/data_tweet30.csv', index=False)

