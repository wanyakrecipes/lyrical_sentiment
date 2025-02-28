# Generate a sample for sentiment analysis
import pandas as pd

song_lyrics_clean_df = pd.read_csv("song_lyrics_clean_df.csv")

# Desired total sample size
total_sample_size = 2000

# Calculate the number of rows in each category
genre_counts = song_lyrics_clean_df['tag'].value_counts()

# Calculate the sample size for each category
stratified_sample_sizes = (genre_counts / genre_counts.sum() * total_sample_size).round().astype(int)

# Perform stratified sampling
song_lyrics_clean_sample_df = song_lyrics_clean_df.groupby('tag').apply(lambda x: x.sample(stratified_sample_sizes.loc[x.name], random_state=1)).reset_index(drop=True)

#Save output to csv
song_lyrics_clean_sample_df.to_csv('song_lyrics_clean_sample.csv')