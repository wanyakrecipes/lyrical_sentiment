import pandas as pd
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import gpt_4o_prompts as gpt_4o

# Load the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded from .env file!")
else:
    print("Error: API key not found in .env file.")

#Read clean dataset
file_path = '../data/processed/song_lyrics_clean_df.csv'

chunk_size = 100000

chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame if needed
print("Concact chunks into data frame...")
song_lyrics_clean_df = pd.concat(chunks, ignore_index=True)
#song_lyrics_clean_df = song_lyrics_clean_df.head(10) #for tetsing.

print("Generate sample...")

#Generate sample of lyrics - only
song_lyrics_clean_sample_df = song_lyrics_clean_df.copy()

#Songs from 1950 onwards
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df[(song_lyrics_clean_sample_df['year'] >= 1950)]

#Generate sample based on year
#TODO need to put a limit on number of songs that can be represented by a single artist, in the set and in a given year.
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('year').apply(lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)

print("Number of samples: " + str(len(song_lyrics_clean_sample_df)))

print("Apply sentiment with confidence using gpt-4o...")
song_lyrics_clean_sample_df['gpt_4o_score'] = song_lyrics_clean_sample_df['lyrics'].apply(gpt_4o.get_lyrics_sentiment_score)

# Make sure sentiment is numeric
song_lyrics_clean_sample_df['gpt_4o_score'] = pd.to_numeric(song_lyrics_clean_sample_df['gpt_4o_score'], errors='coerce')

# Group by year
yearly_sentiment = song_lyrics_clean_sample_df.groupby('year')['gpt_4o_score'].mean().plot(title="Average sentiment decreases over time [gpt-40-mini]",
                                                                                           ylabel="Sentiment")

