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
#Limit to 30 samples per year given the time it will take to process.
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('year').apply(lambda x: x.sample(n=30, random_state=42) if len(x) > 30 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)
#song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.sample(n=100)

print("Number of samples: " + str(len(song_lyrics_clean_sample_df)))

#Extract top three common phrases from lyrics
#Use uncleaned lyrics, as that contains new lines etc.
print("Extract top three common phrases using gpt-4o-mini...")
song_lyrics_clean_sample_df['common_lyrics_gpt_4o'] = song_lyrics_clean_sample_df['lyrics'].apply(gpt_4o.get_common_phrases_from_lyrics)

print("Extract sentiment from top three common phrases using gpt-4o-mini...")
song_lyrics_clean_sample_df['common_lyrics_sentiment_gpt_4o'] = song_lyrics_clean_sample_df['common_lyrics_gpt_4o'].apply(gpt_4o.get_phrase_sentiment_scores)

print("Parse sentiment from xml......")
song_lyrics_clean_sample_df['common_lyrics_sentiment_gpt_4o'] = song_lyrics_clean_sample_df['common_lyrics_sentiment_gpt_4o'].apply(gpt_4o.parse_phrase_sentiment_scores)

print("Calculate average sentiment...")
song_lyrics_clean_sample_df['common_lyrics_average_sentiment_gpt_4o'] = song_lyrics_clean_sample_df['common_lyrics_sentiment_gpt_4o'].apply(
    lambda lst: round(sum(score for _, score in lst) / len(lst),3) if lst else None
)

#generate graph of sentiment over time
print("Generate sentiment over time...")
positive_sentiment_per_year_df = song_lyrics_clean_sample_df.groupby('year')['common_lyrics_average_sentiment_gpt_4o'].mean().plot(title="TBC (gtp-4o-mini)",
                                                                                                                                   ylabel="positive sentiment")


