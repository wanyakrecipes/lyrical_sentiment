#Import libraries and packages
import pandas as pd
import re

#Read data using chunks

print("Read data...")

file_path = '../data/raw/song_lyrics.csv'

chunk_size = 100000

chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame if needed
print("Concact chunks into data frame...")
song_lyrics_full_df = pd.concat(chunks, ignore_index=True)

#Data Preprocessing using EDA analysis.

#Clean text
def clean_lyrics(text):
    
    #Remove text between brackets - this contains META information on verse and chorus - maybe do this seperately.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove newline and tab characters
    text = re.sub(r'[\n\t]', ' ', text)
    
    # Remove special characters and digits (optional, depending on use case)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

#TODO - have lyrics by section, then in totality.
#Enable chorus sentiment v song sentiment over time.

#Undertake data cleaning
print("Prepare data frame for data cleaning...")
song_lyrics_clean_df = song_lyrics_full_df.copy()

#Filter data for english lyrics
print("Filter for English lyrics...")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['language'] == 'en')]

#Filter data to remove artists containing "Genius"
print("Filter to remove artists containing the word 'Genius'")
song_lyrics_clean_df  = song_lyrics_clean_df [~song_lyrics_clean_df ['artist'].str.contains('Genius', case=False, na=False)]

#Filter for data between 1880 and 2022
print("Filter for songs released between 1880 and 2022")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['year'] >= 1880) & (song_lyrics_clean_df['year'] <= 2022)]

#Filter data for misc genre
print("Remove songs under the misc genre")
song_lyrics_clean_df = song_lyrics_clean_df[~(song_lyrics_clean_df['tag'] == 'misc')]

#Filter this population for songs with views more than 95 percentile
print("Keep songs with in 95th percentile of views")
percentile_95 = song_lyrics_clean_df['views'].quantile(0.95)
print(f"The 95th percentile of views is: {percentile_95}")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['views'] >= percentile_95)]

#Clean text
print("clean song lyrics")
song_lyrics_clean_df['lyrics'] = song_lyrics_clean_df['lyrics'].apply(clean_lyrics)

# Drop unecessary columns
print("drop columns that are not required")
song_lyrics_clean_df = song_lyrics_clean_df.drop(columns=['id','language_cld3','language_ft'])

#Other considerations - the word remix etc.

# save to csv
song_lyrics_clean_df.to_csv("../data/processed/song_lyrics_clean_df.csv")


