#Import libraries and packages
import pandas as pd
import re
import uuid

#Read data using chunks

print("Read lyrics data...")

file_path = '../data/raw/song_lyrics.csv'

chunk_size = 100000

percentile = 0.75
percentile_text = "75th"

chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame if needed
print("Concact chunks into data frame...")
song_lyrics_df = pd.concat(chunks, ignore_index=True)

#Data Preprocessing using EDA analysis.

#Clean text
def clean_lyrics(text):
    
    #Remove text between brackets - this contains META information on verse and chorus - maybe do this seperately.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove newline and tab characters
    #text = re.sub(r'[\n\t]', ' ', text) REMOVE THIS
    
    # Remove special characters and digits (optional, depending on use case)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def clean_lyrics_compact(text):

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # remove new line and tab characters
    text = re.sub(r'[\n\t]', ' ', text)

    return text

### Clean Lyrics ###

#Undertake data cleaning
print("Prepare data frame for data cleaning...")
# song_lyrics_clean_df = song_lyrics_full_df.copy()

# song_lyrics_clean_df = song_lyrics_clean_df.head(100)

#Filter data for english lyrics
print("Filter for English lyrics...")
song_lyrics_df = song_lyrics_df[(song_lyrics_df['language'] == 'en')]

#Filter for data between 1950 and 2022
print("Filter for songs released between 1880 and 2022")
song_lyrics_df = song_lyrics_df[(song_lyrics_df['year'] >= 1950) & (song_lyrics_df['year'] <= 2022)]

#Filter data to remove artists containing "Genius"
print("Filter to remove artists containing the word 'Genius'")
song_lyrics_df  = song_lyrics_df [~song_lyrics_df ['artist'].str.contains('Genius', case=False, na=False)]

#Rename tag as genre
print("update tag as genre")
song_lyrics_df = song_lyrics_df.rename(columns = {"tag" : "genre"})

#Filter data for misc genre
print("Remove songs under the misc genre")
song_lyrics_df = song_lyrics_df[~(song_lyrics_df['genre'] == 'misc')]

#Filter this population for songs with views more than 95 percentile
print("Keep songs with in 95th percentile of views")
percentile_views = song_lyrics_df['views'].quantile(percentile)
print(f"The {percentile} of views is: {percentile_views}")
song_lyrics_df = song_lyrics_df[(song_lyrics_df['views'] >= percentile_views)]

#Clean text - there should really be a clean_lyrics column, not overwrite lyrics.
print(len(song_lyrics_df))
print("clean song lyrics")
song_lyrics_df['clean_lyrics'] = song_lyrics_df['lyrics'].apply(clean_lyrics)
song_lyrics_df['clean_lyrics_compact'] = song_lyrics_df['clean_lyrics'].apply(clean_lyrics_compact)

# Drop unecessary columns
print("drop columns that are not required")
song_lyrics_df = song_lyrics_df.drop(columns=['id','language_cld3','language_ft'])

#Add track id - add this at the end?
print("add track_id")
song_lyrics_df['track_id'] = [str(uuid.uuid4()) for _ in range(len(song_lyrics_df))]

# Column ordering
song_lyrics_df = song_lyrics_df[['track_id','artist','features','title','year','genre','views','language','lyrics','clean_lyrics','clean_lyrics_compact']]

# print(len(song_lyrics_df))

# save to csv
song_lyrics_df.to_csv(f"../data/processed/song_lyrics_clean_{percentile_text}_df.csv",index=False)