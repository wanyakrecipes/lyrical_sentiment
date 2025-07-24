#Import libraries and packages
import pandas as pd
import re
import uuid

#Read data using chunks

print("Read lyrics data...")

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

### Clean Lyrics ###

#Undertake data cleaning
print("Prepare data frame for data cleaning...")
song_lyrics_clean_df = song_lyrics_full_df.copy()

#Sample dataset
# print("Sample tracks...")
# song_lyrics_clean_df= song_lyrics_clean_df.sample(n=3000, random_state=42)

# print(len(song_lyrics_clean_df))

#Rename tag as genre
print("update tag as genre")
song_lyrics_clean_df = song_lyrics_clean_df.rename(columns = {"tag" : "genre"})

#Add track id
print("add track_id")
song_lyrics_clean_df['track_id'] = [str(uuid.uuid4()) for _ in range(len(song_lyrics_clean_df))]

#Filter data for english lyrics
print("Filter for English lyrics...")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['language'] == 'en')]

#Filter data to remove artists containing "Genius"
print("Filter to remove artists containing the word 'Genius'")
song_lyrics_clean_df  = song_lyrics_clean_df [~song_lyrics_clean_df ['artist'].str.contains('Genius', case=False, na=False)]

#Filter for data between 1950 and 2022
print("Filter for songs released between 1880 and 2022")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['year'] >= 1950) & (song_lyrics_clean_df['year'] <= 2022)]

#Filter data for misc genre
print("Remove songs under the misc genre")
song_lyrics_clean_df = song_lyrics_clean_df[~(song_lyrics_clean_df['genre'] == 'misc')]

#Filter this population for songs with views more than 95 percentile
print("Keep songs with in 95th percentile of views")
percentile_95 = song_lyrics_clean_df['views'].quantile(0.95)
print(f"The 95th percentile of views is: {percentile_95}")
song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['views'] >= percentile_95)]

#Clean text - there should really be a clean_lyrics column, not overwrite lyrics.
print(len(song_lyrics_clean_df))
print("clean song lyrics")
song_lyrics_clean_df['clean_lyrics'] = song_lyrics_clean_df['lyrics'].apply(clean_lyrics)

# Drop unecessary columns
print("drop columns that are not required")
song_lyrics_clean_df = song_lyrics_clean_df.drop(columns=['id','language_cld3','language_ft'])

#Other considerations
# the word remix etc. - albeit if it's a popular remix that may be okay.

###Clean lyrics by part ###
# Issue - double new line sometimes works - but not all tracks have double new lines between sections
# One way would be to search for songs where the word "Chorus" or "Refrain" is labbled
# Then search for those only. Tha way, at least we have a clearly labellled sampled
# Other ways would be to use an LLM to extract phrases that are repeated as a proxy for a chorus
# Chunk data into different parts based on page breaks, and then extract the section name.
# Only apply this to the songs with > 95 percentile, as this is what we're interested in

# #Songs from 1950 onwards
# song_lyrics_clean_df = song_lyrics_clean_df[(song_lyrics_clean_df['year'] >= 1950)]

# #Generate sample based on year
# song_lyrics_clean_df = song_lyrics_clean_df.groupby('year').apply(lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x)
# song_lyrics_clean_df = song_lyrics_clean_df.reset_index(drop=True)

# label_pattern = r'^\[(.*?)\]' #Check if part label is between square brackets.

# def chunk_and_label(lyrics):
#     chunks = lyrics.split('\n\n')  # Step 1: chunk by double newlines
    
#     sections = []
#     for chunk in chunks:
#         # Step 2: extract section label if exists e.g [Verse 1]
#         match = re.match(label_pattern, chunk)
#         if match:
#             part = match.group(1).strip()
#             # Remove label from chunk text to keep only lyrics
#             text = re.sub(label_pattern, '', chunk, count=1).strip()
#         else:
#             part = 'unknown'
#             text = chunk.strip()
        
#         sections.append({'part': part, 'lyrics': text})
    
#     return sections

# print("Chunk and label song parts..")

# song_lyrics_clean_df['clean_lyrics_by_part'] = song_lyrics_clean_df['lyrics'].apply(chunk_and_label)

# def clean_sections(sections):
#     # sections is a list of dicts: [{'section': ..., 'text': ...}, ...]
#     for chunk in sections:
#         chunk['lyrics'] = clean_lyrics(chunk['lyrics'])
#     return sections

# print("Clean song lyrics by part")
# song_lyrics_clean_df['clean_lyrics_by_part'] = song_lyrics_clean_df['clean_lyrics_by_part'].apply(clean_sections)

# # Display the result
# for sec in song_lyrics_clean_df['clean_lyrics_by_part'].iloc[31]:
#     print(f"Part: {sec['part']}")
#     print(f"Lyrics:\n{sec['lyrics']}")
#     print('---')

# #Column ordering
song_lyrics_clean_df = song_lyrics_clean_df[['track_id','artist','features','title','year','genre','views','language','lyrics','clean_lyrics']]

# print(len(song_lyrics_clean_df))

# save to csv
song_lyrics_clean_df.to_csv("../data/processed/song_lyrics_clean_df.csv",index=False)