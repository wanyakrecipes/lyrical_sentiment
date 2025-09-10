#Quick analysis of how effectively LLMs classiy genre based on the music lyrics
#AI Safety research question - is there bias in the way LLMs categorise createive content
#Can we also use one shot to improve accruacy?

#Import libraires
import pandas as pd
import numpy as np
import gpt_4o_prompts as gpt_4o
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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

#Adjust rb for R&B
song_lyrics_clean_df['genre'] = np.where(song_lyrics_clean_df['genre'] == 'rb','r&b',song_lyrics_clean_df['genre'])

#Check genres in correct format
# rap is most represented - makes sense as it's rap genius 
popular_genre_df = song_lyrics_clean_df.groupby('genre').agg({'track_id':'count'}).reset_index()
popular_genre_df = popular_genre_df.sort_values(by = 'track_id',ascending = False)
popular_genre_df = popular_genre_df.rename(columns = {"track_id" : "number of tracks"})
print(popular_genre_df) 

#Generate a sample - need to make sure decades are represneted but do that later
song_lyrics_clean_sample_df = song_lyrics_clean_df.copy()
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('genre').apply(lambda x: x.sample(n=250, random_state=42) if len(x) > 30 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)

print("Predict genre using an LLM with " + str(len(song_lyrics_clean_sample_df)) + " samples...")

#Predict genre from lyrics
song_lyrics_clean_sample_df["predicted_genre_gpt_4o"] = song_lyrics_clean_sample_df["clean_lyrics"].apply(gpt_4o.get_genre_from_lyrics)

#Clean lyrics so R&B = r&B
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'R&B','r&b',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'musical','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'dancehall','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'gospel','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'Christian','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'reggae','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'folk','pop',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])


#Confusion matrix
# Overall metrics
accuracy = accuracy_score(song_lyrics_clean_sample_df['genre'], song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
print(f"Overall Accuracy: {accuracy:.3f}")
print(f"Total Songs: {len(song_lyrics_clean_sample_df)}")
        
# Detailed classification report
print(f"\nDetailed Metrics:")
print(classification_report(song_lyrics_clean_sample_df['genre'], 
                            song_lyrics_clean_sample_df['predicted_genre_gpt_4o'], 
                            zero_division=0))

cm = confusion_matrix(song_lyrics_clean_sample_df['genre'], song_lyrics_clean_sample_df['predicted_genre_gpt_4o'], labels=['country','pop','r&b','rap','rock'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['country','pop','r&b','rap','rock'])

disp.plot()