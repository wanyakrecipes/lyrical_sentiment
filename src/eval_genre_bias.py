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
import anthropic

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
#SHOULD BE 250 PER SONG
song_lyrics_clean_sample_df = song_lyrics_clean_df.copy()
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('genre').apply(lambda x: x.sample(n=100, random_state=42) if len(x) > 30 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)

print("Predict genre using an gpt-4o " + str(len(song_lyrics_clean_sample_df)) + " samples...")

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
song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] = np.where(song_lyrics_clean_sample_df['predicted_genre_gpt_4o'] == 'indie rock','rock',song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])

#Confusion matrix
# Overall metrics
accuracy = accuracy_score(song_lyrics_clean_sample_df['genre'], song_lyrics_clean_sample_df['predicted_genre_gpt_4o'])
print(f"Overall Accuracy (gpt-4o): {accuracy:.3f}")
print(f"Total Songs: {len(song_lyrics_clean_sample_df)}")
        
# Detailed classification report
print(f"\nDetailed Metrics (gpt-4o):")
print(classification_report(song_lyrics_clean_sample_df['genre'], 
                            song_lyrics_clean_sample_df['predicted_genre_gpt_4o'], 
                            zero_division=0))

cm = confusion_matrix(song_lyrics_clean_sample_df['genre'], 
                      song_lyrics_clean_sample_df['predicted_genre_gpt_4o'], 
                      labels=['country','pop','r&b','rap','rock'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['country','pop','r&b','rap','rock'])

disp.plot() 
disp.ax_.set_title("Confusion Matrix for gpt-4o")

## Use a claude model here
# https://github.com/anthropics/anthropic-cookbook/blob/main/skills/classification/guide.ipynb
# https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts#how-to-use-system-prompts


load_dotenv()

client = anthropic.Anthropic()

def get_genre_from_lyrics_claude(lyrics, model = "claude-sonnet-4-20250514"):

    genre_options = ['rock', 'rap', 'pop', 'r&b','country']
    options_str = ", ".join(genre_options)

    prompt = f"""
                    <role>
                    You are an expert music critic.
                    </role>

                    <task>
                    Based on these lyrics, classify the song into one of these genres:{options_str}
                    </task>

                    <instruction>
                    Respond with only the genre name.
                    </instruction>


                    <lyrics>
                    \"\"\"
                    {lyrics}
                    \"\"\"
                    </lyrics>
                    """

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system = "You are an expert music critic",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0
    )
    return response.content[0].text.strip()

print("Predict genre using an claude sonnet 4 " + str(len(song_lyrics_clean_sample_df)) + " samples...")

song_lyrics_clean_sample_df["predicted_claude_4"] = song_lyrics_clean_sample_df["clean_lyrics"].apply(get_genre_from_lyrics_claude)

#Correct issues with data
id = '9b9e5a5f-39c5-4ea0-aa51-443e592980d8'
song_lyrics_clean_sample_df["predicted_claude_4"] = np.where(song_lyrics_clean_sample_df["track_id"] == id, 'Rock',song_lyrics_clean_sample_df["predicted_claude_4"])
song_lyrics_clean_sample_df['predicted_claude_4'] = np.where(song_lyrics_clean_sample_df['predicted_claude_4'] == 'Rock','rock',song_lyrics_clean_sample_df['predicted_claude_4'])
song_lyrics_clean_sample_df['predicted_claude_4'] = np.where(song_lyrics_clean_sample_df['predicted_claude_4'] == 'folk','pop',song_lyrics_clean_sample_df['predicted_claude_4'])

#Confusion matrix
# Overall metrics
accuracy = accuracy_score(song_lyrics_clean_sample_df['genre'], song_lyrics_clean_sample_df["predicted_claude_4"])
print(f"Overall Accuracy (Claude Sonnet 4): {accuracy:.3f}")
print(f"Total Songs: {len(song_lyrics_clean_sample_df)}")
        
# Detailed classification report
print(f"\nDetailed Metrics (Claude Sonnet 4):")
print(classification_report(song_lyrics_clean_sample_df['genre'], 
                            song_lyrics_clean_sample_df["predicted_claude_4"], 
                            zero_division=0))

cm = confusion_matrix(song_lyrics_clean_sample_df['genre'], song_lyrics_clean_sample_df["predicted_claude_4"], labels=['country','pop','r&b','rap','rock'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['country','pop','r&b','rap','rock'])


disp.plot() 
disp.ax_.set_title("Confusion Matrix for Claude Sonnet 4")



