import pandas as pd

#Script to classify lyric sentiment using GPT 4.

import os
from dotenv import load_dotenv
import pandas as pd
import openai
import matplotlib.pyplot as plt


# Load the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded from .env file!")
else:
    print("Error: API key not found in .env file.")

#Read data ready for sampling
song_lyrics_clean_sample_df = pd.read_csv('song_lyrics_clean_sample.csv')

# Function to classify text using ChatGPT - move this to a function.
def classify_lyric_sentiment(lyrics):
   
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  #Could even try o1-mini to see if there is a difference
            messages=[
                {"role": "system", "content": "Classify the sentiment of the following music lyrics as 'positive' or 'negative'. The output should be a single word: 'positive' or 'negative'. When analysing the lyrics, please take into account the context of the whole text."},
                {"role": "user", "content": f"Classify the sentiment of the following music lyrics: {lyrics}"}
            ],
            max_tokens=10,  # Limit response tokens since the output is short
            temperature=0  # Reduce randomness for consistent results
        )

        classification = response.choices[0].message.content
        return classification
    except Exception as e:
        return f"Error: {str(e)}"

# Apply the classification function to the DataFrame
song_lyrics_clean_sample_df["predicted_label_chat_gpt"] = song_lyrics_clean_sample_df["lyrics"].apply(classify_lyric_sentiment)

#Write results
song_lyrics_clean_sample_df.to_csv('song_lyrics_clean_sample_chat_gpt_labels_df.csv')

#Box plot with results?