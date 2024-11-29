#Script to classify lyric sentiment using GPT 4.

import os
from dotenv import load_dotenv
import pandas as pd
import openai
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re

# Load the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded from .env file!")
else:
    print("Error: API key not found in .env file.")


# Read labelled data
song_lyrics_positive_labelled_df = pd.read_csv('song_lyrics_positive_labelled.csv')
song_lyrics_negative_labelled_df = pd.read_csv('song_lyrics_negative_labelled.csv')

song_lyrics_labelled_df = pd.concat([song_lyrics_positive_labelled_df,song_lyrics_negative_labelled_df])

# Function to classify text using ChatGPT
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
song_lyrics_labelled_df["predicted_label_chat_gpt"] = song_lyrics_labelled_df["lyrics"].apply(classify_lyric_sentiment)

#Confusion Matrix
accuracy = accuracy_score(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_chat_gpt'])
print("Chat GPT 4 results using unseen data")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_chat_gpt']))

# Print Confusion Matrix
cm = confusion_matrix(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_chat_gpt'], labels=['negative','positive'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['negative','positive'])

disp.plot()
plt.show()

song_lyrics_labelled_df.to_csv('song_lyrics_labelled_chat_gpt_pred_df.csv')
