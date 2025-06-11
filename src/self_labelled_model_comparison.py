#Script to classify lyric sentiment using GPT 4.

import os
from dotenv import load_dotenv
import pandas as pd
import openai
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re
import torch
import roberta_model as roberta

# Read and concaternate labelled data
song_lyrics_positive_labelled_df = pd.read_csv('../data/processed/song_lyrics_positive_labelled.csv')
song_lyrics_negative_labelled_df = pd.read_csv('../data/processed/song_lyrics_negative_labelled.csv')

song_lyrics_labelled_df = pd.concat([song_lyrics_positive_labelled_df,song_lyrics_negative_labelled_df])


##### gpt-4o-mini performance #####

# Load the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded from .env file!")
else:
    print("Error: API key not found in .env file.")


# Function to classify text using ChatGPT
def classify_lyric_sentiment_gpt(lyrics):
   
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
song_lyrics_labelled_df["predicted_label_gpt4o"] = song_lyrics_labelled_df["lyrics"].apply(classify_lyric_sentiment_gpt)

#Confusion Matrix
accuracy = accuracy_score(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_gpt4o'])
print("GPT-4o results using unseen data")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_gpt4o']))

# Print Confusion Matrix
cm = confusion_matrix(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_gpt4o'], labels=['negative','positive'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['negative','positive'])

disp.plot()
plt.show()

### Load TRBS Model ####

# Automatically use MPS on Mac if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer once
print("Load BERT model...")
tokenizer, model, labels = roberta.load_trbs_model(device = device)

print("Apply senitment over chunks...")
song_lyrics_labelled_df['prediction_trbs'] = song_lyrics_labelled_df.apply(lambda row: roberta.sentiment_over_chunks(lyrics = row['lyrics'],tokenizer=tokenizer,labels = labels, model = model, device= device), axis=1)

print("Extract binary prediction between positive and negative sentiment...")
song_lyrics_labelled_df['prediction_trbs_bin_sent_label'] = song_lyrics_labelled_df['prediction_trbs'].apply(lambda x: roberta.collapse_to_binary(x['scores']))

#Confusion Matrix
accuracy_trbs = accuracy_score(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['prediction_trbs_bin_sent_label'])
print("TRBS Model results using unseen data")
print(f"Accuracy: {accuracy_trbs}")
print("Classification Report:")
print(classification_report(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['prediction_trbs_bin_sent_label']))

# Print Confusion Matrix
cm = confusion_matrix(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['prediction_trbs_bin_sent_label'], labels=['negative','positive'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['negative','positive'])

disp.plot()
plt.show()