from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import bert_model_sentiment as trbs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Read and concaternate labelled data
song_lyrics_positive_labelled_df = pd.read_csv('../data/processed/song_lyrics_positive_labelled.csv')
song_lyrics_negative_labelled_df = pd.read_csv('../data/processed/song_lyrics_negative_labelled.csv')
song_lyrics_labelled_df = pd.concat([song_lyrics_positive_labelled_df,song_lyrics_negative_labelled_df])

### Load TRBS Model ####

# Automatically use MPS on Mac if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer once
print("Load TRBS model...")
tokenizer, model, labels = trbs.load_model(device = device)

print("Apply senitment over chunks...")
song_lyrics_labelled_df['prediction_trbs'] = song_lyrics_labelled_df.apply(lambda row: trbs.sentiment_over_chunks(lyrics = row['lyrics'],tokenizer=tokenizer,labels = labels, model = model, device= device), axis=1)

print("Extract binary prediction between positive and negative sentiment...")
song_lyrics_labelled_df['prediction_trbs_bin_sent_label'] = song_lyrics_labelled_df['prediction_trbs'].apply(lambda x: trbs.collapse_to_binary(x['scores']))

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