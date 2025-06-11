from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import roberta_model as roberta

#Added to ensure that it uses the apple M1 cores.
#https://github.com/jeffheaton/app_deep_learning/blob/main/install/pytorch-install-aug-2023.ipynb
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()

# Automatically use MPS on Mac if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

#Read clean dataset
file_path = '../data/processed/song_lyrics_clean_df.csv'

chunk_size = 100000

chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame if needed
print("Concact chunks into data frame...")
song_lyrics_clean_df = pd.concat(chunks, ignore_index=True)

#song_lyrics_clean_df = song_lyrics_clean_df.head(100) #for tetsing.

print("Generate sample...")

#Generate sample of lyrics - only
song_lyrics_clean_sample_df = song_lyrics_clean_df.copy()

#Songs from 1950 onwards
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df[(song_lyrics_clean_sample_df['year'] >= 1950)]

#Generate sample based on year
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('year').apply(lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)

# Load model and tokenizer once
print("Load BERT model...")
tokenizer, model, labels = roberta.load_trbs_model(device = device)

print("Apply senitment over chunks...")
song_lyrics_clean_sample_df['sentiment'] = song_lyrics_clean_sample_df.apply(lambda row: roberta.sentiment_over_chunks(lyrics = row['lyrics'],tokenizer=tokenizer,labels = labels, model = model, device= device), axis=1)

print("Extract sentiment labels into new columns...")
song_lyrics_clean_sample_df['sentiment_label'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['label'])
song_lyrics_clean_sample_df['sentiment_positive'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['positive'])
song_lyrics_clean_sample_df['sentiment_negative'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['negative'])
song_lyrics_clean_sample_df['sentiment_neutral'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['neutral'])

#generate graph of sentiment over time
positive_sentiment_per_year_df = song_lyrics_clean_sample_df.groupby('year')['sentiment_positive'].mean().plot(title="Average Positive Sentiment Over Time")
