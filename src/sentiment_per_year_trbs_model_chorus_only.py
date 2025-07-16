from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import bert_model_sentiment as bms
import re
import torch.nn.functional as F
import ast

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

#Update clean_lyrics_by_part as dict
song_lyrics_clean_df['clean_lyrics_by_part'] = song_lyrics_clean_df['clean_lyrics_by_part'].apply(ast.literal_eval)

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
print("Load BERT-TRBS model...")
tokenizer, model, labels = bms.load_model(device = device,
                                          model_name = "cardiffnlp/twitter-roberta-base-sentiment")


# Sentiment scoring function
def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    return {label: probs[i].item() for i, label in enumerate(labels)}

# Extract sentiment for chorus sections with their order
def chorus_sentiment_over_time(sections):
    chorus_scores = []
    for i, section in enumerate(sections):
        if re.search(r'chorus', section['part'], re.IGNORECASE):
            score = get_sentiment_score(section['lyrics'])
            score['section_index'] = i
            score['label'] = section['part']
            chorus_scores.append(score)
    return chorus_scores if chorus_scores else None

song_lyrics_clean_sample_df['chorus_sentiment_timeline'] = song_lyrics_clean_sample_df['clean_lyrics_by_part'].apply(chorus_sentiment_over_time)

