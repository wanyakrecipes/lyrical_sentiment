from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import seaborn as sns

#Added to ensure that it uses the apple M1 cores.
#https://github.com/jeffheaton/app_deep_learning/blob/main/install/pytorch-install-aug-2023.ipynb
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()

# Automatically use MPS on Mac if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

#Read clean dataset
file_path = '../data/song_lyrics_clean_df.csv'

chunk_size = 100000

chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame if needed
print("Concact chunks into data frame...")
song_lyrics_clean_df = pd.concat(chunks, ignore_index=True)

print("Generate sample...")

#Generate sample of lyrics - only
song_lyrics_clean_sample_df = song_lyrics_clean_df.copy()

#Songs from 1950 onwards
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df[(song_lyrics_clean_sample_df['year'] >= 1950)]

#Generate sample based on year
song_lyrics_clean_sample_df= song_lyrics_clean_sample_df.groupby('year').apply(lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x)
song_lyrics_clean_sample_df = song_lyrics_clean_sample_df.reset_index(drop=True)

#Function for chunking text in case lyrics are longer than 512 tokens
def chunk_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        token_chunk = tokens[i:i + max_tokens]
        text_chunk = tokenizer.convert_tokens_to_string(token_chunk)
        chunks.append(text_chunk)
    return chunks

# Load model and tokenizer once
print("Load BERT model...")

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
labels = ['negative', 'neutral', 'positive']

#Function to classify lyrics
def classify_sentiment(text):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()  # move to CPU for NumPy
    return dict(zip(labels, map(float, probs)))


def sentiment_over_chunks(lyrics):
    
    chunks = chunk_text(lyrics, tokenizer, max_tokens=512)
    
    all_scores = [classify_sentiment(chunk) for chunk in chunks]
    
    # Aggregate: average scores per label
    avg_scores = {label: np.mean([score[label] for score in all_scores]) for label in labels}
    
    # Add label with highest average score
    top_label = max(avg_scores, key=avg_scores.get)
    
    return {
        'label': top_label,
        'scores': avg_scores
    }

print("Apply senitment over chunks...")
song_lyrics_clean_sample_df['sentiment'] = song_lyrics_clean_sample_df['lyrics'].apply(sentiment_over_chunks)

print("Extract sentiment labels into new columns...")
song_lyrics_clean_sample_df['sentiment_label'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['label'])
song_lyrics_clean_sample_df['sentiment_positive'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['positive'])
song_lyrics_clean_sample_df['sentiment_negative'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['negative'])
song_lyrics_clean_sample_df['sentiment_neutral'] = song_lyrics_clean_sample_df['sentiment'].apply(lambda x: x['scores']['neutral'])

#generate graph of sentiment over time
positive_sentiment_per_year_df = song_lyrics_clean_sample_df.groupby('year')['sentiment_positive'].mean().plot(title="Average Positive Sentiment Over Time")
