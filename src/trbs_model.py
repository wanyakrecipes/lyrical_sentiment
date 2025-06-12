#TODO learn how to make a class with these functions
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


def load_model(device):

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    labels = ['negative', 'neutral', 'positive']

    return tokenizer, model, labels

#Function to classify lyrics
def classify_sentiment(text,labels,model,tokenizer,device):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()  # move to CPU for NumPy
    return dict(zip(labels, map(float, probs)))

#Function for chunking text in case lyrics are longer than 512 tokens
def chunk_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        token_chunk = tokens[i:i + max_tokens]
        text_chunk = tokenizer.convert_tokens_to_string(token_chunk)
        chunks.append(text_chunk)
    return chunks


def sentiment_over_chunks(lyrics, tokenizer,labels, model, device):
    
    chunks = chunk_text(lyrics, tokenizer, max_tokens=512)
    
    all_scores = [classify_sentiment(chunk,labels,model,tokenizer,device) for chunk in chunks]
    
    # Aggregate: average scores per label
    avg_scores = {label: np.mean([score[label] for score in all_scores]) for label in labels}
    
    # Add label with highest average score
    top_label = max(avg_scores, key=avg_scores.get)
    
    return {
        'label': top_label,
        'scores': avg_scores
    }


def collapse_to_binary(label_scores):
    if label_scores['positive'] > label_scores['negative']:
        return 'positive'
    else:
        return 'negative'
