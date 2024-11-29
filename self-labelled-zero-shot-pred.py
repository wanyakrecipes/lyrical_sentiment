import torch
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Added to ensure that it uses the apple M1 cores.
#https://github.com/jeffheaton/app_deep_learning/blob/main/install/pytorch-install-aug-2023.ipynb
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the zero-shot classification pipeline - could look at different models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=device)

# Read labelled data
song_lyrics_positive_labelled_df = pd.read_csv('song_lyrics_positive_labelled.csv')
song_lyrics_negative_labelled_df = pd.read_csv('song_lyrics_negative_labelled.csv')

song_lyrics_labelled_df = pd.concat([song_lyrics_positive_labelled_df,song_lyrics_negative_labelled_df])


#Sentiment labels
sentiment_labels = ["positive","negative"]

# Function to classify text and store results
def classify_lyrics(lyrics, candidate_labels):
    result = classifier(lyrics, candidate_labels)
    return result

# Apply classification to each row and store results
song_lyrics_labelled_df['classification'] = song_lyrics_labelled_df['lyrics'].apply(lambda x: classify_lyrics(x, sentiment_labels))

# Extract the scores for each label
for label in sentiment_labels:
    song_lyrics_labelled_df[label] = song_lyrics_labelled_df['classification'].apply(lambda x: x['scores'][x['labels'].index(label)])

# Extract the predicted label and score
song_lyrics_labelled_df['predicted_label_zero_shot'] = song_lyrics_labelled_df['classification'].apply(lambda x: x['labels'][0])
song_lyrics_labelled_df['predicted_score_zero_shot'] = song_lyrics_labelled_df['classification'].apply(lambda x: x['scores'][0])

# Drop the intermediate 'Classification' column
song_lyrics_labelled_df = song_lyrics_labelled_df.drop(columns=['classification'])

#Confusion Matrix
accuracy = accuracy_score(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_zero_shot'])
print("Zero shot model using BART model results using unseen data")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_zero_shot']))

# Print Confusion Matrix
cm = confusion_matrix(song_lyrics_labelled_df['actual_label'], song_lyrics_labelled_df['predicted_label_zero_shot'], labels=['negative','positive'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['negative','positive'])

disp.plot()
plt.show()

