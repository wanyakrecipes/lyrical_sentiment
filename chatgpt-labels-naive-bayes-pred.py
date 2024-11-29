import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Read data ready for sampling
song_lyrics_clean_sample_chat_gpt_labels_df = pd.read_csv('song_lyrics_clean_sample_chat_gpt_labels_df.csv')

# Data Preprocesing - Balance the dataset between postive and negative rows
song_lyrics_naive_bayes_neg_df = song_lyrics_clean_sample_chat_gpt_labels_df[(song_lyrics_clean_sample_chat_gpt_labels_df['predicted_label_chat_gpt'] == "negative")]
song_lyrics_naive_bayes_neg_df = song_lyrics_naive_bayes_neg_df.sample(n=703,random_state=42)
song_lyrics_naive_bayes_pos_df = song_lyrics_clean_sample_chat_gpt_labels_df[(song_lyrics_clean_sample_chat_gpt_labels_df['predicted_label_chat_gpt'] == "positive")]
song_lyrics_naive_bayes_df = pd.concat([song_lyrics_naive_bayes_pos_df,song_lyrics_naive_bayes_neg_df])

# Convert the text data into TF-IDF features
tfidf = TfidfVectorizer()

#Feature vector
X = tfidf.fit_transform(song_lyrics_naive_bayes_df['lyrics'])

#Print vector X
print("Number of samples: ", X.shape[0])
print("Number of words in lyric corpus: ", X.shape[1])

#Target vector
y = song_lyrics_naive_bayes_df['predicted_label_chat_gpt']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=song_lyrics_naive_bayes_df['predicted_label_chat_gpt'])

# Initialize the Naive Bayes classifier
nb = MultinomialNB()

# Train the model
nb.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the classification report
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=nb.classes_)

disp.plot()
plt.show()