# Music lyrics

This repo contains work on researching the effectiveness of text classification methods for the sentiment analysis of music lyrics.

The project will be using the [Genius Song Lyric](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data?select=song_lyrics.csv) dataset from Kaggle.

For information on work so far, please see [reports/summary.md](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/summary.md)


# Ideas
* Generate emotion classification using bhadresh-savani/bert-base-uncased-emotion. Do so on chorus and all lyrics for comparison.
* Sentiment analysis over time using all lyrics:
    * gpt-40-mini - need to fine tune prompt [DONE]
    * cardiffnlp/twitter-roberta-base-sentiment [DONE]
* Sentiment analysis over time (chorus only):
    * gpt-40-mini - need to fine tune prompt
    * cardiffnlp/twitter-roberta-base-sentiment
        * noticed that the // line space for chorus doesn't work, as some tracks the chrous is on the following line.
        * may need to section songs in a different way.

# Analysis Inspired by BSE Course.

## Day 1
* On data pre-processing, have a dataset with just the raw lyrics. Create another dataset where the lyrics are split into sections. Use the tags [Verse, chorus etc.] to identify sections where applicable. Can identify sections using page breaks. Look at chunking in Day 4.1. [DONE]
* Part 5 - Generate a dictionary of words based on key labels. Perhaps use the labels joy, sadness, anger, fear, surprise, disgust, neutral from bhadresh-savani/bert-base-uncased-emotion

## Day 2
* Generate a classifier which ranks whether a song genre is "rock" (1) or "not rock" (0) based on the lyrics. Compare TF-IDF and ELMO [Workshop2.1 and 2.2]

## Day 3
* Compare fine tuned BERT model to an untuned BERT model using the examples based on your music library. Compare this with GPT and other specific bert models (twitter-roberta-base-sentiment and bert-base-uncased-emotion). DO this with all lyrics and chorus only.

## Day 4
* Generate a RAG / LLM system to query music lyrics. This would involved embedding the artist name, genre etc. as tags [see GPT Chat]

## Day 5
* Generate a ReAct tool where I ask the LLM to provide more context to music lyrics.


