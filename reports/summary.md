# Summary of Analysis so far


## EDA

* Completed EDA analysis to ensure I extract the right data for sentiment analysis. Further info:
    * [notebooks/exploratory_data_analysis.ipynb](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/notebooks/exploratory_data_analysis.ipynb)
    * [src/clean_data.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/clean_data.py)


## Sentiment Analyis

* Conducted sentiment analysis across time using a sample of lyrics from each year. I used BERT and gpt-40-mini to evaluate sentiment of the complete song. The analysis suggest lyrics have become more negative. It also may suggest that older songs in the dataset (1950s) may be bias towards more positive sounding tracks. For further info:
    * [reports/average_positive_sentiment_over_time_bert_trbs.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_positive_sentiment_over_time_bert_trbs.png)
    * [reports/average_sentiment_over_time_gpt_4o_mini.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_sentiment_over_time_gpt_4o_mini.png)
    * [src/sentiment_per_year_gpt_4o.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_gpt_4o.py)
    * [src/sentiment_per_year_trbs_model.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_trbs_model.py)
* Conducted sentiment analysis of hooks / common lines from lyrics. Used gpt-4o-mini to extract the top three most frequent lines from a song. A line is defined as sequence of more han two words on a new line. The analysis suggest lyrics have become more negative. For further info:
    * [reports/average_phrase_sentiment_over_time_gpt_4o_mini.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_phrase_sentiment_over_time_gpt_4o_mini.png)
    * [src/chorus_sentiment_per_year_gpt_4o.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/chorus_sentiment_per_year_gpt_4o.py)


## AI Safety & Bias
* Exploring how effective LLMs are classifying genre based on music lyrics. Exploring if there is bias in the way LLMs catergorise creative content. Claude sonnet 4 seems to perform better than gpt-4o. However, both struggle to classify pop and R&B. This may be because the genre labels in this dataset are too high level. For further info:
    * [src/eval_genre_bias.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/eval_genre_bias.py)
    * [reports/gpt_4o_genre_classification_confusion_matrix.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/gpt_4o_genre_classification_confusion_matrix.png)
    * [reports/confusion_matrix_genre_class_claude_sonnet_4.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_genre_class_claude_sonnet_4.png)
    * [reports/confusion_matrix_genre_class_gpt4o](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_genre_class_gpt4o.png)
