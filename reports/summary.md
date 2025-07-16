# Summary of Analysis so far

* Completed EDA analysis to ensure I extract the right data for sentiment analysis. Further info:
    * [notebooks/exploratory_data_analysis.ipynb](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/notebooks/exploratory_data_analysis.ipynb)
    * [src/clean_data.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/clean_data.py)
* Conducted sentiment analysis across time using a sample of lyrics from each year. I used BERT and gpt-40-mini to evaluate sentiment of the complete song. The analysis suggest lyrics have become more negative. It also may suggest that older songs in the dataset (1950s) may be bias towards more positive sounding tracks. For further info:
    * [reports/average_positive_sentiment_over_time_bert_trbs.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_positive_sentiment_over_time_bert_trbs.png)
    * [reports/average_sentiment_over_time_gpt_4o_mini.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_sentiment_over_time_gpt_4o_mini.png)
    * [src/sentiment_per_year_gpt_4o.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_gpt_4o.py)
    * [src/sentiment_per_year_trbs_model.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_trbs_model.py)