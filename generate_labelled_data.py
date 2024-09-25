#Code to generate labelled sample data

import pandas as pd

song_lyrics_clean_df = pd.read_csv("song_lyrics_clean_df.csv")

#Dataset for tracks labelled "positive"
artists_positive = ['Sade','Anita Baker','Metallica','Peter Gabriel','Radiohead','Barry White','Chic','The Cure','D\'Angelo','FKA twigs','Fleetwood Mac','Marvin Gaye','Jimmy Eat World','M83','Steely Dan','AC/DC']

song_lyrics_positive_to_label_df = song_lyrics_clean_df[song_lyrics_clean_df['artist'].isin(artists_positive)]

song_lyrics_positive_to_label_df.to_csv("song_lyrics_positive_to_label_df.csv")

# Dataset for tracks labelled negative
artists_negative = ['Sade','Anita Baker','Metallica','Nine Inch Nails','Peter Gabriel','Pink Floyd','Mr. Fingers','Alice in Chains','Radiohead','Run The Jewels','Robyn','Talk Talk','Weezer','Chic','The Cure','Curtis Mayfield','Danny Brown','Deftones','FKA twigs','Fleetwood Mac','Fontaines D.C.','Frank Ocean','Linkin Park','Marvin Gaye','Kendrick Lamar','Joy Division']

song_lyrics_negative_to_label_df = song_lyrics_clean_df[song_lyrics_clean_df['artist'].isin(artists_negative)]

song_lyrics_negative_to_label_df.to_csv("song_lyrics_negative_to_label_df.csv")