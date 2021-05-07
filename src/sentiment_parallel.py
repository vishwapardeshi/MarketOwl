import numpy as np
import pandas as pd
import multiprocessing
import dask.dataframe as ddf
#import dask.multiprocessing
#dask.config.set(scheduler='processes')

#for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class Sentiment:
    def __init__(self, method = 'vader'):
        self.method = method

    def _vader(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        dask_dataframe = ddf.from_pandas(df[:20], npartitions=multiprocessing.cpu_count())
        print("This is text col", text_col)
        print('***********', df.columns)
        print(df.columns)
        vader = SentimentIntensityAnalyzer()
        print("starting dask")
        result = dask_dataframe[text_col].map_partitions(lambda x: vader.polarity_scores(x)['compound'])
        print(result)
        print('1 done')
        df['polarity_score'] = result.compute()
        print("saved result")
        print(df.columns)


        # create a list of our conditions
        conditions = [
            (df['polarity_score'] <= -0.05),
            (df['polarity_score'] > -0.05) & (df['polarity_score'] < 0.05),
            (df['polarity_score'] >= 0.05)
            ]

        # create a list of the values we want to assign for each condition
        values = ['Negative', 'Neutral', 'Positive']

        # create a new column and use np.select to assign values to it using our lists as arguments
        df['sentiment_label'] = np.select(conditions, values)

    def _textblob(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        #iterate through rows to get polarity score
        for ix, row in df.iterrows():
            df.loc[ix, 'polarity_score'] = round(TextBlob(row[text_col]).sentiment.polarity, 3)

        df['polarity_score'] = df.text_col.map_partitions(lambda x: round(TextBlob(x).sentiment.polarity, 3))

        # create a list of our conditions
        conditions = [
            (df['polarity_score'] < 0),
            (df['polarity_score'] == 0),
            (df['polarity_score'] > 0)
            ]

        # create a list of the values we want to assign for each condition
        values = ['Negative', 'Neutral', 'Positive']

        # create a new column and use np.select to assign values to it using our lists as arguments
        df['sentiment_label'] = np.select(conditions, values)


    def _bert(self, df, text_col):
        """
        Function to get sentiment label using bert
        """

    def get_sentiment(self, df, text_col):
        if self.method == 'vader':
            self._vader(df, text_col)
            df.to_csv('sentiment.csv')
        elif self.method == 'textblob':
            self._textblob(df, text_col)
        else:
            raise ValueError("Incorrect method for extracting sentiments! \
                Should be vader, textblob")
