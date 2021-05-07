import numpy as np
import collections

#for sentiment analysis
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class Sentiment:
    def __init__(self, method = 'vader'):
        self.method = method

    def _vader(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        vader = SentimentIntensityAnalyzer()
        df['polarity_score'] = [vader.polarity_scores(x)['compound'] for x in df[text_col]]

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

    def _vader_sent(self, df, text_col):
        pos_list = []
        neg_list = []
        for i in range(len(df[text_col])):

            # break text into sentences
            sentences = sent_tokenize(df[text_col][i])

            # calculate sentiment scores
            vader = SentimentIntensityAnalyzer()
            polarity_score = np.array([vader.polarity_scores(x)['compound'] for x in sentences])

            # change sentiment scores into labels
            conditions = [
                (polarity_score <= -0.05),
                (polarity_score > -0.05) & (polarity_score < 0.05),
                (polarity_score >= 0.05)
            ]
            values = [-1, 0, 1]
            sentiment_label = np.select(conditions, values)

            # count the ratio of positive/negative sentences
            counter = collections.Counter(sentiment_label)
            pos_list.append(round(counter[1]/sum(counter.values()), 4))
            neg_list.append(round(counter[-1]/sum(counter.values()), 4))

            # assign ratios to new columns
            df['pos_ratio'] = pos_list
            df['neg_ratio'] = neg_list

    def _bert(self, df, text_col):
        """
        Function to get sentiment label using bert
        """

    def get_sentiment(self, df, text_col):
        if self.method == 'vader':
            self._vader(df, text_col)
            df.to_csv('sentiment.csv')    # WHY THIS??? -Tony
        elif self.method == 'textblob':
            self._textblob(df, text_col)
        elif self.method == 'vader_sent':
            self._vader_sent(df, text_col)
        else:
            raise ValueError("Incorrect method for extracting sentiments! \
                Should be 'vader', 'textblob', or 'vader_sent'")
