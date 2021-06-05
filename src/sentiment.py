import numpy as np
import collections

#for sentiment analysis
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class Sentiment:
    def __init__(self, method = 'vader'):
        self.method = method

    #---------------------------------------------------------------------#

    def _vader(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        vader = SentimentIntensityAnalyzer()
        df[str(text_col) + '_vader_polarity_score'] = [vader.polarity_scores(x)['compound'] for x in df[text_col]]

        # create a list of our conditions
        conditions = [
                (df[str(text_col) + '_vader_polarity_score'] <= -0.05),
                (df[str(text_col) + '_vader_polarity_score'] > -0.05) & (df[str(text_col) + '_vader_polarity_score'] < 0.05),
                (df[str(text_col) + '_vader_polarity_score'] >= 0.05)
            ]

        # create a list of the values we want to assign for each condition
        values = ['Negative', 'Neutral', 'Positive']

        # create a new column and use np.select to assign values to it using our lists as arguments
        df[str(text_col) + '_vader_sentiment_label']  = np.select(conditions, values)

    #---------------------------------------------------------------------#

    def _textblob(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        #iterate through rows to get polarity score
        for ix, row in df.iterrows():
            df.loc[ix, str(text_col) + '_textblob_polarity_score'] = round(TextBlob(row[text_col]).sentiment.polarity, 3)

        # create a list of our conditions
        conditions = [
                (df[str(text_col) + '_textblob_polarity_score'] < 0),
                (df[str(text_col) + '_textblob_polarity_score'] == 0),
                (df[str(text_col) + '_textblob_polarity_score'] > 0)
            ]

        # create a list of the values we want to assign for each condition
        values = ['Negative', 'Neutral', 'Positive']

        # create a new column and use np.select to assign values to it using our lists as arguments
        df[str(text_col) + '_textblob_sentiment_label'] = np.select(conditions, values)

    #---------------------------------------------------------------------#

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

        # normalize ratios
        pos_ratio_normalized = (np.array(pos_list) - np.mean(pos_list))/np.std(pos_list)
        neg_ratio_normalized = (np.array(neg_list) - np.mean(neg_list))/np.std(neg_list)

        df[str(text_col) + '_pos_ratio_normalized'] = pos_ratio_normalized
        df[str(text_col) + '_neg_ratio_normalized'] = neg_ratio_normalized

        #---------------------------------------------------------------------#

    def get_sentiment(self, df, text_col):
        if self.method == 'vader':
            for col in text_col:
                self._vader(df, col)
        elif self.method == 'textblob':
            for col in text_col:
                self._textblob(df, col)
        elif self.method == 'vader_sent':
            for col in text_col:
                self._vader_sent(df, col)
        else:
            raise ValueError("Incorrect method for extracting sentiments! \
                Should be 'vader', 'textblob', or 'vader_sent'")
