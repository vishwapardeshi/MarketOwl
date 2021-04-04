class Sentiment:
    def __init__(self, method = 'vader'):
        self.method = method

    def _vader(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        pass
    
    def _textblob(self, df, text_col):
        """
        Function to get sentiment label using vader library
        """
        pass

    def get_sentiment(self, df, text_col):
        if self.method == 'vader':
            self._vader(df, text_col)
        elif self.method == 'textblob':
            self._textblob(df, text_col)
        else:
            raise ValueError("Incorrect method for extracting sentiments! \
                Should be vader, textblob")
