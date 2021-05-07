from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Keywords():

	#---------------------------------------------------------------------#

	def __init__(self, method='tfidf'):
		self.method = method

	#---------------------------------------------------------------------#

	def _count(self, df, text_col):
        """
        Function to find popular terms (2-gram, adding 3-gram is too slow) based on frequency
        (Excluding 1-gram as it contains too much noises)
        """
		vectorizer = CountVectorizer(
			stop_words = 'english', 
			ngram_range = (2,2), 
			min_df = round(len(df[text_col])/20)+1
		)
		X = vectorizer.fit_transform(df[text_col])    # note: X is a sparse matrix
		
		top_keywords_all = []
		for i in range(len(df[text_col])):
		    keywords = {}
		    tf = X[i,:].toarray().tolist()[0]

		    # gather top 100 keywords
		    for word, count in zip(vectorizer.get_feature_names(), tf):
		        keywords[word] = count
		    words_dict = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
		    top_keywords = [x[0] for x in words_dict][:100]
		    top_keywords_all.append(top_keywords)

		# add column to dataframe
		 df['keywords'] = top_keywords_all

	#---------------------------------------------------------------------#

	def _tfidf(self, df, text_col):
        """
        Function to find popular terms (2-gram, adding 3-gram is too slow) based on TFIDF
        (Excluding 1-gram as it contains too much noises)
        """
		vectorizer = TfidfVectorizer(
			stop_words = 'english', 
			ngram_range = (2,2), 
			min_df = round(len(df[text_col])/20)+1
		)
		X = vectorizer.fit_transform(df[text_col])    # note: X is a sparse matrix
		
		top_keywords_all = []
		for i in range(len(df[text_col])):
		    keywords = {}
		    tfidf = X[i,:].toarray().tolist()[0]

		    # gather top 100 keywords
		    for word, count in zip(vectorizer.get_feature_names(), tfidf):
		        keywords[word] = count
		    words_dict = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
		    top_keywords = [x[0] for x in words_dict][:100]
		    top_keywords_all.append(top_keywords)

		# add column to dataframe
		 df['keywords'] = top_keywords_all

	#---------------------------------------------------------------------#

	def get_keywords(self, df, text_col):
        if self.method == 'count':
            self._count(df, text_col)
        elif self.method == 'tfidf':
            self._tfidf(df, text_col)
        else:
            raise ValueError("Incorrect method for extracting keywords! \
                Should be 'tfidf' or 'count'")

	#---------------------------------------------------------------------#