# Gensim
from nltk import text
import gensim
import gensim.corpora as corpora

from pprint import pprint

from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

from utils.text_processing import * 

class TopicModel:
    def __init__(self) -> None:
        self.id2word = None
        self.corpus = None
        self.lda_model = None
        self.data_lemmatized = None

    def _prepare_data(self, df, text_col):
        data = df[text_col].values.tolist()

        #convert to words
        data_words = list(sent_to_words(data))

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        self.data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        self.df['clean_text'] = self.data_lemmatized

        return self.data_lemmatized


    def _create_dic_corpus(self, df, text_col):
        # Create Corpus
        texts = self._prepare_data(df, text_col)

        # Create Dictionary
        self.id2word = corpora.Dictionary(self.data_lemmatized)

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]
    
    def lda(self, df, text_col):
        # Build LDA model
        self._create_dic_corpus(df, text_col)
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                id2word=self.id2word,
                                                num_topics=20, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    
    def get_topics(self):
        pprint(self.lda_model.print_topics())
        doc_lda = self.lda_model[self.corpus]
    
    def get_metrics(self):
        # Compute Perplexity
        print('\nPerplexity: ', self.lda_model.log_perplexity(self.corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
    

