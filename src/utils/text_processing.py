import spacy
from textstat.textstat import textstatistics, legacy_round
#from textstat.textstat import textstatistics, easy_word_set, legacy_round

import nltk
from nltk.tokenize import TweetTokenizer

def tokenize(df, text_col):
    """
    Function to tokenize the text input using tweetTokenizer
    """
    tweet_tokenizer = TweetTokenizer()
    df['tokenized_text'] = df[text_col].apply(tweet_tokenizer.tokenize)
    return df

def break_sentences(text):
    """
    Splits the text into sentences, using Spacy's sentence segmentation which can
    be found at https://spacy.io/usage/spacy-101
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list(doc.sents)

def word_count(text):
    """
    Returns Number of Words in the text
    """
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words

def sentence_count(text):
    """
    Returns the number of sentences in the text
    """
    sentences = break_sentences(text)
    return len(sentences)

def avg_sentence_length(text):
    """
    Returns average sentence length
    """
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length

def syllables_count(word):
    """
    Textstat is a python package, to calculate statistics from text to determine readability
    complexity and grade level of a particular corpus.
    Package can be found at https://pypi.python.org/pypi/textstat
    """
    return textstatistics().syllable_count(word)

def avg_syllables_per_word(text):
    """
    Returns the average number of syllables per word in text
    """
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)

def difficult_words(text):
    """
    Return total Difficult Words in a text
    """
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]

    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in easy_word_set and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)

def poly_syllable_count(text):
    """
    A word is polysyllablic if it has more than 3 syllables this functions
    returns the number of all such words present in text
    """
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]


    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count
