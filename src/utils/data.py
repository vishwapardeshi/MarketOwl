#use this script to pull data for experimentation purpose
import nltk
from nltk.tokenize import TweetTokenizer

import config

def load_data(filename, type = 'transcipt'):
    """
    To load data of three types - 
    1. Transcripts
    2. 10 K 
    3. 10 Q

    Args:
    filename : Name of file present in input folder
    type : Type of data to be loaded

    Returns:
    df (DataFrame) : Containing text column extracted along with unique ID 
    text_col (String) : Return name of text column to focus on 
    """
    df = pd.read_csv(config.INPUT_FOLDER + filename)
    if type == 'transcript':
        #load only company ticker 
        text_col = 'transcript'
    elif type == '10k':
        text_col = 'text'
    elif type == '10q':
        text_col = 'text'
    else:
        raise ValueError("Incorrect data type! \
                Should be transcript, 10k or 10q")

def tokenize(df, text_col):
    """
    Function to tokenize the text input using tweetTokenizer
    """
    tweet_tokenizer = TweetTokenizer()
    df['tokenized_text'] = df[text_col].apply(tweet_tokenizer.tokenize)
    return df
        