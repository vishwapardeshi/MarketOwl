import argparse

from sentiment import Sentiment
from credibility import Credibility
from utils.data import load_data

def main(sentiment, credibility, data):
    #fetch data - if nothing mentioned do for all three 
    if data is None:
        df_transcript = load_data('transcripts.csv', 'transcript')
        df_10k = load_data('10k.csv'. '10k')
        df_10q = load_data('10q.csv', '10q')

        raise NotImplementedError
        #perform sentiment anlaysis

        #perform credibility analysis

        #save results 

    else:
        if data == 'transcript':
            df_transcript, text_col = load_data('transcripts.csv', 'transcript')
            #perform sentiment analysis 
            sentiment = Sentiment(method=sentiment)
            sentiment.get_sentiment(df_transcript, text_col)

            #perform credibility analysis
            credibility = Credibility(method=credibility)
            credibility.get_credibility(df_transcript, text_col)

            #save data checkpoints
            df_transcript.to_csv('transcripts_analysis.csv')
        elif data == '10k':
            df_10k = load_data('10k.csv'. '10k')
            raise NotImplementedError
             
        elif data == '10q':
            df_10q = load_data('10q.csv', '10q')
            raise NotImplementedError
    
if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-s", "--sentiment", required=True,
    help="sentiment index")
    ap.add_argument("-c", "--credibility", required=True,
    help="credibilty index")
    ap.add_argument("-data", "--datatype", required=False,
    help="data type")
    args = vars(ap.parse_args())
    main(ap['sentiment'], ap['credibility'], ap['datatype'])
