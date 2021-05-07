import argparse
import time
from sentiment_parallel import Sentiment
from credibility_parallel import Credibility
from utils.data import load_data

def main(sentiment, credibility, data):
    #fetch data - if nothing mentioned do for all three
    if data is None:
        df_transcript = load_data('transcripts.csv', 'transcript')
        df_10k = load_data('10k.csv', '10k')
        df_10q = load_data('10q.csv', '10q')

        raise NotImplementedError
        #perform sentiment anlaysis

        #perform credibility analysis

        #save results

    else:
        if data == 'transcript':
            print("\n\nLoading transcript file...")
            df_transcript, text_col = load_data('transcripts.csv', 'transcript')
            #print("Loaded into dataframe of size", df_transcript.shape)
            print("This is the text col", text_col)

            #perform sentiment analysis
            print("\nPerforming sentiment analysis using method:", sentiment)
            st = time.time()
            sentiment = Sentiment(method=sentiment)
            sentiment.get_sentiment(df_transcript, text_col)
            print("Total time taken for performing sentiment analysis", time.time() - st)

            #perform credibility analysis
            #st = time.time()
            #print("\nPerforming credibility analysis using method:", credibility)
            #credibility = Credibility(method=credibility)
            #credibility.get_credibility(df_transcript, text_col)
            #print("Total time taken for performing credbility analysis", time.time() - st)

            #save data checkpoints
            print("\nSaving sentiment & credibility result for transcript file...\n\n")
            df_transcript.to_csv('transcripts_analysis.csv')
        elif data == '10k':
            print("\n\nLoading transcript file...")
            df_10k, text_col = load_data('sec_10k.csv', '10k')
            #print("Loaded into dataframe of size", df_10k.shape)

            #perform sentiment analysis
            print("\nPerforming sentiment analysis using method:", sentiment)
            sentiment = Sentiment(method=sentiment)
            sentiment.get_sentiment(df_10k, text_col)

            #perform credibility analysis
            print("\nPerforming credibility analysis using method:", credibility)
            credibility = Credibility(method=credibility)
            credibility.get_credibility(df_10k, text_col)

            #save data checkpoints
            print("\nSaving sentiment & credibility result for transcript file...\n\n")
            df_10k.to_csv('sec_10k_item1_analysis.csv')


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
    print(args)
    main(args['sentiment'], args['credibility'], args['datatype'])
