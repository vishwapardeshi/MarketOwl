import argparse
import time
from sentiment import Sentiment
from credibility import Credibility
from topic_modeling import TopicModel
from utils.data import load_data

def main(sentiment, credibility, topic, data, file):
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
            if file == None:
                file = 'transcripts.csv'
            print("\n\nLoading transcript file...")
            df_transcript, text_col = load_data(file, 'transcript')
            print("Loaded into dataframe of size", df_transcript.shape)
            

            if sentiment != 'n':
                #perform sentiment analysis
                print("\nPerforming sentiment analysis using method:", sentiment)
                st = time.time()
                sentiment = Sentiment(method=sentiment)
                sentiment.get_sentiment(df_transcript, text_col)
                print("Total time taken for performing sentiment analysis", time.time() - st)

            if credibility != 'n':
                #perform credibility analysis
                st = time.time()
                print("\nPerforming credibility analysis using method:", credibility)
                credibility = Credibility(method=credibility)
                credibility.get_credibility(df_transcript, text_col)
                print("Total time taken for performing credbility analysis", time.time() - st)

            if topic == 'y':
                #perform topic modeling
                st = time.time()
                print("\nPerforming topic modeling using LDA")
                topic_model = TopicModel()
                topic_model.lda(df_transcript, text_col)
                topic_model.get_topics()
                topic_model.get_metrics()
                print("Total time taken for performing credbility analysis", time.time() - st)

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
    help="sentiment method")
    ap.add_argument("-c", "--credibility", required=True,
    help="credibilty mtehod")
    ap.add_argument("-topic", "--topic", required=True,
    help="Topic Modeling")
    ap.add_argument("-data", "--datatype", required=True,
    help="data type")
    ap.add_argument("-file", '--filename', required=False,
    help='File name')
    args = vars(ap.parse_args())
    print(args)
    main(args['sentiment'], args['credibility'], args['topic'], args['datatype'], args['filename'])
