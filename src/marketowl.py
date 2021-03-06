import argparse
import time
from datetime import date

from sentiment import Sentiment
from credibility import Credibility
from summary import Summary
from keywords import Keywords
from transpipeline import QuestionAnswering
from utils.data import load_data

def main(sentiment, credibility, summary, qa, keyword, data, file, sectional):
    #fetch data - if nothing mentioned do for all three
    if data is None:
        df_transcript = load_data('transcripts.csv', 'transcript')
        df_10k = load_data('10k.csv', '10k')
        df_10q = load_data('10q.csv', '10q')

        raise NotImplementedError

    else:
        if data == 'transcript':
            analysis = ""
            if file == None:
                file = 'transcripts.csv'
            print("\n\nLoading transcript file {}...".format(file))
            if sectional == None or sectional.lower() == 'n':
                sectional = False
            else:
                sectional = True
            df_transcript, text_col = load_data(file, 'transcript', sectional)
            print("Loaded into dataframe of size", df_transcript.shape)

            if sentiment != None and sentiment.lower() != 'n':
                #perform sentiment analysis
                print("\nPerforming sentiment analysis using method:", sentiment)
                st = time.time()
                sentiment = Sentiment(method=sentiment)
                sentiment.get_sentiment(df_transcript, text_col)
                print("Total time taken for performing sentiment analysis", time.time() - st)
                analysis += '_sentiment'

            if credibility != None and credibility.lower() != 'n':
                #perform credibility analysis
                st = time.time()
                print("\nPerforming credibility analysis using method:", credibility)
                credibility = Credibility(method=credibility)
                credibility.get_credibility(df_transcript, text_col)
                print("Total time taken for performing credbility analysis", time.time() - st)
                analysis += '_credibility'

            if keyword != None and keyword.lower() not in ['simple', 'quant', 'n']:
                raise ValueError("Incorrect method for extracting keywords! Should be 'simple' or 'quant'")
            else:
                if keyword != None and keyword.lower()  == 'simple':
                    #perform keyword extraction
                    print("\nPerforming non-quant keyword extraction")
                    st = time.time()
                    keywords = Keywords(quants=False)
                    keywords.get_keywords(df_transcript, text_col)
                    print("Total time taken for performing keyword extraction", time.time() - st)
                    analysis += '_keywords'

                elif keyword != None and keyword.lower()  == 'quant':
                    #perform keyword extraction
                    print("\nPerforming quant keyword extraction")
                    st = time.time()
                    keywords = Keywords(quants=True)
                    keywords.get_keywords(df_transcript, text_col)
                    print("Total time taken for performing keyword extraction", time.time() - st)
                    analysis += '_quantKeywords'

            if summary != None and summary.lower() == 'y':
                #perform summarization
                print("\nPerforming summarization")
                st = time.time()
                summary = Summary()
                summary.get_summary(df_transcript, text_col)
                print("Total time taken for performing summarization", time.time() - st)
                analysis += '_summary'

            if qa != None and qa.lower() == 'y':
                #perform summarization
                print("\nPerforming Question Answering")
                print("Question: What drives revenue growth?")
                st = time.time()
                qa = QuestionAnswering("What drives revenue growth?")  # change here to change the question
                qa.get_answer(df_transcript, text_col)
                print("Total time taken for performing question answering", time.time() - st)
                analysis += '_qa'

            #save data checkpoints
            print("\nSaving analysis result for transcript file {} in ../output folder...\n\n".format(file))
            df_transcript.to_csv('../output/transcripts_analysis' + analysis + '_' + date.today().strftime("%d%m%Y") + '.csv')

        elif data == '10k':
            print("\n\nLoading 10k file {}...".format(file))
            df_10k, text_col = load_data(file, '10k')
            raise NotImplementedError

        elif data == '10q':
            df_10q = load_data('10q.csv', '10q')
            raise NotImplementedError

if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-s", "--sentiment", required=False,
    help="sentiment method")
    ap.add_argument("-c", "--credibility", required=False,
    help="credibilty mtehod")
    ap.add_argument("-data", "--datatype", required=True,
    help="data type")
    ap.add_argument("-file", '--filename', required=False,
    help='File name')
    ap.add_argument("-summary", '--summary', required=False,
    help = "Generate Summary? y or n")
    ap.add_argument("-qa", '--qa', required=False,
    help = "Warning: this will take a long time. Sure to generate Q&A answers?  y or n")
    ap.add_argument("-keywords", '--keywords', required=False,
    help = "Generate Keywords? simple or quant")
    ap.add_argument("-sectional", "--sectional_analysis", required=False,
    help="To perform sectional analysis, enter y else n")

    args = vars(ap.parse_args())
    print(args)

    if any(v is not None for v in [args['sentiment'], args['credibility'], args['summary'], args['qa'], args['keywords']]):
        main(args['sentiment'], args['credibility'], args['summary'], args['qa'], args['keywords'], args['datatype'], args['filename'], \
        args['sectional_analysis'])
    else:
        raise ValueError("Specify atleast one analysis to perform. Choose from sentiment, credibility, summary, qa, keywords")
