#Libraries for Summarization
from gensim.summarization.summarizer import summarize
import en_core_web_sm

class Summary:
    def __init__(self) -> None:
        self.nlp = en_core_web_sm.load()
    def _generate_summary(self, df, text_col):
    # Get wiki content.
        summ_per = [''] * len(df)
        summ_words = [''] * len(df) 
        for ix, row in df.iterrows():
            text = row[text_col]
            doc = self.nlp(text)

            # Summary (3% of the original content).
            summ_per = summarize(text, ratio = 0.03)

            # Summary (200 words)
            summ_words = summarize(text, word_count = 200)
            df.loc[ix,  str(text_col) + '_summary_percent'] = summ_per
            df.loc[ix,  str(text_col) + '_summary_wordcount'] = summ_words

    def get_summary(self, df, text_col_list):
        for text_col in text_col_list:
            self._generate_summary(df, text_col)