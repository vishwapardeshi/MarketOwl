#Libraries for Keywords
import yake
import nltk
import re

class Keywords():
    def __init__(self, quants = True) -> None:
        self.quants = quants

    def _filter_verbs(self, keywords):
        curr_kw = set()
        for ix in range(len(keywords)):
            text,_ = keywords[ix]
            if len(text) == 1:
                word, pos = nltk.pos_tag(nltk.word_tokenize(text))[0]
                if pos[0] != 'V':
                    curr_kw.add(word)
            else:
                curr_kw.add(text)
        return list(curr_kw) if curr_kw else []

    def _generate_keywords(self, df, text_col):
        custom_kw_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, top=30, features=None)
        keywords_2d = []
        for ix, row in df.iterrows():
            keywords = custom_kw_extractor.extract_keywords(row[text_col])
            keywords_2d.append(self._filter_verbs(keywords))
        df[str(text_col) + '_keywords'] = keywords_2d

    def _generate_quant_keywords(self, text):
        return re.findall('[0-9]+\s[a-zA-Z]+\s[a-zA-Z]+\s[a-zA-Z]*', text)
    
    def get_keywords(self, df, text_col_list):
        for text_col in text_col_list:
            #generate keywords
            self._generate_keywords(df, text_col)
            if self.quants:
                #generate quant keywords
                df[str(text_col) + '_quantitative_keywords'] = df[text_col].apply(self._generate_quant_keywords)
