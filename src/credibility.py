import numpy as np

from utils.text_processing import *

class Credibility:
    def __init__(self, method = 'gunning-fog'):
        self.method = method
    
    def _gunning_fog(self, text):
        per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
        grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
        return grade

    def _flesch_reading_ease(self, text):
        """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
            ASL = average sentence length (number of words 
                divided by number of sentences)
            ASW = average word length in syllables (number of syllables 
                divided by number of words)
        """
        FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
            float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)
    
    def _smog_index(self, text):
        """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here, 
        polysyllable count = number of words of more
        than two syllables in a sample of 30 sentences.
        """
    
        if sentence_count(text) >= 3:
            poly_syllab = poly_syllable_count(text)
            SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) \
                    + 3.1291
            return legacy_round(SMOG, 1)
        else:
            return 0
    
    
    def _dale_chall_readability_score(self, text):
        """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
        """
        words = word_count(text)
        # Number of words not termed as difficult words
        count = word_count - difficult_words(text)
        if words > 0:
    
            # Percentage of words not on difficult word list
    
            per = float(count) / float(words) * 100
        
        # diff_words stores percentage of difficult words
        diff_words = 100 - per
    
        raw_score = (0.1579 * diff_words) + \
                    (0.0496 * avg_sentence_length(text))
        
        # If Percentage of Difficult Words is greater than 5 %, then;
        # Adjusted Score = Raw Score + 3.6365,
        # otherwise Adjusted Score = Raw Score
    
        if diff_words > 5:       
    
            raw_score += 3.6365
            
        return legacy_round(score, 2)

    def get_credibility(self, df, text_col):
        if self.method == 'gunning-fog':
            df['crebility_index'] = df[text_col].apply(self._gunning_fog)
        elif self.method == 'flesch':
            df['crebility_index'] = df[text_col].apply(self._flesch_reading_ease)
        elif self.method == 'smog':
            df['crebility_index'] = df[text_col].apply(self._smog_index)
        elif self.method == 'dale-chall':
            df['crebility_index'] = df[text_col].apply(self._dale_chall_readability_score)
        else:
            raise ValueError("Incorrect method for extracting crebility index! \
                Should be gunning-fog, flesch, smog or dale-chall")