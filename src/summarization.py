import os

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# might need to download these first time using huggingface transformers
#!pip install transformers
#!pip install sentencepiece

class Summarization:

    def __init__(self, model='bart'):
        """
        Note: I think BART produces the best result, while the others do not make meaningful summaries.
        """
        self.model = model

    def _summarize(self, text):
        """
        Function to summarize a text
        (Note that this is different from other classes in this package: we only summarize
        a text, not a whole column here, because the summarization speed is slow.)]
        """

        # break text into tokens to summarize part by part
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # select model
        if self.model == 't5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            model_name = 't5-base'
        elif self.model == 'bart':
            from transformers import BartTokenizer, BartForConditionalGeneration
            model_name = 'facebook/bart-large-cnn'
        elif self.model == 'pegasus':
            from transformers import PegasusTokenizer, PegasusForConditionalGeneration
            model_name = "human-centered-summarization/financial-summarization-pegasus"
        elif self.model == 'led':
            from transformers import LEDTokenizer, LEDForConditionalGeneration, LEDConfig
            model_name = 'allenai/led-base-16384'

        # run seleted model
        if self.model in ['t5', 'bart', 'pegasus']:
            # initialize model
            model = BartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name)

            # initialize parsing method for text (text length < 512 tokens for these transformer models)
            window = round(512/(len(words)/len(sentences))) - 1
            stride = round(window*0.85)
            begin = 0
            end = window
            summary = []

            # summary text part by part
            while (begin < len(sentences)):
                # Tokenize our text
                # If you want to run the code in Tensorflow, please remember to return the particular tensors as simply as using return_tensors = 'tf'
                if end <= len(sentences):
                    text_to_summarize = ' '.join(sentences[begin:end])
                elif end > len(sentences):
                    text_to_summarize = ' '.join(sentences[begin:])

                if self.model == 't5':
                    input_ids = tokenizer.encode("summarize: " + text_to_summarize, truncation=True, return_tensors="pt")
                else:
                    input_ids = tokenizer(text_to_summarize, truncation=True, return_tensors="pt").input_ids

                # Generate the output (Here, we use beam search but you can also use any other strategy you like)
                output = model.generate(
                    input_ids,
                    num_beams=5,
                    early_stopping=True
                )
                summary.append(tokenizer.decode(output[0], skip_special_tokens=True))

                begin += stride
                end += stride

        elif self.model == 'led':
            model = LEDForConditionalGeneration.from_pretrained(model_name)
            tokenizer = LEDTokenizer.from_pretrained(model_name)
            inputs = tokenizer([text], truncation=True, return_tensors='pt')
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, early_stopping=True)
            summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

        return summary

    #---------------------------------------------------------------------#

    def get_summary(self, text):
        if self.model in ['t5', 'bart', 'pegasus', 'led']:
            summary = self._summarize(text)
        else:
            raise ValueError("Incorrect model name for extracting keywords! Should be 't5', 'bart', 'pegasus', or 'led'.")

        return summary

    #---------------------------------------------------------------------#

    def save_summary(self, summary, file_name, destination):
        path = os.path.join(destination, file_name)
        with open(path, "w") as text_file:
            text_file.write('* ' + ' \n* '.join(summary))

    #---------------------------------------------------------------------#
