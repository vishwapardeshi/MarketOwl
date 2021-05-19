import os

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# might need to download these first time using huggingface transformers
#!pip install transformers
#!pip install sentencepiece

#==============================================================================================================================#

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
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        elif self.model == 'bart':
            from transformers import BartTokenizer, BartForConditionalGeneration
            model_name = 'facebook/bart-large-cnn'
            model = BartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name)
        elif self.model == 'pegasus':
            from transformers import PegasusTokenizer, PegasusForConditionalGeneration
            model_name = "human-centered-summarization/financial-summarization-pegasus"
            model = PegasusForConditionalGeneration.from_pretrained(model_name)
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
        elif self.model == 'led':
            from transformers import LEDTokenizer, LEDForConditionalGeneration, LEDConfig
            model_name = 'allenai/led-base-16384'
            model = LEDForConditionalGeneration.from_pretrained(model_name)
            tokenizer = LEDTokenizer.from_pretrained(model_name)

        # run seleted model
        if self.model in ['t5', 'bart', 'pegasus']:

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

#==============================================================================================================================#

class QuestionAnswering:

    def _answering(self, question, context):

        from transformers import pipeline

        sentences = sent_tokenize(context)
        words = word_tokenize(context)

        window = round(4096/(len(words)/len(sentences))) - 1
        stride = round(window*0.85)
        begin = 0
        end = window
        qa_list = []
        while (begin < len(sentences)):
            # Tokenize our text
            # If you want to run the code in Tensorflow, please remember to return the particular tensors as simply as using return_tensors = 'tf'
            if end <= len(sentences):
                text_to_summarize = ' '.join(sentences[begin:end])
            elif end > len(sentences):
                text_to_summarize = ' '.join(sentences[begin:])

            # Generating an answer to the question in context
            model_name = "mrm8488/longformer-base-4096-finetuned-squadv2"
            qa = pipeline(
                "question-answering",
                model = model_name,
                tokenizer = model_name,
                device=0
            )
            answer = qa(question=question, context=text_to_summarize)
            qa_list.append((
                answer['answer'],
                round(answer['score'], 4),
                text_to_summarize[max(0,answer['start']-100): min(len(text_to_summarize), answer['end']+100)]
            ))

            begin += stride
            end += stride

        qa_list = sorted(qa_list, key = lambda x: x[1], reverse=True)

        return qa_list

    #---------------------------------------------------------------------#

    def get_answer(self, question, context):
        qa_list = self._answering(question, context)
        answer = qa_list[:5]

        return answer

    #---------------------------------------------------------------------#

    def save_answer(self, answer, file_name, destination):
        path = os.path.join(destination, file_name)
        with open(path, "w") as text_file:
            for item in answer:
                temp = [str(x) for x in item]
                output = ','.join(temp)
                text_file.write(output + '\n')

    #---------------------------------------------------------------------#

#==============================================================================================================================#
