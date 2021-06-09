# MarketOwl - Trend Analysis Tool for Portfolio Managers

Tool for Portfolio Managers to track market, industry sentiment and credibility trends for S&amp;P 500 companies


## How to run?
Command to get sentiment and credbility analysis for transcripts without parallelization

`python marketowl.py -s textblob -c flesch -data transcript`

Command to generate credibility analysis by using
1. Gunning Fog
`python marketowl.py -s n -c gunning-fog -summary n -qa n -keywords n -data transcript -file Transcript_Extract_10_companies.csv`

For 755 rows of transcripts, it took 8968.298764944077 seconds

2. Flesch Reading Ease
`python marketowl.py -s n -c flesch -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 11754.16016292572 seconds

3. Smog Index
`python marketowl.py -s n -c smog -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 8666.06145977974 seconds

4. Dale Challe
`python marketowl.py -s n -c dale-chall -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 16478.965087890625 seconds.

You need to specify analysis to perform, the below will throw error
`python marketowl.py -data transcript -file Transcript_Extract_10_companies.csv`

Sentiment analysis specified
`python marketowl.py -data transcript -file Transcript_Extract_10_companies.csv --sentiment 'vader_sent'`

Credibility & summary specified
`python marketowl.py -data transcript -file Transcript_Extract_10_companies.csv -c gunning-fog -summary y`

Keywords on sectional 
`python marketowl.py -data transcript -file Transcript_Extract_Sections.csv -keywords simple -sectional y `

## Transformers Pipeline
Note: It is not recommended to run the code on all transcripts since it takes forever to run! However, you can still do question answering on all transcripts with `MarketOwl.py`. Currently I set default question to "What drives revenue growth?" If you want to change the default question for `MarketOwl.py`, there is an option at line 87 of `MarketOwl.py`; just switch the string inside `QuestionAnswering()`.
* To see how to run transformer models (transformer-based summary and question answering), please check the Jupyter Notebook `main_tony.ipynb`. For more elaborated documentation please check https://huggingface.co/transformers/model_summary.html. 
* Currently I set the running device to CPU. If you want to change it to GPU, line 165 of `transpipeline.py` has the option to change it (change -1 to 0). However, please make sure that you have enough VRAM since it crashes on my laptop, which only has 4GB of VRAM.  
* I think it is better to build a more interative interface to present the results of question answering. Our current code can produce the results but how they are presented can be improved in the future. 
