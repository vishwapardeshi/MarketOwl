# MarketOwl - Trend Analysis Tool for Portfolio Managers

Tool for Portfolio Managers to track market, industry sentiment and credibility trends for S&amp;P 500 companies


## How to run?
Command to get sentiment and credbility analysis for transcripts without parallelization

`python3 marketowl.py -s textblob -c flesch -data transcript`

Command to generate credibility analysis by using
1. Gunning Fog
`python3 marketowl.py -s n -c gunning-fog -topic n -data transcript -file Transcript_Extract_10_companies.csv`

For 755 rows of transcripts, it took 8968.298764944077 seconds

2. Flesch Reading Ease
`python3 marketowl.py -s n -c flesch -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 11754

3. Smog Index
`python3 marketowl.py -s n -c smog -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 8666.06145977974 seconds

4. Dale Challe
`python3 marketowl.py -s n -c dale-chall -topic n -data transcript -file Transcript_Extract_10_companies.csv`
For 755 rows of transcripts, it took 