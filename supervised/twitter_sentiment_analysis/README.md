We'll be using [TextBlob](https://textblob.readthedocs.io/en/dev/) in our program to analyse the sentiment polarity of the text i.e Real time tweets which are going to get from [tweepy api](http://www.tweepy.org/) .

## Installation

```python
pip install -t lib -r requirements.txt
```

## Running the app

using the command :

```python
python sentiment_analysis.py <tweetHeadline>
```

And the app store the output in `sentiment.csv` file in the following format.

| sentiment ( positive , negative , neutral ) | tweet |
|:------------------------------------------- |:----- |
| ....                                        | ...   |

Sentiment Measurement :

- polarity === 0 , **Neutral**

- polarity >0 , **Positive**

- polarity <0 , **Negative**

Resources:

https://www.youtube.com/watch?v=o_OZdbCzHUA&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=2

https://github.com/llSourcell/twitter_sentiment_challenge

https://www.quora.com/How-does-sentiment-analysis-work
