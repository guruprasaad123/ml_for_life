import tweepy
from textblob import TextBlob
import csv
import sys

def write_to_csv(list):
    with open('sentiment.csv', 'w',encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile)
        print('rows' , list)
        writer.writerows(list)

    csvFile.close()

# Step 1 - Authenticate
consumer_key= 'PRZW3ENQp23NM3ZuN7E1HULUL'
consumer_secret= 'mEhsPr0H1SyloPfg5pNvShif2Y3aFIRAAjKE60jLyKiDGhx9iA'

access_token='962657191-PrrZ2cGPpLKb0XLJRU19VeRBx7xaNEsDYNKiYUyd'
access_token_secret='nqf3sXVAojyRJtp3akcVX8Lrodj0h2Fk7rr01YNX4Thkh'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets

public_tweets = api.search( sys.argv[1] if (len(sys.argv)==2) else 'Modi')



rows = [['label','tweet']]

for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    if analysis.sentiment.polarity == 0:
       rows.append(['Neutral',tweet.text])
    elif analysis.sentiment.polarity >0:
        rows.append(['Positive',tweet.text])
    elif analysis.sentiment.polarity <0:
        rows.append(['Negative',tweet.text])

write_to_csv(rows)

