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
consumer_key= '<consumer_key>'
consumer_secret= '<consumer_secret>'

access_token='<access_token>'
access_token_secret='<access_token_secret>'

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

