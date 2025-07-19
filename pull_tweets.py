# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm

# === SSL Patch for certifi ===
import snscrape.base
import requests
import certifi
import ssl

# Create a custom requests session with certifi's CA bundle
session = requests.Session()
session.verify = certifi.where()

# Patch the _request method to use this session
snscrape.base._httpSession = session
# === End SSL Patch ===



ticket_symbols = ["ZIM","XIFR","TRIN","REFI","INSW"]
# Creating list to append tweet data
stock = ticket_symbols[0]
tweets = []
for year in range(2019, 2025):
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in tqdm(
        enumerate(
            sntwitter.TwitterSearchScraper(
                f"{stock} since:{year}-01-01 until:{year+1}-01-01"
            ).get_items()
        ),
        total=1000,
    ):  # declare a username
        if i > 1000:  # number of tweets you want to scrape
            break
        tweets_list.append(
            [tweet.date, tweet.id, tweet.content, tweet.username]
        )  # declare the attributes to be returned
    # Creating a dataframe from the tweets list above
    tweet_df = pd.DataFrame(
        tweets_list, columns=["Datetime", "Tweet Id", "Text", "Username"]
    ).assign(year=year)
    tweets.append(tweet_df)
tweets = pd.concat(tweets).reset_index(drop=True).copy()