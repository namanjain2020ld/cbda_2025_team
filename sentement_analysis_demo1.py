import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

import plotly.io as pio
pio.renderers.default = "browser"

import plotly.offline as pyo


"""to do for now
pull histotric stock prices with python.

explore stock prices and learn how to plot them.

learn how to pull Historic tweet data about a specific stock.

run sentiment analysis on tweets.

see if there is any leading indicators of stock prices with sentement (mostly .)
"""

ticket_symbols = ["ZIM","XIFR","TRIN","REFI","INSW"]

# Define the ticker symbol
#ticket_symbol = "AAPL"

# maks token object
#apple = yf.Ticker(ticket_symbol)
#apple_info = apple.info
# apple_htry = apple.history(period="max") # 5y


# print(apple.info)
# print(apple_htry)

# to get it as columns 
# data = pd.DataFrame(apple_htry)
# print(data)

# showing line chart for open prices .......
#apple_htry["Open"].plot(figsize=(15,5))
#plt.show()

# to see the actions -- shwos dividents and stock splits 
# print(apple.actions)
# or do directly  apple.dividends     apple.splits


# to pridict next earning reports
# print(pd.DataFrame(apple.calendar))
history ={}
figs =[]
for i in ticket_symbols:
    token_obj = yf.Ticker(i)
    history[i] = token_obj.history(period="5y")

    # print(history)
    temp_dataFrame = history[i]
    # demo ploting 
    fig = go.Figure(data=[go.Candlestick(
                    x=temp_dataFrame.index, # the date is index hear
                    open = temp_dataFrame["Open"],
                    high = temp_dataFrame["High"],
                    low = temp_dataFrame["Low"],
                    close = temp_dataFrame["Close"]
                        )]
                    )
    fig.update_layout(
        margin = dict (l=20,r=20,t=60,b=10),
        height = 300,
        paper_bgcolor = "LightSteelBLue",
        title =i,
    )

    figs.append(fig)


# Save all figures to one HTML file
with open("all_charts.html", "w") as f:
    for fig in figs:
        f.write(pyo.plot(fig, include_plotlyjs='cdn', output_type='div'))

import webbrowser
webbrowser.open("all_charts.html")


# pull tweets about each stock 
# file deomo pull_tweets.py


# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm
stock = ticket_symbols[0]

# Creating list to append tweet data
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
            [tweet.date, tweet.id, tweet.content, tweet.user.username]
        )  # declare the attributes to be returned
    # Creating a dataframe from the tweets list above
    tweet_df = pd.DataFrame(
        tweets_list, columns=["Datetime", "Tweet Id", "Text", "Username"]
    ).assign(year=year)
    tweets.append(tweet_df)
tweets = pd.concat(tweets).reset_index(drop=True).copy()