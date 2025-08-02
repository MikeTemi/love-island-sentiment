import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def collect_tweets(search_term, start_date, end_date, max_tweets=1000):
    """
    Collect tweets about Love Island for sentiment analysis.

    Args:
        search_term: str: What to search for (e.g, '#LoveIsland').
        start_date: str: Start date for collection
        end_date: str: End date for collection
        max_tweets: int: Maximum number of tweets to collect

    Returns:
        Lists of tweet data dictionaries
    """

    print(f"Collecting tweets for search term: {search_term}")
    print(f"Date range: {start_date} to {end_date}")

    tweets_data = []

    # Create the search query with the specified date range
    query = f"{search_term} since:{start_date} until:{end_date}"

    try:
        # use snscrape to collect tweets
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_tweets:
                break

            # Extract relevant tweet information
            tweet_data = {
                'date': tweet.date,
                'tweet_id': tweet.id,
                'content': tweet.content,
                'username': tweet.user.username,
                'likes': tweet.likeCount,
                'retweets': tweet.retweetCount,
                'replies': tweet.replyCount,
                'search_term': search_term
            }

            tweets_data.append(tweet_data)

            # Let's add a small delay to be respectful
            time.sleep(0.1)

    except Exception as e:
        print(f"Error collecting tweets: {e}")

    print(f"Collected {len(tweets_data)} tweets")
    return tweets_data