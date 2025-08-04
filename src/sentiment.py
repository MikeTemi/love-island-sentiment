import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os

def clean_text(text):
    """
    Clean text for better sentiment analysis.
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove Reddit formatting
    text = re.sub(r'/u/\w+', '', text) #Remove username/subreddit mentions
    text = re.sub(r'\*\*|__|\*|_', '', text) #Remove bold/italic markdown

    # Remove excess whitespace
    text = ' '.join(text.split())

    return text

def get_sentiment_scores(text):
    """
    Get sentiment scores using TextBlob and VADER.
    Returns: dict with multiple sentiment metrics.
    """
    if not text or text.strip() == '':
        return {
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'vader_compound': 0,
            'vader_positive': 0,
            'vader_negative': 0,
            'vader_neutral': 0
        }
    
    # TextBlob sentiment
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    textblob_subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

    # VADER sentiment (turns out it's better for social media text)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)

    return {
        'textblob_polarity': textblob_polarity,
        'textblob_subjectivity': textblob_subjectivity,
        'vader_positive': vader_scores['pos'],
        'vader_negative': vader_scores['neg'],
        'vader_neutral': vader_scores['neu'],
        'vader_compound': vader_scores['compound']  # Overall sentiment score
    }

def extract_contestant_mentions(text, contestants=None, couples=None):
    """
    Find which Love Island contestants and couples are mentioned in the text.
    """
    if contestants is None:
        contestants = [
            'megan', 'angel', 'cach', 'ty', 'jamie', 'yasmin', 'toni', 'conor', 'shakira', 'harry', 'meg', 'dejon'
        ]

    if couples is None:
        couples = [
            ('angel', 'ty'),
            ('cach', 'toni'),
            ('yasmin', 'jamie'),
            ('meg', 'dejon'),
            ('shakira', 'harry'),
            ('megan', 'conor')
        ]

    text_lower = text.lower()
    mentioned_contestants = []
    mentioned_couples = []

    # Check for individual contestant mentions
    for contestant in contestants:
        if re.search(r'\b' +re.escape(contestant.lower()) + r'\b', text_lower):
            mentioned_contestants.append(contestant.title())

    # Check for couple mentions
    for couple in couples:
        person1, person2 = couple

        # Check if both names appear in the text
        person1_mentioned = re.search(r'\b' + re.escape(person1.lower()) + r'\b', text_lower)
        person2_mentioned = re.search(r'\b' + re.escape(person2.lower()) + r'\b', text_lower)

        if person1_mentioned and person2_mentioned:
            # Check if they're mentioned together (within reasonable distance)
            person1_pos = person1_mentioned.start()
            person2_pos = person2_mentioned.start()
            
            # If mentioed within 50 characters of each other, consider it a couple mention
            if abs(person1_pos - person2_pos) <= 50:
                couple_name = f"{person1.title()} and {person2.title()}"
                mentioned_couples.append(couple_name)

        # Also check or common couple reference patterns
        couple_patterns = [
            f"{person1.title()} {person2.title()}",
            f"{person2.title()} {person1.title()}",
            f"{person1.title()} & {person2.title()}",
            f"{person2.title()} & {person1.title()}",
            f"{person1.title()} and {person2.title()}",
            f"{person2.title()} and {person1.title()}",
            f"{person1.title()}/{person2.title()}",
            f"{person2.title()}/{person1.title()}"
        ]

        for pattern in couple_patterns:
            if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text_lower):
                mentioned_couples.append(pattern)

    return {
        'individual_mentions': mentioned_contestants,
        'couple_mentions': mentioned_couples,
        'total_mentions': len(mentioned_contestants) + len(mentioned_couples)
    }

def analyze_sentiment(df):
    """
    Updated sentiment analysis with couple detection
    """
    print("Starting sentiment analysis...")

    # Clean the text data
    print("Cleaning text...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    df['cleaned_title'] = df['title'].apply(clean_text)

    # Combine title and content for analysis
    df['full_text'] = df['cleaned_title'] + ' ' + df['cleaned_content']

    # Get sentiment scores
    print("Calculating sentiment scores...")
    sentiment_data = []

    for index, row in df.iterrows():
        if index % 50 == 0:
            print(f"Processing row {index + 1}/{len(df)}")

        scores = get_sentiment_scores(row['full_text'])
        sentiment_data.append(scores)

    # Add sentiment columns to dataframe
    sentiment_df = pd.DataFrame(sentiment_data)
    df = pd.concat([df, sentiment_df], axis=1)

    # Extract contestant and couple mentions
    print("Identifying contestant and couple mentions...")

    mention_results = df['full_text'].apply(extract_contestant_mentions)
    
    # Add mention columns
    df['mentioned_individuals'] = mention_results.apply(lambda x: x['individual_mentions'])
    df['mentioned_couples'] = mention_results.apply(lambda x: x['couple_mentions'])
    df['total_mentions'] = mention_results.apply(lambda x: x['total_mentions'])

    # Create flags for easy filtering
    df['mentions_individuals'] = df['mentioned_individuals'].apply(lambda x: len(x) > 0)
    df['mentions_couples'] = df['mentioned_couples'].apply(lambda x: len(x) > 0)
    df['mentions_any'] = df['total_mentions'] > 0

    # Create overall sentiment category
    def categorize_sentiment(compound_score):
        if compound_score >= 0.1:
            return 'positive'
        elif compound_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
        
    df['sentiment_category'] = df['vader_compound'].apply(categorize_sentiment)

    print(f"Sentiment analysis complete!")
    print(f"Total posts/comments: {len(df)}")
    print(f"Posts mentioning individuals: {len(df[df['mentions_individuals']])}")
    print(f"Posts mentioning couples: {len(df[df['mentions_couples']])}")
    print(f"Posts mentioning any contestant/couple: {len(df[df['mentions_any']])}")
    print(f"\nSentiment breakdown:")
    print(f"Positive: {len(df[df['sentiment_category'] == 'positive'])}")
    print(f"Negative: {len(df[df['sentiment_category'] == 'negative'])}")
    print(f"Neutral: {len(df[df['sentiment_category'] == 'neutral'])}")

    return df

def main():
    """
    Load data and perform sentiment analysis.
    """

    # Load the most recent Reddit data
    data_files = [f for f in os.listdir('data/raw') if f.startswith('love_island_reddit')]
    if not data_files:
        print("No Reddit data found! Run reddit_scraper.py first.")
        return None
    
    latest_file = sorted(data_files)[-1]
    print(f"Loading data from: {latest_file}")

    df = pd.read_csv(f'data/raw/{latest_file}')
    print(f"Loaded {len(df)} posts and comments")

    # Perform sentiment analysis
    df_with_sentiment = analyze_sentiment(df)

    # Save the results
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/love_island_with_sentiment.csv'
    df_with_sentiment.to_csv(output_file, index=False)

    print(f"Saved sentiment analysis to: {output_file}")

    return df_with_sentiment

if __name__ == "__main__":
    df = main()