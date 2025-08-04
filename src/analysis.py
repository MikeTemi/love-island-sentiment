import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def load_sentiment_data():
    """
    Load the processed sentiment data
    """
    filepath = 'data/processed/love_island_with_sentiment.csv'
    if not os.path.exists(filepath):
        print("No sentiment data found! Run sentiment.py first.")
        return None
    
    df = pd.read_csv(filepath)
    print(f"loaded {len(df)} posts/comments with sentiment analysis")
    return df

def analyze_individual_popularity(df):
    """
    Analyze individual contestant popularity and sentiment.
    """
    print("\n=== INDIVIDUAL CONTESTANT ANALYSIS ===")

    # Get all individual mentions
    individual_mentions = []
    for mentions_str in df['mentioned_individuals'].dropna():
        if mentions_str != '[]': # Skip empty lists
            # Parse the string representation of the list
            mentions = eval(mentions_str) if isinstance(mentions_str, str) else mentions_str
            individual_mentions.extend(mentions)

    if not individual_mentions:
        print("No individual contestant mentions found.")
        return None
    
    # Count mentions per contestant
    mention_counts = Counter(individual_mentions)

    print("Individual Contestant Mention Counts:")
    for contestant, count in mention_counts.most_common():
        print(f"    {contestant}: {count} mentions")

    # Analyze sentiment for each contestant
    contestant_sentiment = {}

    for contestant in mention_counts.keys():
        # Filter posts that mention this contestant
        contestant_posts = df[df['mentioned_individuals'].str.contains(contestant, na=False)]
                              
        if len(contestant_posts) > 0:
            avg_sentiment = contestant_posts['vader_compound'].mean()
            positive_posts = len(contestant_posts[contestant_posts['sentiment_category'] == 'positive'])
            negative_posts = len(contestant_posts[contestant_posts['sentiment_category'] == 'negative'])
            neutral_posts = len(contestant_posts[contestant_posts['sentiment_category'] == 'neutral'])

            # Weight by engagement (posts with more upvotes matter more)
            weighted_sentiment = (contestant_posts['vader_compound'] * contestant_posts['score']).sum() / contestant_posts['score'].sum() if contestant_posts['score'].sum() > 0 else avg_sentiment

            contestant_sentiment[contestant] = {
                'mentions': mention_counts[contestant],
                'avg_sentiment': avg_sentiment,
                'weighted_sentiment': weighted_sentiment,
                'positive_posts': positive_posts,
                'negative_posts': negative_posts,
                'neutral_posts': neutral_posts,
                'positivity_ratio': positive_posts / len(contestant_posts) if len(contestant_posts) > 0 else 0
            }

    return contestant_sentiment

def analyze_couple_popularity(df):
    """
    Analyze couple popularity and sentiment.
    """
    print("\n=== COUPLE ANALYSIS ===")

    # Get all couple mentions
    couple_mentions = []
    for mentions_str in df['mentioned_couples'].dropna():
        if mentions_str != '[]': # Skip empty lists
            # Parse the string representation of the list
            mentions = eval(mentions_str) if isinstance(mentions_str, str) else mentions_str
            couple_mentions.extend(mentions)

    if not couple_mentions:
        print("No couple mentions found.")
        return None
    
    # Count mentions per couple
    couple_counts = Counter(couple_mentions)

    print("Couple Mention Counts:")
    for couple, count in couple_counts.most_common():
        print(f"    {couple}: {count} mentions")

    # Analyze sentiment for each couple
    couple_sentiment = {}

    for couple in couple_counts.keys():
        # Filter posts that mention this couple
        couple_posts = df[df['mentioned_couples'].str.contains(couple, na=False)]
                              
        if len(couple_posts) > 0:
            avg_sentiment = couple_posts['vader_compound'].mean()
            positive_posts = len(couple_posts[couple_posts['sentiment_category'] == 'positive'])
            negative_posts = len(couple_posts[couple_posts['sentiment_category'] == 'negative'])
            neutral_posts = len(couple_posts[couple_posts['sentiment_category'] == 'neutral'])

            # Weight by engagement (posts with more upvotes matter more)
            weighted_sentiment = (couple_posts['vader_compound'] * couple_posts['score']).sum() / couple_posts['score'].sum() if couple_posts['score'].sum() > 0 else avg_sentiment

            couple_sentiment[couple] = {
                'mentions': couple_counts[couple],
                'avg_sentiment': avg_sentiment,
                'weighted_sentiment': weighted_sentiment,
                'positive_posts': positive_posts,
                'negative_posts': negative_posts,
                'neutral_posts': neutral_posts,
                'positivity_ratio': positive_posts / len(couple_posts) if len(couple_posts) > 0 else 0
            }

    return couple_sentiment

def predict_winner(individual_sentiment, couple_sentiment):
    """
    Predict Love Island winner based on sentiment analysis.
    """
    print("\n=== WINNER PREDICTION ===")

    if couple_sentiment:
        print("Couple Rankings (by combined score):")

        couple_scores = {}
        for couple, data in couple_sentiment.items():
            # Combined score considering mentions, sentiment, and positivity
            popularity_score = data['mentions'] * 0.4 # 40% weight to popularity
            sentiment_score = (data['weighted_sentiment'] + 1) * 50 * 0.3 # 30% weight to sentiment(normalized to 0-100)
            positivity_score = data['positivity_ratio'] * 100 * 0.3 # 30% weight to positivity ratio

            combined_score = popularity_score + sentiment_score + positivity_score

            couple_scores[couple] = {
                'combined_score': combined_score,
                'popularity_score': popularity_score,
                'sentiment_score': sentiment_score,
                'positivity_score': positivity_score,
                'data': data
            }

            # Sort couples by combined score
            ranked_couples = sorted(couple_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)

            for i, (couple, scores) in enumerate(ranked_couples, 1):
                print(f"\n{i}. {couple}")
                print(f"   Combined Score: {scores['combined_score']:.2f}")
                print(f"   Mentions: {scores['data']['mentions']}")
                print(f"   Avg Sentiment: {scores['data']['avg_sentiment']:.3f}")
                print(f"   Positivity Ratio: {scores['data']['positivity_ratio']:.1%}")

            if ranked_couples:
                winner = ranked_couples[0][0]
                print(f"\n PREDICTED WINNER: {winner}")
                return winner
            
    return None

def create_visualizations(df, individual_sentiment, couple_sentiment):
    """Create visualizations of the analysis"""
    print("\n=== CREATING VISUALIZATIONS ===")

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Love Island Sentiment Analysis', fontsize=16, fontweight='bold')

    # 1. Overall sentiment distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Overall Sentiment Distribution')

    # 2. Individual contestant sentiment (if available)
    if individual_sentiment:
        contestants = list(individual_sentiment.keys())[:6]  # Top 6
        sentiments = [individual_sentiment[c]['avg_sentiment'] for c in contestants]

        bars = axes[0, 1].bar(contestants, sentiments)
        axes[0, 1].set_title('Individual Contestant Sentiment')
        axes[0, 1].set_ylabel('Average Sentiment Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Color bars based on sentiment
        for bar, sentiment in zip(bars, sentiments):
            if sentiment > 0.1:
                bar.set_color('green')
            elif sentiment < -0.1:
                bar.set_color('red')
            else:
                bar.set_color('gray')

    # 3. Couple sentiment (if available)
    if couple_sentiment:
        couples = list(couple_sentiment.keys())
        couple_sentiments = [couple_sentiment[c]['avg_sentiment'] for c in couples]

        bars = axes[1, 0].bar(range(len(couples)), couple_sentiments)
        axes[1, 0].set_title('Couple Sentiment Scores')
        axes[1, 0].set_ylabel('Average Sentiment Score')
        axes[1, 0].set_xticks(range(len(couples)))
        axes[1, 0].set_xticklabels([c.replace(' and ', '\n&\n') for c in couples], fontsize=8)

        for bar, sentiment in zip(bars, couple_sentiments):
            if sentiment > 0.1:
                bar.set_color('green')
            elif sentiment < -0.1:
                bar.set_color('red')
            else:
                bar.set_color('gray')

    # 4. Mentions over time
    df['date'] = pd.to_datetime(df['date'])
    daily_mentions = df[df['mentions_any'] == True].groupby(df['date'].dt.date).size()

    axes[1, 1].plot(daily_mentions.index, daily_mentions.values, marker='o')
    axes[1, 1].set_title('Love Island Mentions Over Time')
    axes[1, 1].set_ylabel('Number of Mentions')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save the plot
    os.makedirs('data/results', exist_ok=True)
    plt.savefig('data/results/love_island_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to data/results/love_island_sentiment_analysis.png")
    
    plt.show()

def main():
    """Main analysis function"""
    # Load the sentiment data
    df = load_sentiment_data()
    if df is None:
        return
    
    # Analyze individual contestant popularity
    individual_sentiment = analyze_individual_popularity(df)

    # Analyze couple popularity
    couple_sentiment = analyze_couple_popularity(df)

    # Predict the winner based on sentiment analysis
    predicted_winner = predict_winner(individual_sentiment, couple_sentiment)

    # Create visualizations
    create_visualizations(df, individual_sentiment, couple_sentiment)

    return {
        'individual_sentiment': individual_sentiment,
        'couple_sentiment': couple_sentiment,
        'predicted_winner': predicted_winner
    }

if __name__ == "__main__":
    results = main()