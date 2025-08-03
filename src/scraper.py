# import snscrape.modules.twitter as sntwitter
import praw
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def collect_reddit_posts(subreddit_name="Love Island TV", limit=200):
    """
    Collect Love Island posts and comments from Reddit.

    Args:
        subreddit_name: str: The name of the subreddit to scrape.
        limit: int: Number of posts to collect.

    Returns:
        DataFrame with posts and coments.
    """
    
    print(f"Collecting data from r/{subreddit_name}")

    # Calculate cutoff date (1 week ago)
    week_ago = datetime.now() - timedelta(days=7)
    
    # Initializing a Reddit instance (read-only, no authentication required)
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="LoveIslandSentimentAnalysis"
    )

    # Access the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    all_data = []

    try:
        # Get hot posts (most active discussions)
        print("Collecting hot posts...")
        for post in subreddit.hot(limit=limit//2):
            post_date = datetime.fromtimestamp(post.created_utc)

            # Skip posts older than 1 week
            if post_date < week_ago:
                continue
            try:
                post_data = {
                    'date': datetime.fromtimestamp(post.created_utc),
                    'type': 'post',
                    'id': getattr(post, 'id', 'unknown'),
                    'title': getattr(post, 'title', ''),
                    'content': post.selftext if hasattr(post, 'selftext') and post.selftext else post.title,
                    'author': str(post.author) if post.author else "[deleted]",
                    'score': getattr(post, 'score', 0),
                    'upvotes': getattr(post, 'ups', 0),
                    'comments_count': getattr(post, 'num_comments', 0),
                    'url': getattr(post, 'url', ''),
                    'subreddit': subreddit_name
                }
                all_data.append(post_data)

                # Get top comments for esch posts
                post.comments.replace_more(limit=0) # Remove load more comments
                for comment in post.comments.list()[:5]: # Get top 5 comments
                    comment_date = datetime.fromtimestamp(comment.created_utc)

                    # Skip old comments
                    if comment_date < week_ago:
                        continue
                    
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        # Check if comment already exists
                        if any(item['id'] == comment.id for item in all_data if item['type'] == 'comment'):
                            continue
                        comment_data = {
                            'date': datetime.fromtimestamp(comment.created_utc),
                            'type': 'comment',
                            'id': comment.id,
                            'title': f"Re: {post.title[:50]}...",
                            'content': comment.body,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'score': comment.score,
                            'upvotes': comment.ups,
                            'comments_count': 0,
                            'url': f"https://reddit.com{comment.permalink}",
                            'subreddit': subreddit_name
                        }
                        all_data.append(comment_data)

            except Exception as e:
                print(f"Skipping problematic post: {e}")
                continue

            time.sleep(0.5) # Be respectful to Reddit's servers, lol

        # Get new posts (latest discussions)
        print("Collecting new posts...")
        for post in subreddit.new(limit=limit//2):
            post_date = datetime.fromtimestamp(post.created_utc)

            # Skip posts older than 1 week
            if post_date < week_ago:
                continue

            #check if post already exists
            if not any(item.get('id') == getattr(post, 'id', None) for item in all_data):
                try:
                    post_data = {
                        'date': datetime.fromtimestamp(post.created_utc),
                        'type': 'post',
                        'id': post.id,
                        'title': post.title,
                        'content': post.selftext if post.selftext else post.title,
                        'author': str(post.author) if post.author else '[deleted]',
                        'score': post.score,
                        'upvotes': post.ups,
                        'comments_count': post.num_comments,
                        'url': post.url,
                        'subreddit': subreddit_name
                    }
                    all_data.append(post_data)
                
                except Exception as e:
                    print(f"Skipping problematic post: {e}")
                    continue

                time.sleep(0.5) # Another moment to be respectful to Reddit's servers

    except Exception as e:
        print(f"Skipping problematic post: {e}")

    print(f"Collected {len(all_data)} posts and comments")
    return all_data

   
def main():
    """
    main function to collect Love Island Reddit data
    """

    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # List of Love Island related subreddits
    subreddits = [
        "LoveIsland",
        "LoveIslandTV",
        "LoveIslandITV"
    ]

    all_reddit_data = []

    # Collect from each subreddit
    for subreddit in subreddits:
        try:
            data = collect_reddit_posts(subreddit_name=subreddit, limit=100)
            all_reddit_data.extend(data)
            print(f"Collected {len(data)} items from r/{subreddit}")
            time.sleep(2)  # Pause between subreddits
        except Exception as e:
            print(f"Could not collect from r/{subreddit}: {e}")
            continue

    # Convert to DataFrame and save
    if all_reddit_data:
        df = pd.DataFrame(all_reddit_data)
        
        #Remove duplicates
        df = df.drop_duplicates(subset=['id'])

        df['date'] = pd.to_datetime(df['date'])
        # Sort by date (newest first)
        df = df.sort_values(by='date', ascending=False)

        # Save raw data to CSV
        timestamp = datetime.now().strftime("%Y-%m-$d_%H-%M-%S")
        filename = f"data/raw/love_island_reddit_data_{timestamp}.csv"
        df.to_csv(filename, index=False)

        print(f"\nSaved {len(df)} unique posts/comments to {filename}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Posts: {len(df[df['type'] == 'post'])}")
        print(f"Comments: {len(df[df['type'] == 'comment'])}")

        # Show sample content
        print(f"\nSample recent posts:")
        recent_posts = df[df['type'] == 'post'].head(3)
        for _, post in recent_posts.iterrows():
            print(f"- {post['title']} (Score: {post['score']})")

        return df
    else:
        print("No data collected!")
        return None


if __name__ == "__main__":
    df = main()