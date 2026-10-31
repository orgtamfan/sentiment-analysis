# Reddit Comments Dataset: CS2 vs. VALORANT

This raw dataset was collected for the research project:
**"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

## 1. Folder structure

```text
data/
+-- raw/
    +-- cs2_comments_raw.csv       # Raw comments from r/GlobalOffensive
    +-- valorant_comments_raw.csv  # Raw comments from r/VALORANT
    +-- reddit_comments_raw.csv    # Combined raw dataset (CS2 + VALORANT)
```

## 2. CSV schema

Each CSV file has the following columns:

- `game`: Game label (`CS2` or `VALORANT`)
- `subreddit`: Source subreddit (`GlobalOffensive` or `VALORANT`)
- `post_id`: Reddit post ID
- `post_title`: Reddit thread or post title
- `post_url`: Reddit post URL
- `comment_id`: Unique comment ID
- `comment_text`: Raw comment text (no lowercasing, stop-word removal, stemming, or punctuation removal)
- `comment_score`: Net upvote/downvote score for the comment
- `comment_created_at`: Comment creation time (ISO 8601 UTC format)
- `author`: Comment author's username (metadata)
- `scraped_at`: Timestamp when scraping was performed (UTC)

## 3. Data integrity notes

- Comment text is preserved as originally written (raw) for subsequent preprocessing.
- Comments marked `[deleted]` and `[removed]` were automatically filtered out during collection.
- Duplicate `comment_id` values were automatically filtered out.
