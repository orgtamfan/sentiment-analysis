# Reddit Comment Dataset: CS2 vs. VALORANT

## 1. Folder structure

```text
data/
  raw/
    +-- cs2_comments_raw.csv       # Raw comments from r/GlobalOffensive
    +-- valorant_comments_raw.csv  # Raw comments from r/VALORANT
    +-- reddit_comments_raw.csv    # Combined raw dataset (CS2 + VALORANT)
```

## 2. CSV schema

Each CSV file contains the following columns:

- `game`: Game label (`CS2` or `VALORANT`)
- `subreddit`: Source subreddit (`GlobalOffensive` or `VALORANT`)
- `post_id`: Reddit post ID
- `post_title`: Post or thread title
- `post_url`: Post URL
- `comment_id`: Unique comment ID
- `comment_text`: Original comment text (no lowercasing, stop-word removal, stemming, or punctuation removal)
- `comment_score`: Net upvote/downvote score
- `comment_created_at`: Comment creation time in ISO 8601 UTC format
- `author`: Comment author's username (metadata)
- `scraped_at`: Scrape timestamp in UTC

## 3. Data integrity notes

- Comment text is preserved in its original form for later preprocessing.
- Comments marked `[deleted]` or `[removed]` were automatically filtered during collection.
- Duplicate `comment_id` values were automatically removed.
