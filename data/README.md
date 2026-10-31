# Reddit Comments Dataset: CS2 vs. VALORANT

This dataset was collected for the research project:
**"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

## 1. Folder structure

```text
data/
+-- raw/                                        # gitignored — raw scrape output
|   +-- reddit_comments_raw.json               # Combined raw dataset (CS2 + VALORANT)
|   +-- cs2_comments_raw.json                  # Raw comments from r/GlobalOffensive
|   +-- valorant_comments_raw.json             # Raw comments from r/VALORANT
+-- processed/
    +-- reddit_comments_label_review.csv       # gitignored — VADER suggestions pending review
    +-- reddit_comments_labeled.csv            # gitignored — final labeled dataset (full, 10 000 rows)
    +-- reddit_comments_labeled_analysis/      # committed — report snapshot: full 5 000/game run
    |   +-- model_metrics.json
    |   +-- sentiment_distribution.csv
    +-- vader_subset_600_for_repo/             # committed — report snapshot: 300/game VADER-labeled subset
    |   +-- model_metrics.json
    |   +-- sentiment_distribution.csv
    +-- manual_labeled_for_repo/               # committed — report snapshot: 300/game manually labeled subset
        +-- model_metrics.json
        +-- sentiment_distribution.csv
```

> [!NOTE]
> `data/raw/` and the two top-level CSVs are gitignored because they contain raw user-generated text. Only pre-computed report snapshots (JSON + CSV) are committed.

## 2. Raw JSON schema

Each JSON file is a list of comment objects with the following fields:

| Field | Description |
|---|---|
| `game` | Game label (`CS2` or `VALORANT`) |
| `subreddit` | Source subreddit (`GlobalOffensive` or `VALORANT`) |
| `post_id` | Reddit post ID |
| `post_title` | Reddit thread or post title |
| `post_url` | Reddit post URL |
| `comment_id` | Unique comment ID |
| `comment_text` | Raw comment text (no preprocessing applied) |
| `comment_score` | Net upvote/downvote score |
| `comment_created_at` | Comment creation time (ISO 8601 UTC) |
| `author` | Comment author username |
| `scraped_at` | Timestamp when scraping was performed (UTC) |

## 3. Processed CSV schema

The labeled CSVs (`reddit_comments_label_review.csv` and `reddit_comments_labeled.csv`) contain:

| Column | Values |
|---|---|
| `game` | `cs2` or `valorant` |
| `text` | Anonymized comment text |
| `sentiment` | `positive`, `neutral`, or `negative` |
| `sentiment_suggestion` | VADER suggestion (for reference) |
| `vader_compound` | VADER compound score (−1 to 1) |
| `review_status` | `reviewed` or `needs_review` |

## 4. Data integrity notes

- Comment text is preserved as originally written (raw) for subsequent preprocessing.
- Comments marked `[deleted]` and `[removed]` were automatically filtered during collection.
- Duplicate `comment_id` values were automatically deduplicated.
