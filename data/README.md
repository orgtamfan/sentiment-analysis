# Reddit Comments Dataset: CS2 vs. VALORANT

This dataset was collected for the research project:
**"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

## 1. Folder structure

```text
data/
+-- raw/                                        # gitignored — raw scrape output (contains author usernames)
|   +-- reddit_comments_raw.json               # Combined raw dataset (CS2 + VALORANT)
|   +-- cs2_comments_raw.json                  # Raw comments from r/GlobalOffensive
|   +-- valorant_comments_raw.json             # Raw comments from r/VALORANT
+-- processed/
    +-- reddit_comments_label_review.csv       # gitignored — VADER suggestions pending review
    +-- reddit_comments_labeled.csv            # gitignored — VADER auto-labeled dataset (full, 10 000 rows)
    +-- manual_label.csv                       # gitignored — 3-annotator manual annotation dataset (600 rows)
    +-- reddit_comments_labeled_analysis/      # committed — report snapshot: full 5 000/game VADER baseline run
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
> `data/raw/` and processed comment CSVs are gitignored to preserve user privacy (anonymization and avoiding distributing raw social media scrapes with personal usernames). Pre-computed aggregate report snapshots (JSON + CSV) and the model code are committed for research transparency and reproducibility.

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

The labeled CSVs contain:

| Column | Values |
|---|---|
| `game` | `cs2` or `valorant` |
| `text` | Comment text |
| `sentiment` | `positive`, `neutral`, or `negative` |
| `sentiment_suggestion` | VADER suggestion (for reference) |
| `vader_compound` | VADER compound score (−1 to 1) |
| `review_status` | `vader_auto` (automatic VADER labels), `needs_review`, or `reviewed` (human adjudicated) |

## 4. Data integrity notes

- Comment text is preserved as originally written (raw) for subsequent preprocessing.
- Comments marked `[deleted]` and `[removed]` were automatically filtered during collection.
- Duplicate `comment_id` values were automatically deduplicated.
