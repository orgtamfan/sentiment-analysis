# Sentiment Analysis of CS2 and VALORANT Communities on Reddit

Research workspace for **"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

This project uses Reddit comments as the unit of analysis. VADER generates **initial label suggestions** for review, while TF-IDF and Multinomial Naive Bayes are used as a three-class classification baseline: `positive`, `neutral`, and `negative`.

## Workspace structure

```text
data/
  raw/                  # Raw Reddit data, for example reddit_comments_raw.json
  processed/            # CSV files with label suggestions and final labels
reports/                # Label distribution and model evaluation reports
src/
  prepare_labels.py     # Generate VADER label suggestions for review
  sentiment_analysis.py # Train and evaluate a Naive Bayes model per game
tests/                  # Automated tests
README.md
requirements.txt
pyproject.toml
```

Local data in `data/raw/` and `data/processed/`, and generated output in `reports/`, are ignored by Git. Do not commit datasets containing personal information or data you are not authorized to publish.

## Environment setup

Use Python 3.10 or later. In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If environment activation is blocked, use `.\.venv\Scripts\python.exe` to run Python commands.

## 1. Prepare label suggestions

Place the raw JSON file at `data/raw/reddit_comments_raw.json`. The JSON must be a list of comment objects containing at least the `game` and `comment_text` fields. Recognized game names are `CS2` and `VALORANT`.

Run:

```powershell
python -m src.prepare_labels
```

By default, the script reads `data/raw/reddit_comments_raw.json` and creates `data/processed/reddit_comments_label_review.csv`. Specify alternate input and output files with:

```powershell
python -m src.prepare_labels --input data/raw/reddit_comments_raw.json --output data/processed/reddit_comments_label_review.csv
```

The review CSV contains:

- `sentiment_suggestion`: VADER suggestion (`positive`, `neutral`, or `negative`)
- `vader_compound`: VADER score from -1 to 1
- `sentiment`: intentionally blank for the final label after review
- `review_status`: initially set to `needs_review`
- `game`, `text`, and comment metadata

Open the CSV in a spreadsheet. Compare each suggestion with the comment and its context, fill in the `sentiment` column, and then change `review_status` to `reviewed`. Save the completed copy as `data/processed/reddit_comments_labeled.csv`. Do not overwrite the raw JSON. The script also will not overwrite an existing review CSV.

**Important:** VADER is a lexicon-based aid, not an infallible annotator. Gaming slang, sarcasm, very short comments, and off-topic text can produce incorrect suggestions. Labels copied directly from VADER are not independent ground truth; evaluating a model against them measures agreement with VADER suggestions, not accuracy against validated human annotations.

## 2. Labeled CSV format

Required columns for analysis:

| Column | Values |
|---|---|
| `game` | `cs2` or `valorant` |
| `text` | Anonymized post or comment text |
| `sentiment` | `positive`, `neutral`, or `negative` |

Other columns may be retained. If a `review_status` column is present, analysis will reject rows still marked `needs_review`. Both games must be present, and each sentiment class needs at least two examples per game. A larger sample may be needed for an 80:20 stratified split so every class appears in both training and test sets.

## 3. Training and evaluation

Run the analysis with an 80% training and 20% test stratified split:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --test-size 0.2 --random-state 42
```

By default, reports are saved to:

- `reports/sentiment_distribution.csv` - label counts and proportions by game.
- `reports/model_metrics.json` - accuracy, macro/weighted F1, per-class reports, and confusion matrices.

Models are built and evaluated separately for CS2 and VALORANT. Use comparable collection, filtering, and annotation procedures when comparing the two communities.

## Most recent recorded run

The most recent saved run used 5,000 comments per game, split into 4,000 training and 1,000 test comments for each game, with `random_state=42`.

| Game | Accuracy | Macro F1 |
|---|---:|---:|
| CS2 | 0.483 | 0.304 |
| VALORANT | 0.512 | 0.268 |

The baseline model has limited performance, especially on negative and neutral classes. See the confusion matrices and per-class metrics in `reports/model_metrics.json`; do not rely on accuracy alone. This run reflects the dataset labels used at the time of analysis and does not automatically represent independent human annotations.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

## Research limitations

- Naive Bayes learns from the supplied labels; it does not turn automatic labels into ground truth.
- A random stratified split is not an evaluation across time or authors. Duplicate comments or comments from the same post may cause the training and test data to be correlated.
- TF-IDF and Multinomial Naive Bayes are baseline methods. Document preprocessing choices and limitations involving language, slang, negation, and sarcasm.
- Comment sentiment does not establish a causal relationship between a game and community opinion.
- Follow Reddit's terms, subreddit rules, and applicable ethics requirements; minimize the storage of personal information.
