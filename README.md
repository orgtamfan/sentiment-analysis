# Sentiment Analysis of CS2 and VALORANT Communities on Reddit

Research workspace for **"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

This project uses Reddit comments as the unit of analysis. VADER generates **initial label suggestions** for review, while TF-IDF and Multinomial Naive Bayes are used as a three-class classification baseline: `positive`, `neutral`, and `negative`. Inter-annotator agreement is evaluated using **Cohen's Kappa** (pairwise) and **Fleiss' Kappa** (multi-rater).

## Workspace structure

```text
data/
  raw/                                        # gitignored — raw scrape output (JSON)
  processed/
    reddit_comments_label_review.csv          # gitignored — VADER suggestions pending review
    reddit_comments_labeled.csv               # gitignored — VADER auto-labeled dataset (full)
    manual_label.csv                          # gitignored — 3-annotator manual annotation dataset
    reddit_comments_labeled_analysis/         # committed — report snapshot: full 5 000/game VADER baseline run
    vader_subset_600_for_repo/                # committed — report snapshot: 300/game VADER-labeled subset
    manual_labeled_for_repo/                  # committed — report snapshot: 300/game manually labeled subset
reports/                                      # gitignored — regenerated on each local run
src/
  prepare_labels.py             # Generate VADER label suggestions for review
  sentiment_analysis.py         # Train and evaluate a Naive Bayes model per game
  inter_annotator_agreement.py  # Cohen's Kappa, Fleiss' Kappa, and VADER agreement rate
tests/                          # Automated unit tests
README.md
requirements.txt
pyproject.toml
```

Raw data in `data/raw/`, full comment CSVs, and the generated `reports/` directory are gitignored to preserve user privacy. Pre-computed report snapshots in `data/processed/*/` are committed for reproducibility. Do not commit datasets containing personal usernames or data you are not authorized to publish.

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

Open the CSV in a spreadsheet. Compare each suggestion with the comment and its context, fill in the `sentiment` column, and then change `review_status` to `reviewed` (or `vader_auto` if keeping automated labels). Save the completed copy as `data/processed/reddit_comments_labeled.csv`. Do not overwrite the raw JSON. The script also will not overwrite an existing review CSV.

**Important:** VADER is a lexicon-based aid, not an infallible annotator. Gaming slang, sarcasm, very short comments, and off-topic text can produce incorrect suggestions. Labels copied directly from VADER are not independent ground truth; evaluating a model against them measures agreement with VADER suggestions, not accuracy against validated human annotations.

## 2. Inter-Annotator Agreement (Cohen's Kappa & Fleiss' Kappa)

To measure the reliability and agreement between multiple annotators (e.g. 3 annotators on a sample subset of 600 comments), run:

```powershell
python -m src.inter_annotator_agreement --input data/processed/manual_label.csv --by-game
```

This calculates:
- **Pairwise Cohen's Kappa** using `scikit-learn` across all pairs of annotators.
- **Fleiss' Kappa** across all annotators simultaneously using `statsmodels`.
- **Agreement rate** between VADER suggestions and final consensus manual labels (Rumusan Masalah #4).
- Automatic benchmark interpretation based on **Landis & Koch (1977)**:
  - `< 0.00` : Poor (worse than chance)
  - `0.00 – 0.20` : Slight
  - `0.21 – 0.40` : Fair
  - `0.41 – 0.60` : Moderate
  - `0.61 – 0.80` : Substantial
  - `0.81 – 1.00` : Almost Perfect

### Annotator agreement on 600 sample comments:

| Metric | Overall | CS2 (N=300) | VALORANT (N=300) |
|---|---|---|---|
| **Mean Cohen's Kappa** | 0.4098 (Moderate) | 0.3715 (Fair) | 0.4431 (Moderate) |
| **Fleiss' Kappa** | 0.4067 (Moderate) | 0.3693 (Fair) | 0.4381 (Moderate) |

Pairwise Cohen's Kappa breakdown:
- Annotator 1 vs. Annotator 2: **0.5483** (Moderate)
- Annotator 1 vs. Annotator 3: **0.3119** (Fair)
- Annotator 2 vs. Annotator 3: **0.3692** (Fair)

## 3. Labeled CSV format

Required columns for analysis:

| Column | Values |
|---|---|
| `game` | `cs2` or `valorant` |
| `text` | Comment text |
| `sentiment` | `positive`, `neutral`, or `negative` |

Other columns may be retained. If a `review_status` column is present, analysis will reject rows still marked `needs_review` (accepted values: `reviewed` or `vader_auto`). Both games must be present, and each sentiment class needs at least two examples per game. A larger sample may be needed for an 80:20 stratified split so every class appears in both training and test sets.

## 4. Training and evaluation

Run the analysis with an 80% training and 20% test stratified split:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --test-size 0.2 --random-state 42
```

By default, reports are saved to:

- `reports/sentiment_distribution.csv` - label counts and proportions by game.
- `reports/model_metrics.json` - accuracy, macro/weighted F1, per-class reports, and confusion matrices.

Models are built and evaluated separately for CS2 and VALORANT. Use comparable collection, filtering, and annotation procedures when comparing the two communities.

To reproduce a committed snapshot, point `--output-dir` at the corresponding subfolder:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --output-dir data/processed/reddit_comments_labeled_analysis
```

## Recorded runs

Three model snapshots are committed under `data/processed/`. All runs use TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes, 80/20 stratified split, `random_state=42`.

### 1. Full dataset (VADER automated labels) — `reddit_comments_labeled_analysis/`

5,000 comments per game (4,000 train / 1,000 test) labeled using VADER lexicon suggestions (`vader_auto`) to establish a large-scale automated baseline.

**Label distribution:**

| Game | Negative | Neutral | Positive |
|---|---:|---:|---:|
| CS2 | 1,381 (27.6%) | 1,414 (28.3%) | 2,205 (44.1%) |
| VALORANT | 1,225 (24.5%) | 1,312 (26.2%) | 2,463 (49.3%) |

**Model results:**

| Game | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| CS2 | 0.483 | 0.304 | 0.357 |
| VALORANT | 0.512 | 0.268 | 0.365 |

### 2. VADER-labeled subset — `vader_subset_600_for_repo/`

300 comments per game (240 train / 60 test), labels taken directly from VADER suggestions without manual review.

**Label distribution:**

| Game | Negative | Neutral | Positive |
|---|---:|---:|---:|
| CS2 | 91 (30.3%) | 86 (28.7%) | 123 (41.0%) |
| VALORANT | 79 (26.3%) | 80 (26.7%) | 141 (47.0%) |

**Model results:**

| Game | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| CS2 | 0.450 | 0.288 | 0.329 |
| VALORANT | 0.467 | 0.212 | 0.297 |

### 3. Manually labeled subset — `manual_labeled_for_repo/`

300 comments per game (240 train / 60 test), labels assigned through manual annotation and consensus.

**Label distribution:**

| Game | Negative | Neutral | Positive |
|---|---:|---:|---:|
| CS2 | 64 (21.3%) | 135 (45.0%) | 101 (33.7%) |
| VALORANT | 58 (19.3%) | 111 (37.0%) | 131 (43.7%) |

**Model results:**

| Game | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| CS2 | 0.467 | 0.283 | 0.355 |
| VALORANT | 0.400 | 0.193 | 0.251 |

---

The baseline model has limited performance across all runs, especially on negative and neutral classes. See the confusion matrices and per-class metrics in the corresponding `model_metrics.json` files; do not rely on accuracy alone. Results reflect the dataset labels used at the time of analysis and do not automatically represent independent human annotations.

## Run the tests

Execute the automated test suite:

```powershell
python -m unittest discover -s tests -v
```

## Research limitations

- Naive Bayes learns from the supplied labels; it does not turn automatic labels into ground truth.
- A random stratified split is not an evaluation across time or authors. Duplicate comments or comments from the same post may cause the training and test data to be correlated.
- TF-IDF and Multinomial Naive Bayes are baseline methods. Document preprocessing choices and limitations involving language, slang, negation, and sarcasm.
- Comment sentiment does not establish a causal relationship between a game and community opinion.
- Follow Reddit's terms, subreddit rules, and applicable ethics requirements; minimize the storage of personal information.
