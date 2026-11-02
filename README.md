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
  sentiment_analysis.py         # Train and evaluate Naive Bayes & majority baseline models
  inter_annotator_agreement.py  # Cohen's Kappa, Fleiss' Kappa, and VADER agreement rate
tests/
  test_prepare_labels.py             # Tests for label suggestion pipeline & JSON parsing
  test_sentiment_analysis.py         # Tests for dataset validation, duplicates, and split logic
  test_inter_annotator_agreement.py  # Tests for Kappa calculations, normalization, and metrics
README.md
requirements.txt
pyproject.toml
```

Raw data in `data/raw/`, full comment CSVs, and the generated `reports/` directory are gitignored to preserve user privacy. Pre-computed report snapshots in `data/processed/*/` are committed for inspection, but the source CSVs are not included, so the reported experiments cannot currently be reproduced from this repository alone. Do not commit datasets containing personal usernames or data you are not authorized to publish.

## Environment setup

Use Python 3.10 or later. In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Or install the package in editable/development mode:

```powershell
pip install -e ".[dev]"
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

Key methodological features:
- **Label Normalization:** Leading/trailing whitespaces and casing are cleaned, empty strings are treated as missing values (`pd.NA`), and labels are strictly validated against `{'positive', 'neutral', 'negative'}`.
- **Pairwise Cohen's Kappa (`scikit-learn`):** Evaluates agreement for every pair of annotators using all available shared ratings (without dropping rows if an un-involved third rater is missing).
- **Fleiss' Kappa (`statsmodels`):** Evaluates multi-rater categorical agreement simultaneously using all subjects with complete ratings.
- **Agreement rate:** Measures the percentage match between VADER suggestions and final consensus labels (Research Question 4).
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
- Annotator 1 vs. Annotator 2: **0.5483** (Moderate) [N=600]
- Annotator 1 vs. Annotator 3: **0.3119** (Fair) [N=600]
- Annotator 2 vs. Annotator 3: **0.3692** (Fair) [N=600]

## 3. Labeled CSV format & data integrity

Required columns for analysis:

| Column | Values |
|---|---|
| `game` | `cs2` or `valorant` |
| `text` | Comment text |
| `sentiment` | `positive`, `neutral`, or `negative` |

Additional data integrity checks in `load_dataset()`:
- **`comment_id` uniqueness:** Rejects datasets with duplicate comment IDs.
- **Duplicate text consistency:** Detects identical comment texts and ensures they do not have conflicting sentiment labels.
- **Strict `review_status` validation:** Enforces valid values (`reviewed` or `vader_auto`). Rejects missing, case-mismatched, or unreviewed rows (`needs_review`).

## 4. Training and evaluation

Run the analysis with group-aware splitting (prevents post-level data leakage):

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --split-strategy auto --test-size 0.2 --random-state 42
```

### Evaluation methodology:
- **Leak-free group splitting (`StratifiedGroupKFold` + text-clustering):** When `post_id` is present, comments from the same Reddit thread are kept together. Furthermore, identical comments appearing across multiple posts are clustered into the same group via disjoint-set union, strictly guaranteeing zero text overlap and zero post overlap between train and test sets.
- **Split integrity validation:** After splitting, the pipeline explicitly asserts that all 3 sentiment classes exist in both training and test sets, and verifies zero post_id and text intersections.
- **Auditable fallback reporting:** If group splitting cannot be used (e.g., missing or insufficient `post_id`), `split_info` records an explicit `fallback_reason` (e.g., `"missing 'post_id' column"`).
- **Leak-free text fallback:** Without usable post IDs, identical texts are assigned to the same `StratifiedGroupKFold` cluster. The analysis stops if there are too few independent clusters or if any class is absent from train or test.
- **Reproducibility metadata:** `model_metrics.json` records the split strategy, group/cluster counts, train/test sizes and class counts, requested and achieved test fractions, random seed, and versioned model configuration for each game.
- **Majority-class baseline (`DummyClassifier`):** A dummy classifier predicting the most frequent class is trained alongside MultinomialNB as an honest reference point.
- **Macro F1 prioritization:** Because sentiment classes can be imbalanced, Macro F1 is treated as the primary metric, supported by per-class Precision, Recall, and Confusion Matrices.

By default, reports are saved to:
- `reports/sentiment_distribution.csv` - label counts and proportions by game.
- `reports/model_metrics.json` - accuracy, macro/weighted F1, per-class reports, baseline comparisons, and confusion matrices.

To reproduce a committed snapshot, point `--output-dir` at the corresponding subfolder:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --output-dir data/processed/reddit_comments_labeled_analysis
```

## Recorded runs

Three historical model snapshots are committed under `data/processed/`. Their reported scores were produced before the current group-aware split safeguards, so they are retained as historical artifacts and must not be cited as results from the current implementation. The repository does not contain the source CSVs needed to regenerate them; rerun against the authorized final labeled datasets before reporting updated scores. The current code uses TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes, with `random_state=42` and a target test fraction of 0.2 by default. `auto` uses post and identical-text groups when at least five valid post IDs exist; otherwise, it uses identical-text groups with `StratifiedGroupKFold`. For group splits, the achieved test fraction is approximate and can differ substantially from the target, especially for high requested fractions or large clusters. Each report records the actual fraction. Use `stratified` for a row-level split that targets the requested fraction; that strategy does not prevent post-level or duplicate-text leakage.

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

Execute the automated test suite with Python's built-in `unittest`:

```powershell
python -m unittest discover -s tests -v
```

Or using `pytest`:

```powershell
pytest tests/ -v
```

## Research limitations

### Data coverage and reproducibility

- The study covers Reddit comments associated with `r/GlobalOffensive` and `r/VALORANT`, not all players, platforms, or gaming communities. Findings should not be generalized beyond this sampling frame without additional data.
- The committed workspace does not contain the raw scrape, final labeled CSVs, or a sampling log. The collection period, post/comment selection procedure, and any sampling exclusions therefore cannot be independently checked from this repository. Preserve these details with the final research materials.
- The three committed report snapshots are historical artifacts produced before the current group-aware split safeguards. Their JSON does not record current split metadata, so their scores are not directly comparable to a rerun with the current pipeline. Do not present them as current evaluation results.

### Labels and agreement

- VADER is a lexicon-based source of suggestions. Its suggestions can miss gaming slang, sarcasm, context, negation, and mixed sentiment; a model evaluated against `vader_auto` labels measures agreement with VADER, not validated human accuracy.
- The documented 600-comment annotation sample has mean Cohen's Kappa of 0.4098 and Fleiss' Kappa of 0.4067. These values indicate meaningful disagreement between annotators, so labels and downstream scores contain annotation uncertainty. Landis–Koch categories are descriptive heuristics, not universal quality thresholds.
- Pairwise Cohen scores use the rows available to each annotator pair, while Fleiss' Kappa uses only rows complete for every selected annotator. When annotations are missing, the metrics may describe different subsets. Kappa values can also change with category prevalence and annotator label distributions.
- The repository documents a 300-comment-per-game manual subset, but it does not preserve the annotator instructions, recruitment details, adjudication procedure, or sampling method. Those are needed to assess how representative and repeatable the annotations are.

### Model and evaluation

- TF-IDF with unigram/bigram features and Multinomial Naive Bayes is a baseline. It represents words and short phrases without deeper context, and does not use post titles, thread context, authors, or comment history. Sarcasm, slang, code-switching, and sentiment targets that are not explicit in the comment can be difficult to classify.
- The recorded results show weak and uneven class performance. In the historical snapshots, some classes have zero F1 (for example, negative in the full VALORANT VADER run, and negative/neutral in the manual VALORANT subset). Accuracy alone can hide this; report per-class precision/recall, Macro F1, support, and confusion matrices.
- Each analysis uses one seeded holdout split and the first fold returned by the splitter. It does not estimate variation across repeated splits or provide confidence intervals. Group-aware `test_size` is a target: cluster sizes and fold assignment can make the achieved test fraction differ, sometimes substantially. The report records the achieved fraction.
- Group splitting keeps identical text and, when usable `post_id` values exist, posts on one side of the split. It does not group by author, detect paraphrases or near-duplicate text, or model temporal dependence. The explicit `stratified` option is row-level and can place comments from the same post or duplicate texts on both sides; use it only when that evaluation design is intended.
- The current reports do not include significance tests for CS2-versus-VALORANT score differences. Differences may reflect label distributions, sampling, or annotation quality as well as sentiment patterns; they do not establish that one game's community is more positive or negative.
- Comment-level sentiment is observational and does not establish a causal relationship between a game and community opinion.

### Ethics and privacy

- Follow Reddit's terms, subreddit rules, and applicable ethics requirements. Minimize retained personal information, avoid publishing usernames, and document the handling of deleted or removed content.
