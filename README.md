# Sentiment Analysis of CS2 and VALORANT Communities on Reddit

Research workspace for **"A Comparative Sentiment Analysis of Counter-Strike 2 and VALORANT Player Communities on Reddit Using the Naive Bayes Algorithm."**

This project uses Reddit comments as the unit of analysis. VADER generates **initial label suggestions** for review, while TF-IDF and Multinomial Naive Bayes are used as a three-class classification baseline: `positive`, `neutral`, and `negative`. Inter-annotator agreement is evaluated using **Cohen's Kappa** (pairwise) and **Fleiss' Kappa** (multi-rater).

## Workspace structure

```text
data/
  raw/                                        # ignored — raw scrape files (not included)
  processed/
    reddit_comments_label_review.csv          # gitignored — VADER suggestions pending review
    reddit_comments_labeled.csv               # gitignored — VADER auto-labeled dataset (full)
    manual_label.csv                          # ignored — primary Model B dataset, 600 manual labels
    reddit_comments_labeled_analysis/         # committed historical snapshot: 5,000/game VADER-labeled run
    vader_subset_600_for_repo/                # committed historical snapshot: 300/game VADER-labeled subset
    manual_labeled_for_repo/                  # committed historical snapshot: 300/game manual subset
reports/                                      # generated output; two report files are currently tracked
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

Raw data in `data/raw/` and full comment CSVs are ignored to protect privacy. Report snapshots under `data/processed/*/` are committed for inspection, but the source CSVs are absent, so those experiments cannot currently be reproduced from this repository alone. The top-level `reports/` directory is listed in `.gitignore`, but `reports/model_metrics.json` and `reports/sentiment_distribution.csv` are already tracked; ignore rules do not untrack existing files, and running the analysis may modify them. Use a new `--output-dir` to preserve the tracked reports. Do not commit datasets containing personal usernames or data you are not authorized to publish.

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

The preparation script trims surrounding whitespace and uses VADER's compound score thresholds: scores at or above `0.05` suggest `positive`, scores at or below `-0.05` suggest `negative`, and intermediate scores suggest `neutral`. These suggestions are written to `sentiment_suggestion`; the `sentiment` field is left blank and `review_status` is `needs_review` until a person completes the review or the dataset is explicitly retained as `vader_auto`.

### Dataset roles and relationship

The repository describes three distinct analysis datasets. They are separate experiments and their scores must not be pooled or treated as though all comments received manual labels:

- **Primary study result — Model B, manually labeled subset:** `manual_label.csv` is a separate file used for inter-annotator agreement. The sample contains 600 comments total (300 per game), taken from the first 300 comments in each game's 5,000-comment set. The 600 comments were mixed in the annotation sheet and independently labeled by three annotators; majority vote supplies `sentiment_final`. `manual_labeled_for_repo/` contains the corresponding historical model report. Use this subset for the primary model evaluation against human judgments.
- **Full VADER-labeled dataset:** `reddit_comments_labeled.csv` is the full analysis input after the VADER review workflow. The historical report snapshot contains 5,000 comments per game (10,000 total) labeled `vader_auto`. This is a large-scale auxiliary automated-label experiment, not a 10,000-comment manual annotation.
- **VADER-labeled subset:** `vader_subset_600_for_repo/` contains a historical report for 300 VADER-labeled comments per game (600 total). It is a smaller automated-label experiment and is distinct from the manual subset.

The raw input JSON, final labeled CSVs, and annotation source file are not included in the repository. The manual subset consists of the first 300 comments in each game's 5,000-comment set, mixed together in the annotation sheet. The ordering basis for selecting those first 300 comments is not recorded.

### Collection details

The documented source communities are `r/GlobalOffensive` and `r/VALORANT`. The repository's data notes say `[deleted]` and `[removed]` comments and duplicate `comment_id` values were filtered during collection; the label-preparation script also skips empty comment text. Some collection details below are available from the researcher, while other details remain absent and should be filled from the collection log rather than inferred from report counts:

| Collection item | Study details currently available |
|---|---|
| Collection period and time zone | Scraping reportedly took place over one day on October 18, 2026, around 9 p.m. WIB (UTC+7); exact start/end times are not recorded |
| Post discovery and selection method | Scraping was used; the scraper/query and post-selection procedure are not recorded |
| Comment inclusion/exclusion criteria beyond the filters above | Not documented |
| Per-game collection target | 5,000 comments per game was a hard target/cap, not a census of all eligible comments |
| How the 300-per-game manual subsets were selected | The first 300 comments from each game's 5,000-comment set; the selected comments were mixed in the annotation sheet |

The first-300 selection is not documented as random sampling, so selection bias is possible. The collection counts are capped targets, not estimates of the total number of eligible Reddit comments.

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

### VADER vs. manual consensus agreement rate

The agreement-rate result is **not currently available** in this repository. Calculating it requires paired `sentiment_suggestion` and `sentiment_final` values for the same comments. The annotation CSV, VADER review CSV, and source spreadsheet are not included here; committed report snapshots contain only aggregate model metrics and class distributions, which are insufficient to recover the number of matches. Do not infer or report a percentage from those snapshots. When the paired source data is available, run the agreement command above and record its `Matched` count, valid sample count, and agreement percentage here.

### Annotator agreement on 600 sample comments:

| Metric | Overall | CS2 (N=300) | VALORANT (N=300) |
|---|---|---|---|
| **Mean Cohen's Kappa** | 0.4098 (Moderate) | 0.3715 (Fair) | 0.4431 (Moderate) |
| **Fleiss' Kappa** | 0.4067 (Moderate) | 0.3693 (Fair) | 0.4381 (Moderate) |

Pairwise Cohen's Kappa breakdown:
- Annotator 1 vs. Annotator 2: **0.5483** (Moderate) [N=600]
- Annotator 1 vs. Annotator 3: **0.3119** (Fair) [N=600]
- Annotator 2 vs. Annotator 3: **0.3692** (Fair) [N=600]

### Manual annotation protocol

Three annotators independently labeled the 600-comment sample (300 comments per game). Their labels were combined by majority vote in the `Hasil_Gabungan` spreadsheet: when all three labels matched, that label was used; when two of three matched, the majority label became `sentiment_final`. This procedure produces a majority label for every item with three valid ratings; no separate tie-breaking rule is documented or needed for three raters.

The annotation guidance defines the classes as follows:

- **Positive:** Appreciation, satisfaction, excitement, praise, or support toward a game, player, team, community, or update. Examples include praising gameplay, expressing happiness about a match result, or humor that clearly conveys enjoyment.
- **Neutral:** Informational or factual comments, questions, or comments without a clear positive or negative emotional opinion. This includes technical questions, factual statements, ambiguous sarcasm without a clear direction, and comments with balanced pros and cons.
- **Negative:** Disappointment, criticism, anger, complaints, or dissatisfaction toward a game, player, team, community, or update. This includes complaints about bugs, matchmaking, balance, or other players' behavior.

The exact collection date and time zone, scraper/query configuration, sample selection procedure, and comment-level inclusion criteria are not retained in the repository. The reported one-day collection duration does not replace a calendar date or a reproducible sampling description.

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

### Text preprocessing and TF-IDF configuration

The analysis loader trims leading and trailing whitespace from comment text. `TfidfVectorizer` then uses `lowercase=True`, `strip_accents="unicode"`, and `ngram_range=(1, 2)`. Other settings use scikit-learn defaults: word analysis with the default token pattern, `stop_words=None`, `min_df=1`, `max_df=1.0`, L2 normalization, and smooth inverse-document-frequency weighting. No stemming, lemmatization, or explicit URL/mention removal is performed. Punctuation is handled by the vectorizer's default tokenization rather than by a separate punctuation-cleaning step. The classifier is `MultinomialNB` with its default `alpha=1.0` and prior fitting enabled.

By default, reports are saved to:
- `reports/sentiment_distribution.csv` - label counts and proportions by game.
- `reports/model_metrics.json` - accuracy, macro/weighted F1, per-class reports, baseline comparisons, and confusion matrices.

To generate a report from an available input file, choose a new output directory so that existing tracked reports and historical snapshots are preserved:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --output-dir reports/new_run
```

## Recorded runs

Three historical model snapshots are committed under `data/processed/`. Their reported scores were produced before the current group-aware split safeguards, so they are retained as historical artifacts and must not be cited as results from the current implementation. The repository does not contain the source CSVs needed to regenerate them; rerun against the authorized final labeled datasets before reporting updated scores. The current code uses TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes, with `random_state=42` and a target test fraction of 0.2 by default. `auto` uses post and identical-text groups when at least five valid post IDs exist; otherwise, it uses identical-text groups with `StratifiedGroupKFold`. For group splits, the achieved test fraction is approximate and can differ substantially from the target, especially for high requested fractions or large clusters. Each report records the actual fraction. Use `stratified` for a row-level split that targets the requested fraction; that strategy does not prevent post-level or duplicate-text leakage.

### 1. Full dataset (auxiliary VADER-label experiment) — `reddit_comments_labeled_analysis/`

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

### 2. VADER-labeled subset (auxiliary experiment) — `vader_subset_600_for_repo/`

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

### 3. Manually labeled subset (primary Model B evaluation) — `manual_labeled_for_repo/`

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

In the historical manual-subset report, CS2 has higher accuracy and Macro F1 than VALORANT (0.467/0.283 versus 0.400/0.193), but the model has zero F1 for the negative class on CS2 and for the negative and neutral classes on VALORANT. The VADER-labeled full run also has zero negative-class F1 for VALORANT. This pattern suggests the baseline favors the positive class and has difficulty with some less well-served classes. The CS2–VALORANT score difference is descriptive only: the snapshots lack current split metadata, the datasets have different label distributions and label provenance, and this evaluation does not include uncertainty estimates or significance tests. It is not evidence by itself that one community is more positive or that the model generalizes better to that game's players.

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
- The label guidance and three-rater majority-vote rule are documented above. The subset uses the first 300 comments from each game's set rather than a documented random sample, so selection bias is possible. Annotator recruitment details and the source ordering basis are not recorded, which limits assessment of representativeness.

### Model and evaluation

- TF-IDF with unigram/bigram features and Multinomial Naive Bayes is a baseline. It represents words and short phrases without deeper context, and does not use post titles, thread context, authors, or comment history. Sarcasm, slang, code-switching, and sentiment targets that are not explicit in the comment can be difficult to classify.
- The recorded results show weak and uneven class performance. In the historical snapshots, some classes have zero F1 (for example, negative in the full VALORANT VADER run, and negative/neutral in the manual VALORANT subset). Accuracy alone can hide this; report per-class precision/recall, Macro F1, support, and confusion matrices.
- Each analysis uses one seeded holdout split and the first fold returned by the splitter. It does not estimate variation across repeated splits or provide confidence intervals. Group-aware `test_size` is a target: cluster sizes and fold assignment can make the achieved test fraction differ, sometimes substantially. The report records the achieved fraction.
- Group splitting keeps identical text and, when usable `post_id` values exist, posts on one side of the split. It does not group by author, detect paraphrases or near-duplicate text, or model temporal dependence. The explicit `stratified` option is row-level and can place comments from the same post or duplicate texts on both sides; use it only when that evaluation design is intended.
- The current reports do not include significance tests for CS2-versus-VALORANT score differences. Differences may reflect label distributions, sampling, or annotation quality as well as sentiment patterns; they do not establish that one game's community is more positive or negative.
- Comment-level sentiment is observational and does not establish a causal relationship between a game and community opinion.

### Ethics and privacy

- Follow Reddit's terms, subreddit rules, and applicable ethics requirements. Minimize retained personal information, avoid publishing usernames, and document the handling of deleted or removed content.
