"""TF-IDF and Multinomial Naive Bayes baseline for labeled Reddit data.

Includes support for:
- Standard stratified train/test split or group-aware StratifiedGroupKFold split by `post_id`
- Majority-class baseline (DummyClassifier) alongside MultinomialNB
- Strict validation of `review_status`
- Comprehensive evaluation focusing on Macro F1, weighted F1, per-class metrics, and confusion matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

GAMES = ("cs2", "valorant")
SENTIMENTS = ("negative", "neutral", "positive")
REQUIRED_COLUMNS = {"game", "text", "sentiment"}
VALID_REVIEW_STATUSES = {"reviewed", "vader_auto"}


def _build_leak_free_groups(
    df: pd.DataFrame, group_col: str = "post_id", text_col: str = "text"
) -> pd.Series:
    """Cluster items so that no post_id and no identical text is split across clusters.

    Uses disjoint-set (union-find) to merge posts that share identical comment text.
    """
    parent: dict[Any, Any] = {}

    def find(i: Any) -> Any:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: Any, j: Any) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    posts = df[group_col].unique()
    texts = df[text_col].unique()
    for p in posts:
        parent[("p", p)] = ("p", p)
    for t in texts:
        parent[("t", t)] = ("t", t)

    for p, t in zip(df[group_col], df[text_col]):
        union(("p", p), ("t", t))

    group_ids = {p: find(("p", p)) for p in posts}
    cluster_map = {root: idx for idx, root in enumerate(set(group_ids.values()))}
    return df[group_col].map(lambda p: cluster_map[group_ids[p]])


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Read and validate the annotated CSV dataset."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    dataset = pd.read_csv(input_path)
    if "sentiment" not in dataset.columns and "sentiment_final" in dataset.columns:
        dataset = dataset.rename(columns={"sentiment_final": "sentiment"})

    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataset = dataset.copy()
    dataset["game"] = dataset["game"].astype("string").str.strip().str.lower()
    dataset["text"] = dataset["text"].astype("string").str.strip()
    dataset["sentiment"] = (
        dataset["sentiment"].astype("string").str.strip().str.lower()
    )

    if dataset[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns cannot contain empty values.")
    if dataset["text"].eq("").any():
        raise ValueError("Text entries cannot be empty.")

    # Integrity checks: check duplicate comment_id and report/validate exact text duplicates
    if "comment_id" in dataset.columns:
        duplicate_ids = dataset["comment_id"].duplicated()
        if duplicate_ids.any():
            raise ValueError(
                f"Found {duplicate_ids.sum()} duplicate comment IDs in dataset."
            )

    # Check for text duplicates per game
    duplicate_texts = dataset.duplicated(subset=["game", "text"])
    if duplicate_texts.any():
        n_dupe_texts = int(duplicate_texts.sum())
        # Check if conflicting sentiment labels exist for identical texts
        text_sentiment_counts = (
            dataset.groupby(["game", "text"])["sentiment"]
            .nunique()
        )
        conflicts = text_sentiment_counts[text_sentiment_counts > 1]
        if not conflicts.empty:
            raise ValueError(
                f"Found {len(conflicts)} duplicate text(s) with conflicting sentiment labels."
            )

    # Strict validation of review_status: must be explicitly valid, not just non-'needs_review'
    if "review_status" in dataset.columns:
        status_series = (
            dataset["review_status"].astype("string").str.strip().str.lower()
        )
        if status_series.eq("needs_review").any():
            raise ValueError(
                "Dataset contains comments marked needs_review. "
                "Review the VADER suggestions and update review_status before analysis."
            )
        invalid_mask = status_series.isna() | ~status_series.isin(
            VALID_REVIEW_STATUSES
        )
        if invalid_mask.any():
            invalid_vals = sorted(
                set(dataset.loc[invalid_mask, "review_status"].dropna().astype(str))
            )
            has_nan = dataset.loc[invalid_mask, "review_status"].isna().any()
            err_parts = []
            if invalid_vals:
                err_parts.append(f"unrecognized status(es): {invalid_vals}")
            if has_nan:
                err_parts.append("missing/empty status values")
            raise ValueError(
                f"Dataset contains invalid review_status ({', '.join(err_parts)}). "
                f"Allowed statuses: {sorted(VALID_REVIEW_STATUSES)}."
            )

    unknown_games = sorted(set(dataset["game"].dropna()) - set(GAMES))
    if unknown_games:
        raise ValueError(
            "Unsupported game values: "
            + ", ".join(unknown_games)
            + ". Use 'cs2' or 'valorant'."
        )

    unknown_sentiments = sorted(set(dataset["sentiment"].dropna()) - set(SENTIMENTS))
    if unknown_sentiments:
        raise ValueError(
            "Unsupported sentiment labels: "
            + ", ".join(unknown_sentiments)
            + ". Use 'positive', 'neutral', or 'negative'."
        )

    absent_games = sorted(set(GAMES) - set(dataset["game"].dropna()))
    if absent_games:
        raise ValueError(
            "Dataset must contain both games; missing: " + ", ".join(absent_games)
        )

    return dataset


def _build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )


def _evaluate_predictions(
    test_labels: Sequence[str], predictions: Sequence[str]
) -> dict[str, Any]:
    """Calculate standard evaluation metrics with Macro F1 prioritized."""
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(
            f1_score(
                test_labels,
                predictions,
                labels=list(SENTIMENTS),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                test_labels,
                predictions,
                labels=list(SENTIMENTS),
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=list(SENTIMENTS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=list(SENTIMENTS),
        ).tolist(),
    }


def analyze_dataset(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "auto",
) -> dict[str, Any]:
    """Evaluate one model per game and write label distribution and metrics.

    Args:
        input_path: CSV dataset path.
        output_dir: Destination folder for output reports.
        test_size: Approximate test fraction (default: 0.2).
        random_state: Seed for reproducibility.
        split_strategy: 'auto', 'group', or 'stratified'. 'auto' uses group splitting
            if 'post_id' exists with >= 5 groups, else stratified split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if split_strategy not in {"auto", "group", "stratified"}:
        raise ValueError("split_strategy must be 'auto', 'group', or 'stratified'.")

    dataset = load_dataset(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        dataset.groupby(["game", "sentiment"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    game_totals = counts.groupby("game")["count"].transform("sum")
    counts["proportion"] = counts["count"] / game_totals

    metrics: dict[str, Any] = {
        "model": "TfidfVectorizer(1, 2) + MultinomialNB",
        "baseline_model": "DummyClassifier(strategy='most_frequent')",
        "test_size": test_size,
        "random_state": random_state,
        "sentiment_labels": list(SENTIMENTS),
        "games": {},
    }

    for game in GAMES:
        game_data = dataset.loc[dataset["game"] == game].copy()
        class_counts = game_data["sentiment"].value_counts().reindex(
            SENTIMENTS, fill_value=0
        )
        too_small = sorted(
            label for label, count in class_counts.items() if count < 2
        )
        if too_small:
            raise ValueError(
                f"Game '{game}' needs at least two examples per sentiment class; "
                "too few: "
                + ", ".join(too_small)
            )

        # Decide split strategy
        use_group = False
        fallback_reason = None
        if split_strategy in ("auto", "group"):
            if "post_id" not in game_data.columns:
                fallback_reason = "missing 'post_id' column"
            elif game_data["post_id"].isna().any():
                fallback_reason = "'post_id' contains missing values"
            elif game_data["post_id"].nunique() < 5:
                fallback_reason = f"only {game_data['post_id'].nunique()} distinct post_ids (minimum 5 required)"
            else:
                use_group = True

            if split_strategy == "group" and not use_group:
                raise ValueError(
                    f"Requested 'group' split for '{game}', but cannot perform group splitting: {fallback_reason}."
                )

        if use_group:
            # Build leak-free super-groups combining post_id and shared text
            clusters = _build_leak_free_groups(game_data, group_col="post_id", text_col="text")
            n_clusters = clusters.nunique()
            min_class_count = int(class_counts.min())
            n_splits = int(round(1.0 / test_size)) if test_size > 0 else 5
            n_splits = min(n_splits, n_clusters, min_class_count)
            n_splits = max(2, n_splits)
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(
                sgkf.split(
                    game_data["text"],
                    game_data["sentiment"],
                    groups=clusters,
                )
            )
            train_text = game_data.iloc[train_idx]["text"]
            train_labels = game_data.iloc[train_idx]["sentiment"]
            test_text = game_data.iloc[test_idx]["text"]
            test_labels = game_data.iloc[test_idx]["sentiment"]

            # Validate split integrity: post_id and text disjointness
            post_train = set(game_data.iloc[train_idx]["post_id"])
            post_test = set(game_data.iloc[test_idx]["post_id"])
            post_intersection = post_train.intersection(post_test)
            if post_intersection:
                raise ValueError(
                    f"Group split leak detected for '{game}': {len(post_intersection)} post_ids appear in both train and test."
                )

            text_train = set(train_text)
            text_test = set(test_text)
            text_intersection = text_train.intersection(text_test)
            if text_intersection:
                raise ValueError(
                    f"Text duplicate leak detected for '{game}': {len(text_intersection)} identical text(s) appear in both train and test."
                )

            split_info = {
                "strategy": "StratifiedGroupKFold",
                "n_groups_total": int(game_data["post_id"].nunique()),
                "n_clusters": int(n_clusters),
                "n_groups_train": int(len(post_train)),
                "n_groups_test": int(len(post_test)),
            }
        else:
            duplicate_texts_exist = game_data["text"].duplicated(keep=False).any()
            if duplicate_texts_exist:
                # Even without post IDs, keep identical texts on one side of the split.
                rows = game_data.assign(_row_group=range(len(game_data)))
                text_groups = _build_leak_free_groups(
                    rows, group_col="_row_group", text_col="text"
                )
                n_splits = min(
                    max(2, int(round(1.0 / test_size))),
                    int(text_groups.nunique()),
                    int(class_counts.min()),
                )
                if n_splits < 2:
                    raise ValueError(
                        f"Could not create a text-group split for '{game}'. "
                        "Add more distinct examples or adjust test_size."
                    )
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
                train_idx, test_idx = next(
                    splitter.split(
                        game_data["text"], game_data["sentiment"], groups=text_groups
                    )
                )
                train_text = game_data.iloc[train_idx]["text"]
                train_labels = game_data.iloc[train_idx]["sentiment"]
                test_text = game_data.iloc[test_idx]["text"]
                test_labels = game_data.iloc[test_idx]["sentiment"]
                split_info = {
                    "strategy": "StratifiedGroupKFold",
                    "grouping": "identical_text_only",
                    "n_text_groups": int(text_groups.nunique()),
                }
            else:
                try:
                    train_text, test_text, train_labels, test_labels = train_test_split(
                        game_data["text"],
                        game_data["sentiment"],
                        test_size=test_size,
                        random_state=random_state,
                        stratify=game_data["sentiment"],
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Could not create a stratified split for '{game}'. "
                        "Add more labeled examples or adjust test_size."
                    ) from error
                split_info = {"strategy": "StratifiedShuffleSplit"}
            if fallback_reason:
                split_info["fallback_reason"] = fallback_reason

        # Post-split class distribution validation
        missing_train = set(SENTIMENTS) - set(train_labels.unique())
        missing_test = set(SENTIMENTS) - set(test_labels.unique())
        if missing_train or missing_test:
            raise ValueError(
                f"Invalid split for '{game}'. "
                f"Missing train classes: {sorted(missing_train) if missing_train else 'none'}; "
                f"missing test classes: {sorted(missing_test) if missing_test else 'none'}."
            )

        # Train MultinomialNB model
        model = _build_model()
        model.fit(train_text, train_labels)
        predictions = model.predict(test_text)
        model_eval = _evaluate_predictions(test_labels, predictions)

        # Train majority-class baseline (DummyClassifier) for honest comparison
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(train_text, train_labels)
        dummy_predictions = dummy.predict(test_text)
        dummy_eval = _evaluate_predictions(test_labels, dummy_predictions)
        majority_class = train_labels.value_counts().idxmax()

        metrics["games"][game] = {
            "n_samples": int(len(game_data)),
            "n_train": int(len(train_text)),
            "n_test": int(len(test_text)),
            "split_info": split_info,
            "majority_class": str(majority_class),
            "naive_bayes": model_eval,
            "majority_baseline": dummy_eval,
            # Backward compatibility fields at root of game entry
            "accuracy": model_eval["accuracy"],
            "macro_f1": model_eval["macro_f1"],
            "weighted_f1": model_eval["weighted_f1"],
            "classification_report": model_eval["classification_report"],
            "confusion_matrix": model_eval["confusion_matrix"],
        }

    counts.to_csv(output_dir / "sentiment_distribution.csv", index=False)
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as result_file:
        json.dump(metrics, result_file, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze labeled Reddit sentiment for CS2 and VALORANT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an annotated CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated reports (default: reports).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of each game's data used for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible train/test splits (default: 42).",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["auto", "group", "stratified"],
        default="auto",
        help="Data split strategy: 'auto' (group split if post_id exists, else stratified), "
        "'group' (enforce post_id StratifiedGroupKFold), or 'stratified' (standard stratified).",
    )
    args = parser.parse_args()

    metrics = analyze_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
    )

    print("=" * 65)
    print("SENTIMENT ANALYSIS EVALUATION RESULTS")
    print("=" * 65)
    for game, res in metrics["games"].items():
        print(f"\nGame: {game.upper()} (Split: {res['split_info']['strategy']}, N_test: {res['n_test']})")
        nb = res["naive_bayes"]
        base = res["majority_baseline"]
        print(f"  [Multinomial Naive Bayes]")
        print(f"    Macro F1    : {nb['macro_f1']:.4f}  (Baseline majority: {base['macro_f1']:.4f})")
        print(f"    Weighted F1 : {nb['weighted_f1']:.4f}  (Baseline majority: {base['weighted_f1']:.4f})")
        print(f"    Accuracy    : {nb['accuracy']:.4f}  (Baseline majority: {base['accuracy']:.4f})")
        print("    Per-class metrics:")
        for sentiment in SENTIMENTS:
            rep = nb["classification_report"].get(sentiment, {})
            print(
                f"      {sentiment:8s} -> Precision: {rep.get('precision', 0.0):.3f}, "
                f"Recall: {rep.get('recall', 0.0):.3f}, "
                f"F1: {rep.get('f1-score', 0.0):.3f} (support: {int(rep.get('support', 0))})"
            )
    print("=" * 65)
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
