"""TF-IDF and Multinomial Naive Bayes baseline for labeled Reddit data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

GAMES = ("cs2", "valorant")
SENTIMENTS = ("negative", "neutral", "positive")
REQUIRED_COLUMNS = {"game", "text", "sentiment"}


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Read and validate the annotated CSV dataset."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    dataset = pd.read_csv(input_path)
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
    if "review_status" in dataset.columns:
        pending_review = dataset["review_status"].astype("string").str.strip().eq(
            "needs_review"
        )
        if pending_review.any():
            raise ValueError(
                "Dataset contains comments marked needs_review. "
                "Review the VADER suggestions and update review_status before analysis."
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


def analyze_dataset(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate one model per game and write label distribution and metrics."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

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
        "test_size": test_size,
        "random_state": random_state,
        "sentiment_labels": list(SENTIMENTS),
        "games": {},
    }

    for game in GAMES:
        game_data = dataset.loc[dataset["game"] == game]
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

        model = _build_model()
        model.fit(train_text, train_labels)
        predictions = model.predict(test_text)

        metrics["games"][game] = {
            "n_samples": int(len(game_data)),
            "n_train": int(len(train_text)),
            "n_test": int(len(test_text)),
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
    args = parser.parse_args()

    analyze_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Analysis complete. Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
