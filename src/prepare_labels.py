"""Generate VADER label suggestions for manual review of Reddit comments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_FIELDS = (
    "game",
    "text",
    "sentiment",
    "sentiment_suggestion",
    "vader_compound",
    "review_status",
    "comment_id",
    "post_id",
    "subreddit",
    "post_title",
    "comment_created_at",
)
GAME_NAMES = {"cs2": "cs2", "valorant": "valorant"}


def generate_label_suggestions(
    input_path: Path,
    output_path: Path,
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> dict[str, int]:
    """Write a reviewable CSV without treating automatic suggestions as labels."""
    if not -1 <= negative_threshold < positive_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy -1 <= negative < positive <= 1."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the raw input JSON.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Review CSV already exists: {output_path}. "
            "Choose another --output path or remove it after preserving any edits."
        )

    with input_path.open(encoding="utf-8") as source_file:
        records = json.load(source_file)
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of comment objects.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every item in the input JSON must be an object.")

    analyzer = SentimentIntensityAnalyzer()
    counts = {"written": 0, "skipped_empty_text": 0}
    output_rows: list[dict[str, Any]] = []

    for row_number, record in enumerate(records, start=1):
        raw_text = record.get("comment_text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            counts["skipped_empty_text"] += 1
            continue

        raw_game = record.get("game")
        if not isinstance(raw_game, str):
            raise ValueError(f"Record {row_number} has no valid game value.")
        game = GAME_NAMES.get(raw_game.strip().lower())
        if game is None:
            raise ValueError(
                f"Record {row_number} has unsupported game value: {raw_game!r}."
            )

        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= positive_threshold:
            suggestion = "positive"
        elif compound <= negative_threshold:
            suggestion = "negative"
        else:
            suggestion = "neutral"

        output_rows.append(
            {
                "game": game,
                "text": text,
                "sentiment": "",
                "sentiment_suggestion": suggestion,
                "vader_compound": compound,
                "review_status": "needs_review",
                "comment_id": record.get("comment_id", ""),
                "post_id": record.get("post_id", ""),
                "subreddit": record.get("subreddit", ""),
                "post_title": record.get("post_title", ""),
                "comment_created_at": record.get("comment_created_at", ""),
            }
        )
        counts["written"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create VADER sentiment suggestions for manual review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/reddit_comments_raw.json"),
        help="Raw Reddit comments JSON (default: data/raw/reddit_comments_raw.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reddit_comments_label_review.csv"),
        help=(
            "Review CSV output (default: "
            "data/processed/reddit_comments_label_review.csv)."
        ),
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.05,
        help="VADER compound score at or above this value suggests positive.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=-0.05,
        help="VADER compound score at or below this value suggests negative.",
    )
    args = parser.parse_args()

    counts = generate_label_suggestions(
        input_path=args.input,
        output_path=args.output,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    print(
        f"Label review CSV saved to {args.output}: "
        f"{counts['written']} comments, "
        f"{counts['skipped_empty_text']} empty comments skipped."
    )
    print(
        "Review each suggestion and fill the 'sentiment' column before "
        "training or evaluation."
    )


if __name__ == "__main__":
    main()
