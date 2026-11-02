"""Unit tests for sentiment_analysis module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.sentiment_analysis import analyze_dataset, load_dataset


class SentimentAnalysisTests(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        examples = {
            "negative": [
                "terrible match and frustrating loss",
                "awful game with a bad result",
                "horrible round and disappointing play",
                "poor match and very annoying loss",
            ],
            "neutral": [
                "the match starts at seven",
                "players joined the game today",
                "the update was released this week",
                "the next round begins soon",
            ],
            "positive": [
                "great match and excellent win",
                "amazing game with a good result",
                "wonderful round and impressive play",
                "fun match and very enjoyable win",
            ],
        }
        rows = []
        for game in ("cs2", "valorant"):
            for sentiment, texts in examples.items():
                for i, text in enumerate(texts):
                    # Assign distinct post_id so each game has 12 distinct posts (>= 5 needed for group split)
                    rows.append({
                        "game": game,
                        "text": text,
                        "sentiment": sentiment,
                        "comment_id": f"{game}_{sentiment}_{i}",
                        "post_id": f"post_{game}_{sentiment}_{i}",
                    })
        return pd.DataFrame(rows)

    def test_analysis_writes_reports_for_both_games(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            output_dir = root / "reports"
            self._dataset().to_csv(input_path, index=False)

            result = analyze_dataset(input_path, output_dir)

            self.assertEqual(set(result["games"]), {"cs2", "valorant"})
            self.assertTrue((output_dir / "sentiment_distribution.csv").is_file())
            self.assertTrue((output_dir / "model_metrics.json").is_file())
            self.assertEqual(result["games"]["cs2"]["n_samples"], 12)
            self.assertIn("majority_baseline", result["games"]["cs2"])

    def test_analysis_reports_majority_class_from_training_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            output_dir = root / "reports"
            dataset = self._dataset()
            extra_positive = dataset.loc[
                (dataset["game"] == "cs2") & (dataset["sentiment"] == "positive")
            ].iloc[:4].copy()
            extra_positive["text"] = [f"additional positive comment {i}" for i in range(4)]
            extra_positive["comment_id"] = [f"extra_positive_{i}" for i in range(4)]
            extra_positive["post_id"] = [f"extra_post_{i}" for i in range(4)]
            dataset = pd.concat([dataset, extra_positive], ignore_index=True)
            dataset.to_csv(input_path, index=False)

            result = analyze_dataset(
                input_path, output_dir, split_strategy="stratified"
            )

            self.assertEqual(result["games"]["cs2"]["majority_class"], "positive")

    def test_load_dataset_rejects_unknown_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            self._dataset().to_csv(input_path, index=False)
            dataset = pd.read_csv(input_path)
            dataset.loc[0, "sentiment"] = "mixed"
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Unsupported sentiment labels"):
                load_dataset(input_path)

    def test_load_dataset_rejects_empty_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.loc[0, "sentiment"] = ""
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Required columns cannot contain empty values"):
                load_dataset(input_path)

    def test_load_dataset_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.loc[0, "text"] = "   "
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Text entries cannot be empty"):
                load_dataset(input_path)

    def test_load_dataset_rejects_unknown_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.loc[0, "game"] = "dota2"
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Unsupported game values"):
                load_dataset(input_path)

    def test_load_dataset_rejects_comments_pending_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            dataset["review_status"] = "reviewed"
            dataset.loc[0, "review_status"] = "needs_review"
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "marked needs_review"):
                load_dataset(input_path)

    def test_load_dataset_rejects_duplicate_comment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.loc[1, "comment_id"] = dataset.loc[0, "comment_id"]
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate comment IDs"):
                load_dataset(input_path)

    def test_load_dataset_rejects_conflicting_sentiment_for_same_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            dataset = self._dataset()
            # Same game, identical text, but conflicting sentiment labels
            dataset.loc[1, "text"] = dataset.loc[0, "text"]
            dataset.loc[0, "sentiment"] = "positive"
            dataset.loc[1, "sentiment"] = "negative"
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "conflicting sentiment labels"):
                load_dataset(input_path)

    def test_analysis_rejects_missing_class_without_writing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            output_dir = root / "reports"
            dataset = self._dataset()
            dataset = dataset.loc[
                ~((dataset["game"] == "cs2") & (dataset["sentiment"] == "positive"))
            ]
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "needs at least two examples"):
                analyze_dataset(input_path, output_dir)

            self.assertFalse((output_dir / "sentiment_distribution.csv").exists())
            self.assertFalse((output_dir / "model_metrics.json").exists())

    def test_stratified_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            output_dir = root / "reports"
            self._dataset().to_csv(input_path, index=False)

            metrics = analyze_dataset(
                input_path, output_dir, split_strategy="group"
            )
            self.assertEqual(
                metrics["games"]["cs2"]["split_info"]["strategy"],
                "StratifiedGroupKFold",
            )

    def test_group_split_rejects_missing_classes_in_train_or_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.to_csv(input_path, index=False)

            with patch(
                "src.sentiment_analysis.StratifiedGroupKFold"
            ) as splitter:
                splitter.return_value.split.return_value = iter(
                    [(list(range(8)), list(range(8, 12)))]
                )
                with self.assertRaisesRegex(
                    ValueError, "Missing in train: .*; missing in test:"
                ):
                    analyze_dataset(
                        input_path, root / "reports", split_strategy="group"
                    )

    def test_auto_split_records_fallback_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            self._dataset().drop(columns="post_id").to_csv(input_path, index=False)

            result = analyze_dataset(input_path, root / "reports")

            self.assertEqual(
                result["games"]["cs2"]["split_info"]["fallback_reason"],
                "post_id column is missing",
            )

    def test_analysis_reports_duplicate_texts_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            dataset = self._dataset()
            dataset.loc[1, "text"] = dataset.loc[0, "text"]
            dataset.to_csv(input_path, index=False)

            def split_with_duplicate_crossing(text, labels, **kwargs):
                test_indices = [1, 4, 8] if labels.index.min() == 0 else [13, 16, 20]
                train_indices = labels.index.difference(test_indices)
                return (
                    text.loc[train_indices],
                    text.loc[test_indices],
                    labels.loc[train_indices],
                    labels.loc[test_indices],
                )

            with patch(
                "src.sentiment_analysis.train_test_split",
                side_effect=split_with_duplicate_crossing,
            ):
                result = analyze_dataset(
                    input_path, root / "reports", split_strategy="stratified"
                )

            split_info = result["games"]["cs2"]["split_info"]
            self.assertEqual(split_info["n_duplicate_texts_across_splits"], 1)
            self.assertEqual(
                split_info["n_rows_with_duplicate_text_across_splits"], 2
            )

    def test_analysis_rejects_invalid_split_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            self._dataset().to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "split_strategy must be"):
                analyze_dataset(
                    input_path, root / "reports", split_strategy="unknown"
                )


if __name__ == "__main__":
    unittest.main()
