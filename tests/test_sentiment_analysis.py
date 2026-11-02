"""Unit tests for sentiment_analysis module."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
