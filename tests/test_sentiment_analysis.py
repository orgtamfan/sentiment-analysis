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
        rows = [
            {"game": game, "text": text, "sentiment": sentiment}
            for game in ("cs2", "valorant")
            for sentiment, texts in examples.items()
            for text in texts
        ]
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

    def test_load_dataset_rejects_unknown_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "reddit_posts.csv"
            self._dataset().to_csv(input_path, index=False)
            dataset = pd.read_csv(input_path)
            dataset.loc[0, "sentiment"] = "mixed"
            dataset.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "Unsupported sentiment labels"):
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


if __name__ == "__main__":
    unittest.main()
