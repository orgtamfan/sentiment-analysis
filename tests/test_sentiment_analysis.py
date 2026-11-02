"""Unit tests for sentiment_analysis module."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sentiment_analysis import (
    _build_leak_free_groups,
    _split_game_indices,
    analyze_dataset,
    load_dataset,
)


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

    def test_group_split_prevents_text_and_post_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "leak_test.csv"
            output_dir = root / "reports"

            records = []
            for game in ("cs2", "valorant"):
                for i in range(8):
                    post_id = f"thread_{game}_{i}"
                    # Identical text in post 0 and post 1 with SAME sentiment ("positive")
                    t = "great clutch" if i in (0, 1) else f"comment text {i}"
                    s = "positive" if i in (0, 1) else ("negative" if i % 2 == 1 else "neutral")
                    records.append({
                        "game": game,
                        "text": t,
                        "sentiment": s,
                        "comment_id": f"c_{game}_{i}_a",
                        "post_id": post_id,
                    })
                    records.append({
                        "game": game,
                        "text": f"distinct comment {i}",
                        "sentiment": "neutral" if i % 2 == 0 else "positive",
                        "comment_id": f"c_{game}_{i}_b",
                        "post_id": post_id,
                    })

            df = pd.DataFrame(records)
            df.to_csv(input_path, index=False)

            metrics = analyze_dataset(input_path, output_dir, split_strategy="group")
            # Inspect returned row indices directly to prove both leak barriers.
            for game in ("cs2", "valorant"):
                game_rows = df.loc[df["game"] == game].reset_index(drop=True)
                train_idx, test_idx, info = _split_game_indices(
                    game_rows, "group", 0.2, 42, game
                )
                train_rows = game_rows.iloc[train_idx]
                test_rows = game_rows.iloc[test_idx]
                self.assertFalse(set(train_rows["post_id"]) & set(test_rows["post_id"]))
                self.assertFalse(set(train_rows["text"]) & set(test_rows["text"]))
                self.assertIn("n_clusters", info)

    def test_fallback_reason_recorded_when_post_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "fallback_test.csv"
            output_dir = root / "reports"

            df = self._dataset().drop(columns=["post_id"])
            df.to_csv(input_path, index=False)

            metrics = analyze_dataset(input_path, output_dir, split_strategy="auto")
            for game in ("cs2", "valorant"):
                info = metrics["games"][game]["split_info"]
                self.assertEqual(info["strategy"], "StratifiedGroupKFold")
                self.assertEqual(info["fallback_reason"], "missing 'post_id' column")
                self.assertEqual(sum(info["train_class_counts"].values()), metrics["games"][game]["n_train"])
                self.assertEqual(sum(info["test_class_counts"].values()), metrics["games"][game]["n_test"])
                self.assertIn("n_clusters", info)
                self.assertEqual(info["n_text_groups"], info["n_clusters"])
                self.assertAlmostEqual(
                    info["actual_test_fraction"],
                    metrics["games"][game]["n_test"] / metrics["games"][game]["n_samples"],
                )
            self.assertEqual(metrics["random_state"], 42)
            self.assertEqual(metrics["model_config"]["version"], "tfidf_nb_v1")

    def test_direct_call_rejects_unknown_split_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "reddit_posts.csv"
            self._dataset().to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "split_strategy must be"):
                analyze_dataset(input_path, root / "reports", split_strategy="random")

    def test_auto_fallback_groups_identical_text_without_post_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "duplicate_text.csv"
            dataset = self._dataset().drop(columns=["post_id"])
            for game in ("cs2", "valorant"):
                mask = (dataset["game"] == game) & (dataset["sentiment"] == "positive")
                indexes = dataset.index[mask][:2]
                dataset.loc[indexes[1], "text"] = dataset.loc[indexes[0], "text"]
            dataset.to_csv(input_path, index=False)

            result = analyze_dataset(input_path, root / "reports", split_strategy="auto")

            for game in ("cs2", "valorant"):
                info = result["games"][game]["split_info"]
                self.assertEqual(info["strategy"], "StratifiedGroupKFold")
                self.assertEqual(info["grouping"], "identical_text_only")
                game_rows = dataset.loc[dataset["game"] == game].reset_index(drop=True)
                train_idx, test_idx, _ = _split_game_indices(
                    game_rows, "auto", 0.2, 42, game
                )
                self.assertFalse(
                    set(game_rows.iloc[train_idx]["text"])
                    & set(game_rows.iloc[test_idx]["text"])
                )

    def test_stratified_strategy_uses_row_level_split(self) -> None:
        game_rows = self._dataset().query("game == 'cs2'").reset_index(drop=True)
        train_idx, test_idx, info = _split_game_indices(
            game_rows, "stratified", 0.25, 42, "cs2"
        )
        self.assertEqual(info["strategy"], "StratifiedShuffleSplit")
        self.assertEqual(len(train_idx) + len(test_idx), len(game_rows))

    def test_cluster_ids_are_deterministic(self) -> None:
        rows = pd.DataFrame({
            "post_id": ["p2", "p1", "p3", "p4"],
            "text": ["same", "same", "other", "last"],
        })
        first = _build_leak_free_groups(rows)
        second = _build_leak_free_groups(rows)
        pd.testing.assert_series_equal(first, second)


if __name__ == "__main__":
    unittest.main()
