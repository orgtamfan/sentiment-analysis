"""Additional unit tests for group splitting and pairwise missing data handling."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.inter_annotator_agreement import (
    analyze_agreement,
    compute_pairwise_cohen_kappas,
)
from src.sentiment_analysis import analyze_dataset, load_dataset


class TestEnhancements(unittest.TestCase):
    def test_strict_review_status_rejects_arbitrary_strings(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "test.csv"
            df = pd.DataFrame({
                "game": ["cs2", "cs2", "valorant", "valorant"],
                "text": ["good", "bad", "okay", "fine"],
                "sentiment": ["positive", "negative", "neutral", "positive"],
                "review_status": ["done", "reviewed", "reviewed", "reviewed"],
            })
            df.to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "unrecognized status"):
                load_dataset(csv_path)

    def test_strict_review_status_accepts_vader_auto_and_reviewed(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "test.csv"
            df = pd.DataFrame({
                "game": ["cs2", "cs2", "valorant", "valorant"],
                "text": ["good", "bad", "okay", "fine"],
                "sentiment": ["positive", "negative", "neutral", "positive"],
                "review_status": ["vader_auto", "reviewed", "vader_auto", "reviewed"],
            })
            df.to_csv(csv_path, index=False)
            loaded = load_dataset(csv_path)
            self.assertEqual(len(loaded), 4)

    def test_pairwise_cohen_preserves_partial_annotations(self):
        # Annotator 3 has missing value on row 0, but annotator 1 and 2 are filled.
        # Pair 1 vs 2 should include row 0 (n_samples = 3).
        df = pd.DataFrame({
            "a1": ["positive", "neutral", "negative"],
            "a2": ["positive", "neutral", "negative"],
            "a3": [None, "neutral", "negative"],
        })
        res = compute_pairwise_cohen_kappas(df, ["a1", "a2", "a3"])
        self.assertEqual(res["pairs"]["a1_vs_a2"]["n_samples"], 3)
        self.assertAlmostEqual(res["pairs"]["a1_vs_a2"]["kappa"], 1.0)
        self.assertEqual(res["pairs"]["a1_vs_a3"]["n_samples"], 2)

    def test_group_split_avoids_post_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "grouped.csv"
            out_dir = root / "reports"

            # 6 distinct posts for cs2, each with 2 comments
            # 6 distinct posts for valorant, each with 2 comments
            records = []
            for game in ["cs2", "valorant"]:
                for p in range(6):
                    post_id = f"post_{game}_{p}"
                    records.append({
                        "game": game,
                        "text": f"comment 1 on {post_id}",
                        "sentiment": "positive" if p % 2 == 0 else "negative",
                        "post_id": post_id,
                    })
                    records.append({
                        "game": game,
                        "text": f"comment 2 on {post_id}",
                        "sentiment": "neutral" if p % 3 == 0 else "positive",
                        "post_id": post_id,
                    })

            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)

            metrics = analyze_dataset(
                input_path=csv_path,
                output_dir=out_dir,
                test_size=0.33,
                random_state=42,
                split_strategy="group",
            )
            for game in ["cs2", "valorant"]:
                self.assertEqual(
                    metrics["games"][game]["split_info"]["strategy"],
                    "StratifiedGroupKFold",
                )
                self.assertIn("majority_baseline", metrics["games"][game])


if __name__ == "__main__":
    unittest.main()
