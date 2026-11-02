"""Unit tests for inter_annotator_agreement module."""

import unittest
import numpy as np
import pandas as pd

from src.inter_annotator_agreement import (
    analyze_agreement,
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_pairwise_cohen_kappas,
    compute_vader_vs_consensus_agreement,
    interpret_kappa,
    normalize_labels,
)


class TestInterAnnotatorAgreement(unittest.TestCase):
    def test_perfect_cohen_agreement(self):
        r1 = ["positive", "neutral", "negative"]
        r2 = ["positive", "neutral", "negative"]
        score = compute_cohens_kappa(r1, r2)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_perfect_fleiss_agreement(self):
        df = pd.DataFrame({
            "a": ["positive", "neutral", "negative"],
            "b": ["positive", "neutral", "negative"],
            "c": ["positive", "neutral", "negative"],
        })
        res = compute_fleiss_kappa(df, ["a", "b", "c"])
        self.assertAlmostEqual(res["kappa"], 1.0, places=4)

    def test_statsmodels_fleiss_calculation(self):
        df = pd.DataFrame({
            "annotator_1": ["positive", "neutral", "negative"],
            "annotator_2": ["positive", "negative", "negative"],
            "annotator_3": ["positive", "neutral", "negative"],
        })
        res = compute_fleiss_kappa(df, ["annotator_1", "annotator_2", "annotator_3"])
        self.assertIn("kappa", res)
        self.assertGreater(res["kappa"], 0.5)

    def test_normalization_strips_whitespace_and_handles_empty(self):
        df = pd.DataFrame({
            "r1": [" Positive  ", "  ", None, "NEGATIVE"],
            "r2": ["neutral", "positive", "   ", "negative"],
        })
        norm = normalize_labels(df, ["r1", "r2"])
        self.assertEqual(norm["r1"].iloc[0], "positive")
        self.assertTrue(pd.isna(norm["r1"].iloc[1]))
        self.assertTrue(pd.isna(norm["r1"].iloc[2]))
        self.assertEqual(norm["r1"].iloc[3], "negative")

    def test_normalization_rejects_invalid_categories(self):
        df = pd.DataFrame({
            "r1": ["positive", "sarcastic"],
        })
        with self.assertRaisesRegex(ValueError, "invalid sentiment label"):
            normalize_labels(df, ["r1"])

    def test_pairwise_cohen_with_missing_annotator_data(self):
        # 3 annotators; annotator 3 has missing value on row 0, but 1 & 2 are complete.
        df = pd.DataFrame({
            "a1": ["positive", "neutral", "negative"],
            "a2": ["positive", "neutral", "negative"],
            "a3": ["   ", "neutral", "negative"],  # empty string normalized to NA
        })
        res = compute_pairwise_cohen_kappas(df, ["a1", "a2", "a3"])
        # a1 vs a2 should have all 3 samples evaluated with kappa = 1.0
        self.assertEqual(res["pairs"]["a1_vs_a2"]["n_samples"], 3)
        self.assertAlmostEqual(res["pairs"]["a1_vs_a2"]["kappa"], 1.0)
        # a1 vs a3 should have only 2 valid samples
        self.assertEqual(res["pairs"]["a1_vs_a3"]["n_samples"], 2)

    def test_fleiss_kappa_when_no_complete_rows(self):
        # Every row has at least one missing annotator
        df = pd.DataFrame({
            "a1": ["positive", None],
            "a2": [None, "neutral"],
            "a3": ["positive", "negative"],
        })
        res = compute_fleiss_kappa(df, ["a1", "a2", "a3"])
        self.assertEqual(res["n_subjects"], 0)
        self.assertIsNone(res["kappa"])

    def test_vader_vs_consensus_agreement(self):
        df = pd.DataFrame({
            "sentiment_suggestion": ["positive", "neutral", "negative", "positive"],
            "sentiment_final": ["positive", "neutral", "positive", "positive"],
        })
        res = compute_vader_vs_consensus_agreement(
            df, vader_col="sentiment_suggestion", consensus_col="sentiment_final"
        )
        self.assertEqual(res["n_samples"], 4)
        self.assertEqual(res["n_matches"], 3)
        self.assertAlmostEqual(res["agreement_rate"], 0.75, places=4)
        self.assertEqual(res["agreement_percentage"], "75.00%")

    def test_vader_agreement_normalizes_and_ignores_whitespace_only(self):
        df = pd.DataFrame({
            "vader": [" Positive ", "   ", "sarcastic"],
            "consensus": ["positive", "negative", "sarcastic"],
        })
        with self.assertRaisesRegex(ValueError, "invalid sentiment label"):
            compute_vader_vs_consensus_agreement(df, "vader", "consensus")

        df.loc[2, "vader"] = "negative"
        df.loc[2, "consensus"] = "negative"
        res = compute_vader_vs_consensus_agreement(df, "vader", "consensus")
        self.assertEqual(res["n_samples"], 2)
        self.assertEqual(res["n_matches"], 2)

    def test_agreement_requires_two_annotators(self):
        with self.assertRaisesRegex(ValueError, "At least two annotators"):
            analyze_agreement(pd.DataFrame({"a": ["positive"]}), ["a"])

    def test_interpretation(self):
        self.assertIn("Almost Perfect", interpret_kappa(0.85))
        self.assertIn("Substantial", interpret_kappa(0.70))
        self.assertIn("Moderate", interpret_kappa(0.50))
        self.assertIn("Fair", interpret_kappa(0.30))
        self.assertIn("Slight", interpret_kappa(0.10))
        self.assertIn("Poor", interpret_kappa(-0.10))


if __name__ == "__main__":
    unittest.main()

