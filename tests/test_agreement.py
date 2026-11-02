"""Unit tests for Cohen's Kappa, Fleiss' Kappa (via statsmodels), and VADER agreement."""

import unittest
import pandas as pd

from src.inter_annotator_agreement import (
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_pairwise_cohen_kappas,
    compute_vader_vs_consensus_agreement,
    interpret_kappa,
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

    def test_interpretation(self):
        self.assertIn("Almost Perfect", interpret_kappa(0.85))
        self.assertIn("Substantial", interpret_kappa(0.70))
        self.assertIn("Moderate", interpret_kappa(0.50))
        self.assertIn("Fair", interpret_kappa(0.30))
        self.assertIn("Slight", interpret_kappa(0.10))
        self.assertIn("Poor", interpret_kappa(-0.10))


if __name__ == "__main__":
    unittest.main()
