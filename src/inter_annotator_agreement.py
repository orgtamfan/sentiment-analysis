"""Inter-annotator agreement metrics: Cohen's Kappa, Fleiss' Kappa, and VADER agreement.

This module provides functions and a CLI to compute:
1. Cohen's Kappa per pair of annotators using scikit-learn (`cohen_kappa_score`).
   Pairs only require valid annotations for those two annotators (does not discard rows
   missing labels from an un-involved third annotator).
2. Fleiss' Kappa across all annotators simultaneously using statsmodels (`aggregate_raters` and `fleiss_kappa`),
   using rows with complete ratings across all specified raters.
3. Landis & Koch (1977) interpretation benchmarks for Kappa values.
4. Agreement rate between VADER suggestions and final consensus / manual labels (simple percentage match).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


VALID_CATEGORIES = {"positive", "neutral", "negative"}


def normalize_labels(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Normalize annotator label columns: strip whitespace, lowercase, and validate.

    Empty strings and whitespace-only values are converted to pd.NA.
    Validates that any non-empty label belongs to {'positive', 'neutral', 'negative'}.

    Args:
        df: Input DataFrame containing annotator columns.
        columns: Column names to normalize.

    Returns:
        A copy of df with normalized columns.

    Raises:
        ValueError: If any unrecognized categorical value is encountered.
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        cleaned = (
            result[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
        )
        non_na = cleaned.dropna()
        invalid = set(non_na.unique()) - VALID_CATEGORIES
        if invalid:
            raise ValueError(
                f"Column '{col}' contains invalid sentiment label(s): {sorted(invalid)}. "
                f"Allowed values: {sorted(VALID_CATEGORIES)}."
            )
        result[col] = cleaned
    return result


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value according to Landis & Koch (1977) benchmark.

    Scale:
      < 0.00      : Poor (worse than chance)
      0.00 - 0.20 : Slight
      0.21 - 0.40 : Fair
      0.41 - 0.60 : Moderate
      0.61 - 0.80 : Substantial
      0.81 - 1.00 : Almost Perfect
    """
    if np.isnan(kappa):
        return "Undefined"
    if kappa < 0.0:
        return "Poor (< 0.00)"
    elif kappa <= 0.20:
        return "Slight (0.00 - 0.20)"
    elif kappa <= 0.40:
        return "Fair (0.21 - 0.40)"
    elif kappa <= 0.60:
        return "Moderate (0.41 - 0.60)"
    elif kappa <= 0.80:
        return "Substantial (0.61 - 0.80)"
    else:
        return "Almost Perfect (0.81 - 1.00)"


def compute_cohens_kappa(
    rater1: Sequence[Any],
    rater2: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> float:
    """Compute Cohen's Kappa score between two annotators using scikit-learn."""
    return float(cohen_kappa_score(rater1, rater2, labels=labels))


def compute_pairwise_cohen_kappas(
    df: pd.DataFrame,
    rater_columns: Sequence[str],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute pairwise Cohen's Kappa for all annotator column combinations.

    Only drops rows where either rater in the pair is NaN, preserving rows that
    might be missing an annotation from an uninvolved third rater.

    Args:
        df: DataFrame containing annotations.
        rater_columns: List/tuple of column names corresponding to raters.
        labels: Optional explicit list of class labels.

    Returns:
        Dictionary containing pairwise kappa scores, interpretations, and mean kappa.
    """
    normalized_df = normalize_labels(df, rater_columns)
    pairs = list(itertools.combinations(rater_columns, 2))
    results: dict[str, Any] = {"pairs": {}, "mean_cohens_kappa": 0.0}

    scores = []
    for col1, col2 in pairs:
        valid_mask = normalized_df[col1].notna() & normalized_df[col2].notna()
        r1 = normalized_df.loc[valid_mask, col1]
        r2 = normalized_df.loc[valid_mask, col2]
        if valid_mask.sum() == 0:
            score = float("nan")
        else:
            score = compute_cohens_kappa(r1, r2, labels=labels)
        pair_key = f"{col1}_vs_{col2}"
        kappa_val = round(score, 4) if np.isfinite(score) else None
        results["pairs"][pair_key] = {
            "rater1": col1,
            "rater2": col2,
            "n_samples": int(valid_mask.sum()),
            "kappa": kappa_val,
            "interpretation": interpret_kappa(score),
        }
        if np.isfinite(score):
            scores.append(score)

    if scores:
        mean_score = float(np.mean(scores))
        results["mean_cohens_kappa"] = round(mean_score, 4)
        results["mean_interpretation"] = interpret_kappa(mean_score)
    else:
        results["mean_cohens_kappa"] = None
        results["mean_interpretation"] = "Undefined (no valid pair scores)"

    return results


def compute_fleiss_kappa(
    df: pd.DataFrame,
    rater_columns: Sequence[str],
    method: str = "fleiss",
) -> dict[str, Any]:
    """Compute Fleiss' Kappa across all annotators using statsmodels.

    Uses `statsmodels.stats.inter_rater.aggregate_raters` to convert text categories
    into a contingency count matrix, then computes Fleiss' Kappa.

    Requires complete annotations across all specified rater columns.

    Args:
        df: DataFrame containing annotations (only rows with complete ratings).
        rater_columns: List/tuple of column names representing annotators.
        method: Method argument passed to statsmodels `fleiss_kappa` ('fleiss').

    Returns:
        Dictionary with Fleiss' Kappa score, category mappings, and interpretation.
    """
    normalized_df = normalize_labels(df, rater_columns)
    clean_df = normalized_df.dropna(subset=rater_columns).copy()
    if len(clean_df) == 0:
        return {
            "kappa": None,
            "interpretation": "Undefined (no complete subjects)",
            "categories": sorted(list(VALID_CATEGORIES)),
            "n_subjects": 0,
            "n_raters": len(rater_columns),
        }

    labels_matrix = clean_df[list(rater_columns)].values
    contingency_table, categories = aggregate_raters(labels_matrix)
    score = float(fleiss_kappa(contingency_table, method=method))

    kappa_val = round(score, 4) if np.isfinite(score) else None
    return {
        "kappa": kappa_val,
        "interpretation": interpret_kappa(score),
        "categories": list(categories),
        "n_subjects": int(len(clean_df)),
        "n_raters": len(rater_columns),
    }


def compute_vader_vs_consensus_agreement(
    df: pd.DataFrame,
    vader_col: str = "sentiment_suggestion",
    consensus_col: str = "sentiment_final",
) -> dict[str, Any]:
    """Calculate simple agreement rate between VADER suggestions and final consensus.

    Args:
        df: DataFrame containing both columns.
        vader_col: Column name of VADER suggestions.
        consensus_col: Column name of consensus / manual adjudication labels.

    Returns:
        Dictionary containing matched count, total valid count, and agreement rate.
    """
    if vader_col not in df.columns or consensus_col not in df.columns:
        raise ValueError(
            f"Columns '{vader_col}' or '{consensus_col}' not found in dataframe."
        )

    valid_mask = df[vader_col].notna() & df[consensus_col].notna()
    subset = df.loc[valid_mask].copy()

    vader_clean = subset[vader_col].astype(str).str.strip().str.lower()
    consensus_clean = subset[consensus_col].astype(str).str.strip().str.lower()

    matches = vader_clean == consensus_clean
    n_total = int(len(subset))
    n_matches = int(matches.sum())
    agreement_rate = float(matches.mean()) if n_total > 0 else 0.0

    return {
        "vader_column": vader_col,
        "consensus_column": consensus_col,
        "n_samples": n_total,
        "n_matches": n_matches,
        "agreement_rate": round(agreement_rate, 4),
        "agreement_percentage": f"{agreement_rate:.2%}",
    }


def analyze_agreement(
    df: pd.DataFrame,
    rater_columns: Sequence[str],
    vader_col: str | None = None,
    consensus_col: str | None = None,
) -> dict[str, Any]:
    """Run full agreement analysis: pairwise Cohen's Kappa, Fleiss' Kappa, and optional VADER match.

    Cohen's Kappa uses all available pairs from df (does not unnecessarily drop rows
    if an uninvolved third rater is NaN).
    Fleiss' Kappa uses only rows with complete annotations across all raters.

    Args:
        df: DataFrame with annotations.
        rater_columns: Annotator column names.
        vader_col: Optional column for VADER suggestions.
        consensus_col: Optional column for consensus labels.

    Returns:
        Structured dictionary of agreement metrics.
    """
    # 1. Pairwise Cohen: use full df (each pair drops only its own NaNs)
    pairwise = compute_pairwise_cohen_kappas(df, rater_columns=rater_columns)

    # 2. Fleiss multi-rater: internally filters to rows with complete annotations across all raters
    fleiss = compute_fleiss_kappa(df, rater_columns=rater_columns)

    report: dict[str, Any] = {
        "n_subjects_complete": fleiss["n_subjects"],
        "n_raters": len(rater_columns),
        "raters": list(rater_columns),
        "categories": fleiss["categories"],
        "cohens_kappa": pairwise,
        "fleiss_kappa": {
            "kappa": fleiss["kappa"],
            "interpretation": fleiss["interpretation"],
        },
    }

    if (
        vader_col
        and consensus_col
        and vader_col in df.columns
        and consensus_col in df.columns
    ):
        report["vader_vs_consensus"] = compute_vader_vs_consensus_agreement(
            df, vader_col=vader_col, consensus_col=consensus_col
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's Kappa, Fleiss' Kappa, and VADER agreement rate."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/manual_label.csv"),
        help="Path to CSV dataset (default: data/processed/manual_label.csv).",
    )
    parser.add_argument(
        "--raters",
        nargs="+",
        default=None,
        help="Annotator column names. Defaults to detecting annotator_1..3 or label_1..3.",
    )
    parser.add_argument(
        "--vader-col",
        type=str,
        default="sentiment_suggestion",
        help="Column name for VADER suggestion (default: sentiment_suggestion).",
    )
    parser.add_argument(
        "--consensus-col",
        type=str,
        default=None,
        help="Column name for final consensus label (defaults to detecting sentiment_konsensus or sentiment_final).",
    )
    parser.add_argument(
        "--by-game",
        action="store_true",
        help="If set and 'game' column exists, compute agreement breakdown per game.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON results.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    # Auto-detect rater columns if not specified
    raters = args.raters
    if raters is None:
        candidate_sets = [
            ["annotator_1", "annotator_2", "annotator_3"],
            ["label_1", "label_2", "label_3"],
        ]
        for candidate in candidate_sets:
            if set(candidate).issubset(df.columns):
                raters = candidate
                break

        if raters is None:
            found = [
                c
                for c in df.columns
                if c.startswith(("annotator_", "label_"))
                and c not in ("label_review",)
            ]
            if len(found) >= 2:
                raters = sorted(found)
            else:
                raise ValueError(
                    f"Could not automatically detect at least 2 rater columns in {list(df.columns)}. "
                    "Please specify using --raters col1 col2 col3."
                )

    # Auto-detect consensus column if not specified
    consensus_col = args.consensus_col
    if consensus_col is None:
        for c in ["sentiment_konsensus", "sentiment_final", "sentiment"]:
            if c in df.columns:
                consensus_col = c
                break

    vader_col = args.vader_col if args.vader_col in df.columns else None

    print("=" * 65)
    print("INTER-ANNOTATOR AGREEMENT & VADER BENCHMARK")
    print("=" * 65)
    print(f"Dataset   : {args.input}")
    print(f"Raters ({len(raters)}): {', '.join(raters)}")
    if vader_col and consensus_col:
        print(f"Consensus : {consensus_col} | VADER : {vader_col}")

    results: dict[str, Any] = {
        "overall": analyze_agreement(
            df,
            rater_columns=raters,
            vader_col=vader_col,
            consensus_col=consensus_col,
        )
    }

    overall = results["overall"]
    print(f"Complete subjects (Fleiss): {overall['n_subjects_complete']}")
    print(f"Categories: {', '.join(overall['categories'])}")

    print("\n[1. Cohen's Kappa per Pasangan Annotator (scikit-learn)]")
    for pair_name, details in overall["cohens_kappa"]["pairs"].items():
        k_str = f"{details['kappa']:.4f}" if details['kappa'] is not None else "null"
        print(
            f"  {details['rater1']} vs {details['rater2']}: "
            f"kappa = {k_str} ({details['interpretation']}) [N={details['n_samples']}]"
        )
    mean_k_str = (
        f"{overall['cohens_kappa']['mean_cohens_kappa']:.4f}"
        if overall['cohens_kappa']['mean_cohens_kappa'] is not None
        else "null"
    )
    print(
        f"  Mean Cohen's Kappa: {mean_k_str} "
        f"({overall['cohens_kappa']['mean_interpretation']})"
    )

    print("\n[2. Fleiss' Kappa Semua Annotator Sekaligus (statsmodels)]")
    f_res = overall["fleiss_kappa"]
    f_k_str = f"{f_res['kappa']:.4f}" if f_res['kappa'] is not None else "null"
    print(f"  Fleiss' Kappa: {f_k_str} ({f_res['interpretation']})")

    if "vader_vs_consensus" in overall:
        v_res = overall["vader_vs_consensus"]
        print("\n[3. Agreement Rate VADER vs. Konsensus Manual (Rumusan Masalah #4)]")
        print(f"  Matched: {v_res['n_matches']} / {v_res['n_samples']}")
        print(f"  Agreement rate: {v_res['agreement_percentage']}")

    if args.by_game and "game" in df.columns:
        results["by_game"] = {}
        for game_val, group in df.groupby("game"):
            game_key = str(game_val).strip().lower()
            game_analysis = analyze_agreement(
                group,
                rater_columns=raters,
                vader_col=vader_col,
                consensus_col=consensus_col,
            )
            results["by_game"][game_key] = game_analysis

            print(f"\n--- Game: {game_val} (Complete N = {game_analysis['n_subjects_complete']}) ---")
            for pair_name, details in game_analysis["cohens_kappa"]["pairs"].items():
                k_str = f"{details['kappa']:.4f}" if details['kappa'] is not None else "null"
                print(
                    f"  Cohen {details['rater1']} vs {details['rater2']}: "
                    f"{k_str} ({details['interpretation']}) [N={details['n_samples']}]"
                )
            mean_g_str = (
                f"{game_analysis['cohens_kappa']['mean_cohens_kappa']:.4f}"
                if game_analysis['cohens_kappa']['mean_cohens_kappa'] is not None
                else "null"
            )
            print(
                f"  Mean Cohen: {mean_g_str} "
                f"({game_analysis['cohens_kappa']['mean_interpretation']})"
            )
            fg = game_analysis['fleiss_kappa']
            fg_k_str = f"{fg['kappa']:.4f}" if fg['kappa'] is not None else "null"
            print(
                f"  Fleiss    : {fg_k_str} "
                f"({fg['interpretation']})"
            )
            if "vader_vs_consensus" in game_analysis:
                vg = game_analysis["vader_vs_consensus"]
                print(f"  VADER vs Konsensus: {vg['agreement_percentage']} ({vg['n_matches']}/{vg['n_samples']})")

    print("=" * 65)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
