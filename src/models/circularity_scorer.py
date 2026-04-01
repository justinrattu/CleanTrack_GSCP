"""
CircularityScorer: Score materials and products on circular economy
principles — recyclability, reuse potential, and end-of-life strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


# Material recyclability index (0–1 scale, based on Ellen MacArthur Foundation data)
MATERIAL_RECYCLABILITY = {
    "aluminum": 0.92,
    "steel": 0.88,
    "cardboard": 0.85,
    "glass": 0.80,
    "hdpe_plastic": 0.72,
    "pet_plastic": 0.68,
    "ldpe_plastic": 0.45,
    "mixed_plastic": 0.30,
    "textile": 0.35,
    "electronic_components": 0.55,
    "rubber": 0.40,
    "wood": 0.60,
    "composite": 0.20,
    "unknown": 0.10,
}

EOL_STRATEGIES = {
    (0.8, 1.0): "Recycle",
    (0.6, 0.8): "Refurbish / Resell",
    (0.4, 0.6): "Repurpose / Upcycle",
    (0.2, 0.4): "Energy Recovery",
    (0.0, 0.2): "Landfill (minimize)",
}


@dataclass
class CircularityReport:
    scored_df: pd.DataFrame
    mean_score: float
    material_breakdown: pd.DataFrame

    def summary(self) -> str:
        top_eol = self.scored_df["eol_strategy"].value_counts().idxmax()
        return (
            f"=== Circularity Report ===\n"
            f"Mean Score : {self.mean_score:.3f} / 1.000\n"
            f"Items Scored: {len(self.scored_df)}\n"
            f"Top EOL Strategy: {top_eol}\n"
        )


class CircularityScorer:
    """
    Score items on a 0–1 circular economy index and recommend
    end-of-life (EOL) strategies.

    Scoring formula (weighted composite):
        score = w_recycle * recyclability
              + w_reuse   * reuse_potential
              + w_lifespan* normalized_lifespan
              + w_bio     * is_bio_based

    Parameters
    ----------
    weights : dict
        Custom weights for each scoring dimension.
    """

    DEFAULT_WEIGHTS = {
        "recyclability": 0.40,
        "reuse_potential": 0.30,
        "lifespan": 0.20,
        "bio_based": 0.10,
    }

    def __init__(self, weights: dict | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, inventory_df: pd.DataFrame) -> CircularityReport:
        """
        Score all items in an inventory dataframe.

        Parameters
        ----------
        inventory_df : pd.DataFrame
            Required columns:
            - sku (str): item identifier
            - material_type (str): must be a key in MATERIAL_RECYCLABILITY
            - reuse_potential (float 0–1): estimated reuse/refurb likelihood
            - lifespan_years (float): expected useful life
            - is_bio_based (bool/int): 1 if bio-based material, else 0

        Returns
        -------
        CircularityReport
        """
        self._validate_input(inventory_df)
        df = inventory_df.copy()

        df["recyclability"] = df["material_type"].str.lower().map(
            lambda m: MATERIAL_RECYCLABILITY.get(m, MATERIAL_RECYCLABILITY["unknown"])
        )
        df["lifespan_norm"] = self._normalize(df["lifespan_years"])

        df["circularity_score"] = (
            self.weights["recyclability"] * df["recyclability"]
            + self.weights["reuse_potential"] * df["reuse_potential"].clip(0, 1)
            + self.weights["lifespan"] * df["lifespan_norm"]
            + self.weights["bio_based"] * df["is_bio_based"].clip(0, 1)
        )

        df["eol_strategy"] = df["circularity_score"].map(self._eol_strategy)
        df["circularity_grade"] = df["circularity_score"].map(self._grade)

        material_breakdown = (
            df.groupby("material_type")
            .agg(
                count=("sku", "count"),
                mean_score=("circularity_score", "mean"),
                mean_recyclability=("recyclability", "mean"),
            )
            .sort_values("mean_score", ascending=False)
        )

        return CircularityReport(
            scored_df=df,
            mean_score=df["circularity_score"].mean(),
            material_breakdown=material_breakdown,
        )

    def improvement_plan(self, report: CircularityReport, top_n: int = 10) -> pd.DataFrame:
        """
        Return top_n items with the lowest scores and substitution suggestions.
        """
        low = report.scored_df.nsmallest(top_n, "circularity_score").copy()
        low["suggested_material"] = low["material_type"].map(self._suggest_substitute)
        low["potential_score_gain"] = low["suggested_material"].map(
            lambda m: MATERIAL_RECYCLABILITY.get(m, 0)
        ) - low["recyclability"]
        return low[["sku", "material_type", "circularity_score", "eol_strategy",
                     "suggested_material", "potential_score_gain"]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - series.min()) / rng

    @staticmethod
    def _eol_strategy(score: float) -> str:
        for (low, high), strategy in EOL_STRATEGIES.items():
            if low <= score <= high:
                return strategy
        return "Unknown"

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.80:
            return "A"
        if score >= 0.65:
            return "B"
        if score >= 0.50:
            return "C"
        if score >= 0.35:
            return "D"
        return "F"

    @staticmethod
    def _suggest_substitute(material: str) -> str:
        substitutes = {
            "mixed_plastic": "hdpe_plastic",
            "ldpe_plastic": "hdpe_plastic",
            "composite": "aluminum",
            "textile": "cardboard",
            "rubber": "hdpe_plastic",
            "unknown": "cardboard",
        }
        return substitutes.get(material.lower(), material)

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0; got {total:.3f}")

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required = {"sku", "material_type", "reuse_potential", "lifespan_years", "is_bio_based"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
