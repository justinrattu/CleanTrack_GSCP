"""Tests for CircularityScorer."""
import pytest
import pandas as pd
from src.models.circularity_scorer import CircularityScorer
from src.data.loader import make_circularity_inventory


@pytest.fixture
def inventory():
    return make_circularity_inventory(100)


def test_scores_are_in_range(inventory):
    scorer = CircularityScorer()
    report = scorer.score(inventory)
    assert report.scored_df["circularity_score"].between(0, 1).all()


def test_grades_are_valid(inventory):
    scorer = CircularityScorer()
    report = scorer.score(inventory)
    assert report.scored_df["circularity_grade"].isin(["A", "B", "C", "D", "F"]).all()


def test_improvement_plan_length(inventory):
    scorer = CircularityScorer()
    report = scorer.score(inventory)
    plan = scorer.improvement_plan(report, top_n=5)
    assert len(plan) == 5


def test_invalid_weights_raises():
    with pytest.raises(ValueError, match="sum to 1.0"):
        CircularityScorer(weights={"recyclability": 0.5, "reuse_potential": 0.5,
                                   "lifespan": 0.5, "bio_based": 0.5})


def test_missing_columns_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        CircularityScorer().score(pd.DataFrame({"sku": ["X"]}))
