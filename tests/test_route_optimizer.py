"""Tests for RouteOptimizer."""
import pytest
import pandas as pd
from src.models.route_optimizer import RouteOptimizer
from src.data.loader import make_routes


@pytest.fixture
def sample_routes():
    return make_routes(50)


def test_fit_returns_result(sample_routes):
    opt = RouteOptimizer()
    result = opt.fit(sample_routes)
    assert result.optimal_routes is not None
    assert len(result.optimal_routes) == len(sample_routes)


def test_co2_reduction_is_nonnegative_for_min_emissions(sample_routes):
    opt = RouteOptimizer(objective="min_emissions")
    result = opt.fit(sample_routes)
    assert result.co2_reduction_kg >= -0.01  # small float tolerance


def test_all_objectives_run(sample_routes):
    for obj in ["min_emissions", "min_cost", "balanced"]:
        result = RouteOptimizer(objective=obj).fit(sample_routes)
        assert result.total_emissions_kg > 0


def test_pareto_front_length(sample_routes):
    opt = RouteOptimizer()
    pf = opt.pareto_front(sample_routes, n_points=5)
    assert len(pf) == 5


def test_missing_columns_raises():
    bad_df = pd.DataFrame({"origin": ["A"], "destination": ["B"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        RouteOptimizer().fit(bad_df)


def test_negative_distance_raises():
    routes = make_routes(5)
    routes.loc[0, "distance_km"] = -100
    with pytest.raises(ValueError, match="non-negative"):
        RouteOptimizer().fit(routes)
