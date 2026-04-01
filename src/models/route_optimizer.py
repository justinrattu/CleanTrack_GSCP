"""
RouteOptimizer: Multi-objective logistics route optimization
minimizing carbon emissions and cost simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from loguru import logger

# Emission factors (kg CO2e per tonne-km) by vehicle type
EMISSION_FACTORS = {
    "diesel_truck_heavy": 0.0625,
    "diesel_truck_medium": 0.0890,
    "electric_truck": 0.0120,
    "rail_freight": 0.0280,
    "sea_freight": 0.0160,
    "air_freight": 0.6020,
}

# Cost factors (USD per tonne-km) by vehicle type
COST_FACTORS = {
    "diesel_truck_heavy": 0.12,
    "diesel_truck_medium": 0.18,
    "electric_truck": 0.14,
    "rail_freight": 0.05,
    "sea_freight": 0.03,
    "air_freight": 1.20,
}


@dataclass
class OptimizationResult:
    optimal_routes: pd.DataFrame
    total_emissions_kg: float
    total_cost_usd: float
    baseline_emissions_kg: float
    baseline_cost_usd: float

    @property
    def co2_reduction_kg(self) -> float:
        return self.baseline_emissions_kg - self.total_emissions_kg

    @property
    def co2_reduction_pct(self) -> float:
        if self.baseline_emissions_kg == 0:
            return 0.0
        return (self.co2_reduction_kg / self.baseline_emissions_kg) * 100

    @property
    def cost_delta_usd(self) -> float:
        return self.total_cost_usd - self.baseline_cost_usd

    def summary(self) -> str:
        return (
            f"=== Optimization Result ===\n"
            f"CO2 Reduction : {self.co2_reduction_kg:,.1f} kg "
            f"({self.co2_reduction_pct:.1f}%)\n"
            f"Cost Delta    : ${self.cost_delta_usd:+,.2f}\n"
            f"Total Routes  : {len(self.optimal_routes)}\n"
        )


class RouteOptimizer:
    """
    Multi-objective route optimizer for green logistics.

    Optimizes vehicle mode and routing to minimize a weighted combination
    of carbon emissions and cost using a greedy heuristic with optional
    Pareto-front exploration.

    Parameters
    ----------
    objective : {"min_emissions", "min_cost", "balanced"}
        Primary optimization objective.
    emissions_weight : float
        Weight on emissions in the [0, 1] range (used when objective="balanced").
    cost_weight : float
        Weight on cost in the [0, 1] range (used when objective="balanced").
    """

    def __init__(
        self,
        objective: Literal["min_emissions", "min_cost", "balanced"] = "balanced",
        emissions_weight: float = 0.6,
        cost_weight: float = 0.4,
    ):
        if objective not in {"min_emissions", "min_cost", "balanced"}:
            raise ValueError(f"Unknown objective: {objective!r}")
        self.objective = objective
        self.emissions_weight = emissions_weight
        self.cost_weight = cost_weight
        self._result: OptimizationResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, routes_df: pd.DataFrame) -> OptimizationResult:
        """
        Optimize vehicle mode assignments for a set of shipment legs.

        Parameters
        ----------
        routes_df : pd.DataFrame
            Must contain columns: origin, destination, distance_km,
            load_kg, current_vehicle_type.

        Returns
        -------
        OptimizationResult
        """
        self._validate_input(routes_df)
        logger.info(f"Optimizing {len(routes_df)} route legs (objective={self.objective})")

        routes = routes_df.copy()
        routes["load_tonnes"] = routes["load_kg"] / 1000.0

        # Compute baseline (current vehicle choices)
        routes["baseline_emissions_kg"] = routes.apply(
            lambda r: self._compute_emissions(r["distance_km"], r["load_tonnes"], r["current_vehicle_type"]),
            axis=1,
        )
        routes["baseline_cost_usd"] = routes.apply(
            lambda r: self._compute_cost(r["distance_km"], r["load_tonnes"], r["current_vehicle_type"]),
            axis=1,
        )

        # Select optimal vehicle for each leg
        routes["optimal_vehicle_type"] = routes.apply(self._select_vehicle, axis=1)
        routes["optimal_emissions_kg"] = routes.apply(
            lambda r: self._compute_emissions(r["distance_km"], r["load_tonnes"], r["optimal_vehicle_type"]),
            axis=1,
        )
        routes["optimal_cost_usd"] = routes.apply(
            lambda r: self._compute_cost(r["distance_km"], r["load_tonnes"], r["optimal_vehicle_type"]),
            axis=1,
        )

        self._result = OptimizationResult(
            optimal_routes=routes,
            total_emissions_kg=routes["optimal_emissions_kg"].sum(),
            total_cost_usd=routes["optimal_cost_usd"].sum(),
            baseline_emissions_kg=routes["baseline_emissions_kg"].sum(),
            baseline_cost_usd=routes["baseline_cost_usd"].sum(),
        )

        logger.success(self._result.summary())
        return self._result

    def pareto_front(self, routes_df: pd.DataFrame, n_points: int = 10) -> pd.DataFrame:
        """
        Compute the Pareto front across a range of emissions/cost weight combinations.

        Returns a DataFrame with columns: emissions_weight, cost_weight,
        total_emissions_kg, total_cost_usd.
        """
        self._validate_input(routes_df)
        records = []
        for ew in np.linspace(0, 1, n_points):
            cw = 1 - ew
            optimizer = RouteOptimizer(objective="balanced", emissions_weight=ew, cost_weight=cw)
            result = optimizer.fit(routes_df)
            records.append({
                "emissions_weight": ew,
                "cost_weight": cw,
                "total_emissions_kg": result.total_emissions_kg,
                "total_cost_usd": result.total_cost_usd,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_vehicle(self, row: pd.Series) -> str:
        """Pick the vehicle type with the lowest weighted score for a leg."""
        best_vehicle, best_score = None, float("inf")
        for vehicle in EMISSION_FACTORS:
            # Skip air freight for short distances (< 500 km)
            if vehicle == "air_freight" and row["distance_km"] < 500:
                continue
            # Skip sea/rail for very short legs (< 100 km)
            if vehicle in {"sea_freight", "rail_freight"} and row["distance_km"] < 100:
                continue

            emissions = self._compute_emissions(row["distance_km"], row["load_tonnes"], vehicle)
            cost = self._compute_cost(row["distance_km"], row["load_tonnes"], vehicle)

            if self.objective == "min_emissions":
                score = emissions
            elif self.objective == "min_cost":
                score = cost
            else:
                # Normalize by approximate max values to make weights meaningful
                score = self.emissions_weight * (emissions / 10000) + self.cost_weight * (cost / 5000)

            if score < best_score:
                best_score = score
                best_vehicle = vehicle

        return best_vehicle

    @staticmethod
    def _compute_emissions(distance_km: float, load_tonnes: float, vehicle: str) -> float:
        return EMISSION_FACTORS[vehicle] * distance_km * load_tonnes

    @staticmethod
    def _compute_cost(distance_km: float, load_tonnes: float, vehicle: str) -> float:
        return COST_FACTORS[vehicle] * distance_km * load_tonnes

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required = {"origin", "destination", "distance_km", "load_kg", "current_vehicle_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df["distance_km"].lt(0).any():
            raise ValueError("distance_km must be non-negative")
        if df["load_kg"].lt(0).any():
            raise ValueError("load_kg must be non-negative")
