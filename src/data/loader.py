"""
Synthetic data generators for green supply chain demos and testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

VEHICLE_TYPES = [
    "diesel_truck_heavy", "diesel_truck_medium",
    "electric_truck", "rail_freight", "sea_freight",
]

MATERIAL_TYPES = [
    "aluminum", "steel", "cardboard", "glass",
    "hdpe_plastic", "pet_plastic", "mixed_plastic",
    "textile", "rubber", "wood",
]

CITIES = [
    ("Sydney", "Melbourne"), ("Melbourne", "Brisbane"), ("Perth", "Adelaide"),
    ("Brisbane", "Sydney"), ("Adelaide", "Melbourne"), ("Darwin", "Brisbane"),
    ("Canberra", "Sydney"), ("Hobart", "Melbourne"),
]


def make_routes(n: int = 200) -> pd.DataFrame:
    """Generate synthetic logistics routes."""
    origins, destinations = zip(*[CITIES[i % len(CITIES)] for i in range(n)])
    return pd.DataFrame({
        "origin": origins,
        "destination": destinations,
        "distance_km": RNG.uniform(200, 4000, n).round(1),
        "load_kg": RNG.uniform(500, 20000, n).round(0),
        "current_vehicle_type": RNG.choice(VEHICLE_TYPES[:3], n),
    })


def make_inventory(n: int = 500) -> pd.DataFrame:
    """Generate synthetic inventory records with waste labels."""
    material = RNG.choice(MATERIAL_TYPES, n)
    shelf_life = RNG.integers(7, 365, n)
    days_stored = (shelf_life * RNG.uniform(0.3, 1.2, n)).astype(int).clip(0, 400)

    # Waste is higher when items are stored beyond shelf life
    overstock = np.maximum(days_stored - shelf_life, 0)
    waste_base = RNG.uniform(0, 5, n) + overstock * 0.5
    waste_kg = (waste_base * RNG.uniform(0.8, 1.2, n)).round(2)

    return pd.DataFrame({
        "sku": [f"SKU-{i:04d}" for i in range(n)],
        "material_type": material,
        "node_type": RNG.choice(["warehouse", "retail", "distribution"], n),
        "quantity_units": RNG.integers(10, 500, n),
        "shelf_life_days": shelf_life,
        "storage_temp_c": RNG.uniform(-5, 25, n).round(1),
        "days_in_storage": days_stored,
        "order_frequency": RNG.uniform(0.5, 4.0, n).round(2),
        "demand_variability": RNG.uniform(0.0, 1.0, n).round(3),
        "date": pd.date_range("2023-01-01", periods=n, freq="12h"),
        "waste_kg": waste_kg,
    })


def make_circularity_inventory(n: int = 300) -> pd.DataFrame:
    """Generate synthetic inventory for circularity scoring."""
    material = RNG.choice(MATERIAL_TYPES, n)
    return pd.DataFrame({
        "sku": [f"PROD-{i:04d}" for i in range(n)],
        "material_type": material,
        "reuse_potential": RNG.uniform(0.0, 1.0, n).round(3),
        "lifespan_years": RNG.uniform(0.5, 15.0, n).round(1),
        "is_bio_based": RNG.integers(0, 2, n),
        "unit_weight_kg": RNG.uniform(0.1, 50.0, n).round(2),
    })
