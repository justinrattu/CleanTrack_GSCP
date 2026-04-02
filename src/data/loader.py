"""
Realistic synthetic data generators for green supply chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Material-specific properties
MATERIAL_PROPERTIES = {
    "aluminum": {"spoilage_rate": 0.01, "recyclability": 0.90, "lifespan": 10.0},
    "steel": {"spoilage_rate": 0.02, "recyclability": 0.85, "lifespan": 15.0},
    "cardboard": {"spoilage_rate": 0.15, "recyclability": 0.75, "lifespan": 0.5},
    "glass": {"spoilage_rate": 0.05, "recyclability": 0.95, "lifespan": 20.0},
    "hdpe_plastic": {"spoilage_rate": 0.08, "recyclability": 0.40, "lifespan": 5.0},
    "pet_plastic": {"spoilage_rate": 0.10, "recyclability": 0.35, "lifespan": 5.0},
    "mixed_plastic": {"spoilage_rate": 0.12, "recyclability": 0.15, "lifespan": 3.0},
    "textile": {"spoilage_rate": 0.20, "recyclability": 0.30, "lifespan": 2.0},
    "rubber": {"spoilage_rate": 0.05, "recyclability": 0.25, "lifespan": 8.0},
    "wood": {"spoilage_rate": 0.10, "recyclability": 0.60, "lifespan": 7.0},
}

CITIES = [
    ("Sydney", "Melbourne", 713),     # distance_km
    ("Melbourne", "Brisbane", 1700),
    ("Perth", "Adelaide", 2700),
    ("Brisbane", "Sydney", 909),
    ("Adelaide", "Melbourne", 726),
    ("Darwin", "Brisbane", 3150),
    ("Canberra", "Sydney", 238),
    ("Hobart", "Melbourne", 485),
]

VEHICLE_CHARACTERISTICS = {
    "diesel_truck_heavy": {"max_load_kg": 20000, "cost_per_km": 2.5, "emissions_per_km": 0.85},
    "diesel_truck_medium": {"max_load_kg": 10000, "cost_per_km": 1.8, "emissions_per_km": 0.55},
    "electric_truck": {"max_load_kg": 8000, "cost_per_km": 1.2, "emissions_per_km": 0.10},
    "rail_freight": {"max_load_kg": 100000, "cost_per_km": 0.3, "emissions_per_km": 0.08},
    "sea_freight": {"max_load_kg": 500000, "cost_per_km": 0.05, "emissions_per_km": 0.12},
}


def make_routes(n: int = 200) -> pd.DataFrame:
    """Generate realistic logistics routes with vehicle assignment based on distance/load."""
    selected_routes = [CITIES[i % len(CITIES)] for i in range(n)]
    origins = [r[0] for r in selected_routes]
    destinations = [r[1] for r in selected_routes]
    
    # Realistic distance variation around base route
    base_distances = [r[2] for r in selected_routes]
    distance_km = [d * RNG.uniform(0.9, 1.3) for d in base_distances]
    
    # Load distribution follows real patterns (some routes carry more)
    load_kg = RNG.uniform(500, 20000, n).round(0)
    
    # Assign vehicle type based on distance and load (realistic)
    vehicle_type = []
    for dist, load in zip(distance_km, load_kg):
        if dist > 2000 and load > 10000:
            # Long distance, heavy: rail or sea
            vtype = RNG.choice(["rail_freight", "sea_freight"])
        elif dist > 1500:
            # Long distance: electric truck or diesel heavy
            vtype = RNG.choice(["electric_truck", "diesel_truck_heavy"])
        elif load > 12000:
            # Heavy load: diesel heavy
            vtype = "diesel_truck_heavy"
        else:
            # Short/medium: efficient options
            vtype = RNG.choice(["electric_truck", "diesel_truck_medium"])
        vehicle_type.append(vtype)
    
    return pd.DataFrame({
        "origin": origins,
        "destination": destinations,
        "distance_km": np.array(distance_km).round(1),
        "load_kg": load_kg,
        "current_vehicle_type": vehicle_type,
        "emissions_kg_co2": [
            VEHICLE_CHARACTERISTICS[v]["emissions_per_km"] * d 
            for v, d in zip(vehicle_type, distance_km)
        ],
    })


def make_inventory(n: int = 500) -> pd.DataFrame:
    """Generate realistic inventory with material-specific spoilage patterns."""
    # Material distribution
    material = RNG.choice(list(MATERIAL_PROPERTIES.keys()), n)
    
    # Shelf life varies by material
    shelf_life = np.array([
        RNG.integers(3, 30) if MATERIAL_PROPERTIES[m]["lifespan"] < 1
        else RNG.integers(30, 365)
        for m in material
    ])
    
    # Days in storage: realistic with some items near expiry
    days_stored = (shelf_life * RNG.uniform(0.5, 1.5, n)).astype(int).clip(0, 500)
    
    # Exponential waste model (increases dramatically after shelf life)
    waste_kg = np.zeros(n)
    for i, (mat, shelf, days) in enumerate(zip(material, shelf_life, days_stored)):
        spoilage_rate = MATERIAL_PROPERTIES[mat]["spoilage_rate"]
        if days > shelf:
            # Exponential growth after shelf life
            excess_days = days - shelf
            waste_kg[i] = (5.0 * spoilage_rate) * np.exp(excess_days / 50.0)
        else:
            # Low baseline waste during shelf life
            waste_kg[i] = RNG.uniform(0, 0.5) + (days / shelf) * 0.3
    
    waste_kg = waste_kg.round(2)
    
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
    """Generate inventory with realistic, material-specific circularity metrics."""
    material = RNG.choice(list(MATERIAL_PROPERTIES.keys()), n)
    
    return pd.DataFrame({
        "sku": [f"PROD-{i:04d}" for i in range(n)],
        "material_type": material,
        # Realistic reuse potential based on material
        "reuse_potential": np.array([
            MATERIAL_PROPERTIES[m]["recyclability"] * RNG.uniform(0.7, 1.0)
            for m in material
        ]).round(3),
        # Lifespan from material properties
        "lifespan_years": np.array([
            MATERIAL_PROPERTIES[m]["lifespan"] * RNG.uniform(0.8, 1.2)
            for m in material
        ]).round(1),
        # Bio-based only for certain materials
        "is_bio_based": np.array([
            1 if m in ["textile", "wood"] and RNG.random() > 0.3 else 0
            for m in material
        ]),
        "unit_weight_kg": RNG.uniform(0.1, 50.0, n).round(2),
        "recyclability_region": RNG.choice(["AU", "APAC", "Global"], n),
    })
