# CleanTrack - Green Supply Chain Dashboard

**An open-source Python ML toolkit for sustainable supply chain optimization** — combining logistics efficiency with circular economy principles to reduce carbon footprint and waste across the supply network.

---

## Project Goals

- **Optimize logistics routes** to minimize carbon emissions and fuel consumption
- **Predict & reduce waste** using ML models across the supply chain lifecycle
- **Score circularity** of materials and recommend end-of-life strategies
- **Simulate trade-offs** between cost, speed, and environmental impact

---

## Models & Features

| Module | Description | Algorithms |
|---|---|---|
| `RouteOptimizer` | Multi-objective route planning minimizing emissions + cost | Genetic Algorithm, OR-Tools |
| `WastePrediction` | Forecast waste generation at each supply chain node | XGBoost, LSTM |
| `CircularityScorer` | Rate materials on recyclability and reuse potential | Random Forest, Scoring Engine |
| `EmissionsEstimator` | Estimate Scope 1/2/3 emissions from logistics data | Regression, Emission Factors DB |
| `SupplierRiskModel` | Identify waste-prone and high-emission suppliers | Clustering, Anomaly Detection |

---

## Project Structure

```
CleanTrack_GSCP/
├── src/
│   ├── models/              # ML model implementations
│   │   ├── route_optimizer.py
│   │   ├── waste_predictor.py
│   │   ├── circularity_scorer.py
│   │   └── emissions_estimator.py
│   ├── optimization/        # Solver and constraint engines
│   │   └── multi_objective_solver.py
│   ├── data/                # Data loading, preprocessing, validation
│   │   ├── loader.py
│   │   └── preprocessor.py
│   └── utils/               # Shared utilities
│       ├── emissions_factors.py
│       └── visualizer.py
├── notebooks/               # Exploratory analysis & demos
│   ├── 01_route_optimization_demo.ipynb
│   ├── 02_waste_prediction_demo.ipynb
│   └── 03_circularity_analysis.ipynb
├── tests/                   # Unit and integration tests
├── data/
│   ├── raw/                 # Raw input datasets
│   └── processed/           # Cleaned/transformed data
├── configs/
│   └── config.yaml          # Model and pipeline configuration
├── docs/                    # Extended documentation
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/CleanTrack_GSCP.git
cd CleanTrack_GSCP
pip install -r requirements.txt
```

### 2. Run Route Optimization

```python
from src.models.route_optimizer import RouteOptimizer

optimizer = RouteOptimizer(objective="min_emissions")
result = optimizer.fit(routes_df)
print(result.optimal_routes)
print(f"CO2 saved: {result.co2_reduction_kg:.1f} kg")
```

### 3. Predict Waste

```python
from src.models.waste_predictor import WastePredictor

model = WastePredictor()
model.train(train_df)
forecasts = model.predict(future_df)
```

---

## Data Requirements

The toolkit works with standard supply chain datasets. See [`docs/data_schema.md`](docs/data_schema.md) for schema details.

| Dataset | Required Fields |
|---|---|
| Logistics | origin, destination, distance_km, vehicle_type, load_kg |
| Inventory | sku, quantity, material_type, shelf_life_days, waste_kg |
| Suppliers | supplier_id, location, product_categories, certifications |

Synthetic data generators are included for quick prototyping — see `src/data/loader.py`.

---

## Emission Factors

Built-in emission factor database based on:
- **IPCC** transport emission factors
- **GHG Protocol** Scope 1/2/3 methodology
- **Ecoinvent** material lifecycle data

---

## Roadmap

- [x] Core model architecture
- [x] Route optimization engine
- [x] Waste prediction pipeline
- [ ] Real-time dashboard (Streamlit)
- [ ] REST API wrapper (FastAPI)
- [ ] Integration with SAP / ERP connectors
- [ ] GIS-based emissions map visualization
- [ ] Benchmark datasets & leaderboard

---

## Contributing

Contributions welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a PR.

Areas we especially need help with:
- Additional emission factor datasets
- Integration with logistics APIs (e.g. Google Maps, HERE)
- More diverse industry test cases

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## References

- [GHG Protocol Corporate Standard](https://ghgprotocol.org/)
- [Ellen MacArthur Foundation Circular Economy](https://ellenmacarthurfoundation.org/)
- [OR-Tools by Google](https://developers.google.com/optimization)
