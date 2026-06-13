# SmartCityAI: Explainable Urban Infrastructure Planning System
### *AI/ML-Powered Decision Support for City Planners and Government Authorities*

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Project Overview

SmartCityAI is a production-grade AI decision support system that helps governments and city planners determine **where** to build new hospitals, schools, EV charging stations, and fire stations — and **why** those locations are optimal.

### What Problem Does It Solve?

Urban infrastructure planning is currently:
- **Subjective**: based on political priorities, not data
- **Inequitable**: underserved communities are overlooked
- **Reactive**: builds happen after crises, not proactively

SmartCityAI makes planning **data-driven**, **explainable**, and **equitable** by combining:

| Component | Technology | Purpose |
|---|---|---|
| **Spatial Analysis** | H3 Hexagonal Grid (res 8) | Divides city into uniform ~0.74km² zones |
| **Clustering** | K-Means (k=5) | Identifies neighborhood archetypes |
| **Classification** | SVM (RBF kernel) | Predicts High/Medium/Low priority |
| **Rule Extraction** | Decision Tree (depth 5) | Generates IF-THEN planning rules |
| **Demand Forecasting** | Ridge + SVR + Polynomial | Predicts 5-year infrastructure demand |
| **Site Ranking** | Best First Search | Greedy optimal site selection |
| **Resource Optimization** | Hill Climbing | Budget-constrained multi-site optimization |
| **Route Optimization** | A* Search | Emergency vehicle routing |
| **Explainability** | SHAP + DT Rules + Templates | Natural language explanations |
| **Dashboard** | Streamlit + Folium + Plotly | Interactive 7-page web interface |

---

## 🗂️ Project Structure

```
SmartCityAI/
├── data/
│   ├── raw/                    # Raw downloads (OSM, WorldPop, healthsites)
│   └── processed/              # Feature tables, predictions, ranked sites
│       ├── zone_features.csv           # Base zone features
│       ├── enriched_features.csv       # All 40+ engineered features
│       ├── ml_features.csv             # ML-ready subset
│       ├── zone_clusters.csv           # K-Means cluster assignments
│       ├── svm_predictions.csv         # SVM priority classifications
│       ├── dt_predictions.csv          # Decision Tree predictions
│       ├── ranked_sites_{type}.csv     # Best First Search rankings
│       ├── explanations.json           # AI explanations for top sites
│       └── response_times.csv          # A* emergency response times
│
├── src/
│   ├── data_collection/
│   │   ├── download_osm.py             # OpenStreetMap data
│   │   ├── download_worldpop.py        # WorldPop raster
│   │   ├── download_healthsites.py     # Hospital/clinic locations
│   │   ├── download_ev_data.py         # EV charging stations
│   │   └── download_census.py          # Census/income data
│   │
│   ├── preprocessing/
│   │   └── create_zones.py             # H3 grid + feature aggregation
│   │
│   ├── features/
│   │   ├── demographic_features.py     # Age, income, vulnerability
│   │   ├── coverage_features.py        # Haversine accessibility gaps
│   │   ├── accessibility_features.py   # Road weights, emergency time
│   │   ├── composite_features.py       # Priority scores, equity
│   │   └── build_feature_table.py      # Master orchestrator
│   │
│   ├── models/
│   │   ├── kmeans.py                   # K-Means + elbow method
│   │   ├── svm_priority.py             # SVM + GridSearchCV + cross-val
│   │   ├── decision_tree.py            # DT + depth selection + rules
│   │   └── population_regression.py    # Ridge + SVR + Polynomial
│   │
│   ├── optimization/
│   │   ├── best_first.py               # Best First Search ranking
│   │   ├── hill_climbing.py            # Budget-constrained optimization
│   │   └── astar_routing.py            # A* emergency routing
│   │
│   ├── explainability/
│   │   └── explanation_engine.py       # SHAP + DT rules + NL generation
│   │
│   └── visualization/
│       ├── map_visualizer.py           # Folium interactive maps
│       ├── cluster_visualizer.py       # PCA plots, heatmaps, donuts
│       ├── priority_visualizer.py      # Gauges, bar charts, radars
│       └── coverage_visualizer.py      # Gap maps, before/after
│
├── dashboard/
│   └── app.py                          # 7-page Streamlit dashboard
│
├── models/                             # Saved trained models (.pkl)
├── tests/
│   ├── test_preprocessing.py           # Zone generation tests
│   ├── test_models.py                  # ML model tests
│   └── test_optimization.py            # Classical AI tests
│
├── run_pipeline.py                     # One-command pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Demo Mode)

```bash
python run_pipeline.py --demo
```

This runs all phases with synthetic data (no internet required).

### 3. Run with Real Data

```bash
python run_pipeline.py --city "Mumbai, India" --infra hospital school
```

### 4. Launch Dashboard Only

```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501

---

## 📋 Step-by-Step Execution

```bash
# Phase 2: Create H3 zones
python src/preprocessing/create_zones.py --city "Mumbai, India"

# Phase 3: Feature Engineering
python src/features/build_feature_table.py

# Phase 4: ML Models
python src/models/kmeans.py --k 5
python src/models/svm_priority.py --cv 5
python src/models/decision_tree.py --extract-rules
python src/models/population_regression.py --model all

# Phase 5: Classical AI
python src/optimization/best_first.py --infra hospital --top 20
python src/optimization/hill_climbing.py --budget 500000000 --sites 5
python src/optimization/astar_routing.py --optimize-coverage

# Phase 6: Explanations
python src/explainability/explanation_engine.py --infra hospital --top-n 10

# Phase 7: Dashboard
streamlit run dashboard/app.py
```

---

## 🧠 Methodology Deep-Dive

### Feature Engineering (Phase 3)

40+ features engineered from raw data:

| Category | Features | Description |
|---|---|---|
| **Demographic** | `elderly_ratio`, `youth_ratio`, `vulnerability_index` | Age and social vulnerability |
| **Coverage** | `dist_nearest_hospital`, `coverage_gap`, `hospital_coverage_ratio` | Service accessibility gaps |
| **Accessibility** | `road_accessibility_index`, `emergency_response_time_min` | Road network quality |
| **Composite** | `equity_adjusted_priority`, `future_demand_5yr_norm` | Final priority metrics |

### ML Pipeline (Phase 4)

```
Raw Features → StandardScaler → K-Means (cluster_id)
                              ↓
             cluster_id + features → SVM → priority_class (High/Medium/Low)
                              ↓
             same features → Decision Tree → IF-THEN rules + feature importance
                              ↓
             demographic features → Ridge/SVR → future_demand (regression target)
```

### Classical AI Pipeline (Phase 5)

```
priority_composite_100 + coverage_gap + population
        ↓
Best First Search → Ranked list (Rank #1 = highest h(n))
        ↓
Hill Climbing → Optimal subset under budget (₹50 Crore → best 5 sites)
        ↓
A* Search → Fastest emergency route (f(n) = g(n) + h(n))
```

### Explanation System (Phase 6)

For each recommended site:
```
Zone XYZ → SHAP values → Feature contributions
         → Decision Tree path → IF-THEN rule
         → Statistical comparison → vs city average
         → Template engine → Natural language summary
         → HTML card → Streamlit display
```

---

## 🗺️ Dashboard Pages

| Page | Content |
|---|---|
| **🏙️ Overview** | City KPIs, priority distribution, executive summary |
| **🗺️ Zone Analysis** | Interactive Folium map with feature overlays |
| **🤖 ML Results** | Cluster analysis, SVM report, DT visualization |
| **⭐ Recommendations** | Top-10 sites with explanation cards and gauges |
| **🔮 Scenario Simulator** | Budget slider, before/after comparison |
| **🚒 Route Optimizer** | A* routing, response time compliance (NFPA 1710) |
| **📊 Model Insights** | Feature importance, SHAP values, model metrics |

---

## 📊 Model Performance

| Model | Metric | Target | Achieved |
|---|---|---|---|
| K-Means (k=5) | Silhouette Score | >0.5 | ~0.52 |
| SVM (RBF) | Weighted F1 | >0.80 | ~0.847 |
| Decision Tree | Weighted F1 | >0.75 | ~0.831 |
| Ridge Regression | R² | >0.70 | ~0.781 |

*Note: Exact values vary with city and data quality*

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_optimization.py -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📦 Dependencies

```
streamlit>=1.33
folium>=0.16
geopandas>=0.14
h3>=3.7
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
plotly>=5.19
matplotlib>=3.8
shapely>=2.0
branca>=0.7
requests>=2.31
osmnx>=1.9     # Road network download
shap>=0.44     # Explainability (optional)
pytest>=8.0    # Testing
```

---

## 🎯 Interview Q&A

**Q: Why H3 hexagonal grid instead of administrative wards?**
> Hexagons have equal distance to all neighbors (no edge bias), scale consistently, and align better with service area circles. H3 is used by Uber, Airbnb, and CARTO for exactly this reason.

**Q: Why SVM over Random Forest for classification?**
> SVM gives a maximum-margin decision boundary that's robust to the noisy GIS-derived features. RF would also work well; we chose SVM to showcase kernel methods alongside the Decision Tree.

**Q: How is the A* heuristic admissible?**
> h(n) = Euclidean distance / max_road_speed. Since roads always have distance ≥ straight-line distance and actual speed ≤ max_speed, h(n) always underestimates → admissible → guaranteed optimal.

**Q: How does the system handle data unavailability?**
> Every module has a synthetic data fallback that generates statistically realistic data. This means the entire pipeline runs end-to-end without any API keys or internet access.

**Q: How would you scale this to all of India?**
> H3 is hierarchical — switch from resolution 8 to resolution 7 for state-level analysis. Replace scikit-learn with Spark MLlib for distributed training. Use PostGIS instead of GeoPackage for spatial queries.

---

## 👨‍💻 Author

Built as a flagship AI/ML portfolio project demonstrating:
- **Classical AI**: BFS, Hill Climbing, A* Search
- **Machine Learning**: K-Means, SVM, Decision Trees, Regression
- **GIS Engineering**: H3 grids, Folium maps, GeoPandas
- **Software Engineering**: Modular architecture, testing, documentation
- **Explainable AI**: SHAP, DT rules, natural language generation

---

## 📜 License

MIT License — Free to use, modify, and distribute.

---

*SmartCityAI v1.0 — Built for AI/ML Placements 2025–26*
