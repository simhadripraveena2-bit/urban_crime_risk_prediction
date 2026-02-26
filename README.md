# Urban Crime Risk Prediction (ICML-style Pipeline)

This repository contains a full Python project for predicting urban crime risk from:
- Crime incidents
- Environmental features (bars, streetlights, vacant buildings)
- Census/grid demographics

## Files
- `crime_risk_pipeline.py`: End-to-end preprocessing, feature engineering, spatial CV, modeling, and plots.
- `streamlit_app.py`: Optional interactive app to inspect predictions.
- `requirements.txt`: Dependencies.

## Input Data Schema

### 1) Crime CSV
Required columns:
- `crime_id`, `crime_type`, `date`, `lat`, `lon`, `location_desc`

### 2) Environmental CSVs
Each file must contain:
- `lat`, `lon`

Use separate files for:
- Bars
- Streetlights
- Vacant Buildings

### 3) Census / Spatial Units
Either:
- GeoJSON / shapefile-like format with columns `grid_id`, `population`, `median_income`, `geometry`, or
- CSV with the same columns and geometry as WKT in `geometry`.

## Run Training Pipeline

```bash
python crime_risk_pipeline.py \
  --crime_csv data/crime.csv \
  --bars_csv data/bars.csv \
  --streetlights_csv data/streetlights.csv \
  --vacant_csv data/vacant_buildings.csv \
  --census_path data/census.geojson \
  --output_dir outputs \
  --use_temporal_lag
```

## Spatio-Temporal Graph + ZIP + Fairness Extension

The pipeline now includes an optional graph-based model that:
- Represents neighborhoods as graph nodes.
- Builds graph edges from geographic proximity (`--adjacency_mode proximity`) or a road-connectivity edge list (`--adjacency_mode road`, `--road_edges_csv`).
- Applies a temporal gated convolution layer over monthly crime sequences.
- Applies a spatial graph convolution layer to model risk spillover to adjacent neighborhoods.
- Optimizes a Zero-Inflated Poisson objective with an additional fairness penalty to reduce disparate impact across demographic clusters.

Example:

```bash
python crime_risk_pipeline.py \
  --crime_csv data/crime.csv \
  --bars_csv data/bars.csv \
  --streetlights_csv data/streetlights.csv \
  --vacant_csv data/vacant_buildings.csv \
  --census_path data/census.geojson \
  --output_dir outputs \
  --use_stzi_fair_model \
  --adjacency_mode hybrid \
  --road_edges_csv data/road_edges.csv \
  --k_neighbors 6 \
  --time_window 12 \
  --fairness_col median_income \
  --fairness_weight 0.1
```

Road edge CSV schema (for `road`/`hybrid`):
- `source_grid_id`
- `target_grid_id`

### What it does
1. Aggregates crimes per spatial unit.
2. Computes `crime_rate` per 1,000 population.
3. Spatially maps each environmental dataset and counts features per unit.
4. Merges all features with demographics.
5. Trains:
   - Linear Regression
   - Poisson Regression
   - Random Forest
   - XGBoost (if installed)
6. Evaluates with spatial cross-validation (`GroupKFold` using coarse spatial bins).
7. Saves:
   - `outputs/cv_results.csv`
   - `outputs/feature_table.geojson`
   - `outputs/best_model.joblib`
   - `outputs/metadata.json`
   - map and feature-importance plots
8. Optionally (when `--use_stzi_fair_model` is enabled), also saves:
   - `outputs/stzi_fair_model.joblib`
   - `outputs/stzi_fair_metrics.csv`
   - `outputs/adjacency_matrix.npy`
   - `outputs/temporal_sequences.npy`

## Optional Streamlit App

```bash
streamlit run streamlit_app.py
```

In the app:
- Configure artifacts directory (default `outputs/`)
- Inspect predicted vs actual risk
- View high-risk area counts
- Download prediction CSV

## Notes
- Optional temporal features are added with `--use_temporal_lag`.
- You can switch target using `--target crime_count` or default `crime_rate`.
- For advanced spatio-temporal/GNN variants, use exported feature table and adjacency construction as a follow-up extension.
