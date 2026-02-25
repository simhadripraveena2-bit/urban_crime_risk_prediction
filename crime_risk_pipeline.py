#!/usr/bin/env python3
"""End-to-end urban crime risk prediction pipeline for ICML-style experiments.

This script:
1) Loads crime, environmental, and census datasets.
2) Performs spatial joins to aggregate counts per spatial unit.
3) Engineers tabular features (and optional lag feature).
4) Trains baseline and advanced ML models.
5) Evaluates with spatial cross-validation.
6) Produces plots and serializes trained artifacts.

Example:
python crime_risk_pipeline.py \
  --crime_csv data/crimes.csv \
  --bars_csv data/bars.csv \
  --streetlights_csv data/streetlights.csv \
  --vacant_csv data/vacant.csv \
  --census_path data/census.geojson \
  --output_dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "geopandas is required. Install dependencies from requirements.txt"
    ) from exc

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42


# --------------------------
# Loading + validation utils
# --------------------------
def load_spatial_units(census_path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load census/grid polygons from CSV(with WKT geometry) or GeoJSON/GeoPackage."""
    if census_path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
        gdf = gpd.read_file(census_path)
    elif census_path.suffix.lower() == ".csv":
        df = pd.read_csv(census_path)
        if "geometry" not in df.columns:
            raise ValueError("CSV census file must include a 'geometry' WKT column.")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs=crs)
    else:
        raise ValueError(f"Unsupported census file type: {census_path.suffix}")

    required_cols = {"grid_id", "population", "median_income", "geometry"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(f"Census data missing required columns: {sorted(missing)}")

    return gdf.to_crs(crs)


def _load_point_csv(path: Path, lat_col: str = "lat", lon_col: str = "lon") -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    needed = {lat_col, lon_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


def load_crime_data(crime_csv: Path) -> gpd.GeoDataFrame:
    required = {"crime_id", "crime_type", "date", "lat", "lon", "location_desc"}
    crime_df = pd.read_csv(crime_csv)
    missing = required - set(crime_df.columns)
    if missing:
        raise ValueError(f"Crime CSV missing required columns: {sorted(missing)}")

    crime_df["date"] = pd.to_datetime(crime_df["date"], errors="coerce")
    crime_df = crime_df.dropna(subset=["lat", "lon", "date"])
    return gpd.GeoDataFrame(
        crime_df,
        geometry=gpd.points_from_xy(crime_df["lon"], crime_df["lat"]),
        crs="EPSG:4326",
    )


# --------------------------
# Feature engineering
# --------------------------
def aggregate_point_features_to_units(
    points_gdf: gpd.GeoDataFrame,
    units_gdf: gpd.GeoDataFrame,
    out_col: str,
    predicate: str = "within",
) -> pd.DataFrame:
    """Spatially join points to polygons and count points per grid_id."""
    joined = gpd.sjoin(points_gdf, units_gdf[["grid_id", "geometry"]], how="left", predicate=predicate)
    counts = joined.groupby("grid_id").size().rename(out_col).reset_index()
    return counts


def build_feature_table(
    crime_gdf: gpd.GeoDataFrame,
    bars_gdf: gpd.GeoDataFrame,
    streetlights_gdf: gpd.GeoDataFrame,
    vacant_gdf: gpd.GeoDataFrame,
    units_gdf: gpd.GeoDataFrame,
    use_temporal_lag: bool,
) -> gpd.GeoDataFrame:
    units = units_gdf.copy()

    # Crime aggregation
    crime_joined = gpd.sjoin(crime_gdf, units[["grid_id", "geometry"]], how="left", predicate="within")
    crime_counts = (
        crime_joined.groupby("grid_id").size().rename("crime_count").reset_index()
    )

    # Environmental counts
    bars_counts = aggregate_point_features_to_units(bars_gdf, units, "bars_count")
    lights_counts = aggregate_point_features_to_units(streetlights_gdf, units, "streetlights_count")
    vacant_counts = aggregate_point_features_to_units(vacant_gdf, units, "vacant_buildings_count")

    # Merge all features onto units
    feature_df = (
        units[["grid_id", "population", "median_income", "geometry"]]
        .merge(crime_counts, on="grid_id", how="left")
        .merge(bars_counts, on="grid_id", how="left")
        .merge(lights_counts, on="grid_id", how="left")
        .merge(vacant_counts, on="grid_id", how="left")
    )

    # Fill count NAs and derive crime_rate
    count_cols = ["crime_count", "bars_count", "streetlights_count", "vacant_buildings_count"]
    feature_df[count_cols] = feature_df[count_cols].fillna(0)
    feature_df["population"] = feature_df["population"].replace(0, np.nan)
    feature_df["crime_rate"] = (feature_df["crime_count"] / feature_df["population"]) * 1000.0

    # Optional temporal lag feature
    if use_temporal_lag:
        crime_month = crime_joined.dropna(subset=["grid_id"]).copy()
        crime_month["year_month"] = crime_month["date"].dt.to_period("M")
        monthly = (
            crime_month.groupby(["grid_id", "year_month"]).size().rename("monthly_count").reset_index()
        )
        monthly["lag1"] = monthly.groupby("grid_id")["monthly_count"].shift(1)
        lag_feature = monthly.groupby("grid_id")["lag1"].mean().rename("crime_lag1_mean").reset_index()
        feature_df = feature_df.merge(lag_feature, on="grid_id", how="left")
    else:
        feature_df["crime_lag1_mean"] = np.nan

    return gpd.GeoDataFrame(feature_df, geometry="geometry", crs=units.crs)


def build_spatial_groups(units_gdf: gpd.GeoDataFrame, n_bins: int = 5) -> np.ndarray:
    """Build coarse spatial groups for leakage-resistant CV."""
    centroids = units_gdf.to_crs("EPSG:3857").centroid
    x, y = centroids.x.values, centroids.y.values

    x_bin = pd.qcut(x, q=min(n_bins, len(np.unique(x))), duplicates="drop", labels=False)
    y_bin = pd.qcut(y, q=min(n_bins, len(np.unique(y))), duplicates="drop", labels=False)
    groups = (pd.Series(x_bin).astype(str) + "_" + pd.Series(y_bin).astype(str)).to_numpy()
    return groups


# --------------------------
# Modeling
# --------------------------
def get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "linear_regression": LinearRegression(),
        "poisson_regression": PoissonRegressor(alpha=1.0, max_iter=1000),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=4,
        )
    return models


def evaluate_with_spatial_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                feature_cols,
            )
        ]
    )

    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    all_results = []
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, model in get_models().items():
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            y_true = y.iloc[test_idx]

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "r2": float(r2_score(y_true, y_pred)),
                }
            )

        model_result = pd.DataFrame(fold_metrics)
        all_results.append(model_result)

        final_pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        final_pipe.fit(X, y)
        fitted_models[model_name] = final_pipe

    result_df = pd.concat(all_results, ignore_index=True)
    summary = result_df.groupby("model")[["rmse", "mae", "r2"]].mean().sort_values("rmse")
    print("\n=== Spatial CV Summary (mean across folds) ===")
    print(summary)
    return result_df, fitted_models


# --------------------------
# Visualization
# --------------------------
def plot_actual_vs_predicted_map(
    gdf: gpd.GeoDataFrame,
    model_name: str,
    model: Pipeline,
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    pred_col = f"pred_{model_name}"
    gdf[pred_col] = model.predict(gdf[feature_cols])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gdf.plot(column="crime_rate", cmap="Reds", legend=True, ax=axes[0], missing_kwds={"color": "lightgrey"})
    axes[0].set_title("Actual Crime Rate (per 1k)")
    axes[0].axis("off")

    gdf.plot(column=pred_col, cmap="Blues", legend=True, ax=axes[1], missing_kwds={"color": "lightgrey"})
    axes[1].set_title(f"Predicted Crime Rate ({model_name})")
    axes[1].axis("off")

    fig.tight_layout()
    out_path = output_dir / f"actual_vs_pred_{model_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    model_name: str,
    model: Pipeline,
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        importances = np.abs(coef)
    else:
        return

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance: {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=200)
    plt.close(fig)


# --------------------------
# Main orchestration
# --------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban crime risk ML pipeline")
    parser.add_argument("--crime_csv", type=Path, required=True)
    parser.add_argument("--bars_csv", type=Path, required=True)
    parser.add_argument("--streetlights_csv", type=Path, required=True)
    parser.add_argument("--vacant_csv", type=Path, required=True)
    parser.add_argument("--census_path", type=Path, required=True)
    parser.add_argument("--target", type=str, default="crime_rate", choices=["crime_rate", "crime_count"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--use_temporal_lag", action="store_true")
    parser.add_argument("--n_splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    units_gdf = load_spatial_units(args.census_path)
    crime_gdf = load_crime_data(args.crime_csv)
    bars_gdf = _load_point_csv(args.bars_csv)
    lights_gdf = _load_point_csv(args.streetlights_csv)
    vacant_gdf = _load_point_csv(args.vacant_csv)

    print("Building feature table...")
    feature_gdf = build_feature_table(
        crime_gdf=crime_gdf,
        bars_gdf=bars_gdf,
        streetlights_gdf=lights_gdf,
        vacant_gdf=vacant_gdf,
        units_gdf=units_gdf,
        use_temporal_lag=args.use_temporal_lag,
    )

    feature_cols = [
        "crime_count",
        "bars_count",
        "streetlights_count",
        "vacant_buildings_count",
        "population",
        "median_income",
        "crime_lag1_mean",
    ]

    modeling_df = feature_gdf.dropna(subset=[args.target]).copy()
    groups = build_spatial_groups(modeling_df)

    print("Training and evaluating models...")
    cv_results, fitted_models = evaluate_with_spatial_cv(
        modeling_df,
        feature_cols=feature_cols,
        target_col=args.target,
        groups=groups,
        n_splits=args.n_splits,
    )

    cv_results.to_csv(args.output_dir / "cv_results.csv", index=False)
    modeling_df.to_file(args.output_dir / "feature_table.geojson", driver="GeoJSON")

    leaderboard = cv_results.groupby("model")["rmse"].mean().sort_values()
    best_model_name = leaderboard.index[0]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, args.output_dir / "best_model.joblib")
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "target": args.target,
                "feature_cols": feature_cols,
                "xgboost_available": XGBOOST_AVAILABLE,
            },
            f,
            indent=2,
        )

    print(f"Best model: {best_model_name}")
    print("Generating visualizations...")

    # Predicted vs actual map for best model
    plot_actual_vs_predicted_map(modeling_df.copy(), best_model_name, best_model, feature_cols, args.output_dir)

    # Importance plots for each model when available
    for name, mdl in fitted_models.items():
        plot_feature_importance(name, mdl, feature_cols, args.output_dir)

    print(f"Done. Artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
