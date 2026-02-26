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
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
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


def build_adjacency_matrix(
    units_gdf: gpd.GeoDataFrame,
    mode: str = "proximity",
    k_neighbors: int = 6,
    road_edges_path: Path | None = None,
) -> np.ndarray:
    """Build node adjacency from geographic proximity and optional road edges."""
    centroids = units_gdf.to_crs("EPSG:3857").centroid
    coords = np.c_[centroids.x.values, centroids.y.values]
    n_nodes = len(coords)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)

    if n_nodes <= 1:
        return adjacency

    if mode in {"proximity", "hybrid"}:
        dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        k_eff = max(1, min(k_neighbors, n_nodes - 1))
        for idx in range(n_nodes):
            neighbors = np.argpartition(dist[idx], k_eff)[:k_eff]
            adjacency[idx, neighbors] = 1.0
            adjacency[neighbors, idx] = 1.0

    if mode in {"road", "hybrid"} and road_edges_path is not None and road_edges_path.exists():
        road_df = pd.read_csv(road_edges_path)
        needed = {"source_grid_id", "target_grid_id"}
        missing = needed - set(road_df.columns)
        if missing:
            raise ValueError(f"Road edge CSV missing required columns: {sorted(missing)}")
        index_by_grid = pd.Series(units_gdf.index.values, index=units_gdf["grid_id"]).to_dict()
        for row in road_df.itertuples(index=False):
            src = index_by_grid.get(getattr(row, "source_grid_id"))
            dst = index_by_grid.get(getattr(row, "target_grid_id"))
            if src is not None and dst is not None and src != dst:
                adjacency[src, dst] = 1.0
                adjacency[dst, src] = 1.0

    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    deg = adjacency.sum(axis=1)
    deg = np.where(deg == 0, 1.0, deg)
    inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return inv_sqrt @ adjacency @ inv_sqrt


def temporal_gated_conv_layer(sequence: np.ndarray) -> np.ndarray:
    """Simple temporal gated layer over historical monthly crime counts."""
    # sequence shape: [n_nodes, time_window, in_dim]
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [n_nodes, time_window, in_dim]")
    in_dim = sequence.shape[-1]
    gate_w = np.ones((in_dim, in_dim), dtype=float) * 0.25
    cand_w = np.eye(in_dim, dtype=float)
    gate = 1.0 / (1.0 + np.exp(-(sequence @ gate_w)))
    candidate = np.tanh(sequence @ cand_w)
    gated = gate * candidate
    return gated.mean(axis=1)


def spatial_graph_conv_layer(node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """One-hop graph convolution for neighborhood spillover effects."""
    if node_features.ndim != 2:
        raise ValueError("node_features must have shape [n_nodes, hidden_dim]")
    adj_norm = _normalize_adjacency(adjacency)
    return adj_norm @ node_features


def zip_fairness_loss(
    y_true: np.ndarray,
    lam: np.ndarray,
    zero_prob: np.ndarray,
    cluster_ids: np.ndarray,
    fairness_weight: float,
) -> float:
    """Zero-inflated Poisson NLL + disparate impact penalty."""
    eps = 1e-8
    y = np.asarray(y_true, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), eps, None)
    pi = np.clip(np.asarray(zero_prob, dtype=float), eps, 1.0 - eps)

    is_zero = (y == 0).astype(float)
    poisson_zero = np.exp(-lam)
    ll_zero = np.log(pi + (1.0 - pi) * poisson_zero + eps)
    ll_pos = np.log(1.0 - pi + eps) + y * np.log(lam + eps) - lam
    zip_nll = -(is_zero * ll_zero + (1.0 - is_zero) * ll_pos).mean()

    expected_counts = (1.0 - pi) * lam
    global_mean = expected_counts.mean()
    disparity = 0.0
    for grp in np.unique(cluster_ids):
        grp_mean = expected_counts[cluster_ids == grp].mean()
        disparity += (grp_mean - global_mean) ** 2
    fairness_penalty = disparity / max(len(np.unique(cluster_ids)), 1)
    return float(zip_nll + fairness_weight * fairness_penalty)


class STZIFairRegressor:
    """Spatio-temporal regressor with gated temporal conv, graph conv, ZIP loss, and fairness penalty."""

    def __init__(self, fairness_weight: float = 0.1):
        self.fairness_weight = fairness_weight

    def fit(self, sequence: np.ndarray, adjacency: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray) -> "STZIFairRegressor":
        temporal = temporal_gated_conv_layer(sequence)
        self.graph_features_ = spatial_graph_conv_layer(temporal, adjacency)

        target_log = np.log1p(np.asarray(y, dtype=float))
        self.rate_coef_, *_ = np.linalg.lstsq(self.graph_features_, target_log, rcond=None)

        zero_target = (np.asarray(y) == 0).astype(int)
        clf = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
        clf.fit(self.graph_features_, zero_target)
        self.zero_clf_ = clf

        self.loss_ = zip_fairness_loss(
            y_true=y,
            lam=self.predict_lambda(sequence, adjacency),
            zero_prob=self.predict_zero_probability(sequence, adjacency),
            cluster_ids=cluster_ids,
            fairness_weight=self.fairness_weight,
        )
        return self

    def _transform(self, sequence: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        temporal = temporal_gated_conv_layer(sequence)
        return spatial_graph_conv_layer(temporal, adjacency)

    def predict_lambda(self, sequence: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        feats = self._transform(sequence, adjacency)
        return np.expm1(np.clip(feats @ self.rate_coef_, -20, 20)) + 1e-6

    def predict_zero_probability(self, sequence: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        feats = self._transform(sequence, adjacency)
        return self.zero_clf_.predict_proba(feats)[:, 1]

    def predict(self, sequence: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        lam = self.predict_lambda(sequence, adjacency)
        pi = self.predict_zero_probability(sequence, adjacency)
        return (1.0 - pi) * lam


def build_temporal_sequences(
    crime_gdf: gpd.GeoDataFrame,
    units_gdf: gpd.GeoDataFrame,
    time_window: int,
) -> np.ndarray:
    """Build monthly crime-count sequences per grid cell."""
    joined = gpd.sjoin(crime_gdf, units_gdf[["grid_id", "geometry"]], how="left", predicate="within")
    joined = joined.dropna(subset=["grid_id"]).copy()
    joined["year_month"] = joined["date"].dt.to_period("M").astype(str)
    monthly = joined.groupby(["grid_id", "year_month"]).size().rename("monthly_count").reset_index()
    pivot = (
        monthly.pivot(index="grid_id", columns="year_month", values="monthly_count")
        .reindex(units_gdf["grid_id"])
        .fillna(0.0)
    )
    if pivot.shape[1] == 0:
        return np.zeros((len(units_gdf), time_window, 1), dtype=float)

    arr = pivot.to_numpy(dtype=float)
    if arr.shape[1] < time_window:
        pad = np.zeros((arr.shape[0], time_window - arr.shape[1]), dtype=float)
        arr = np.concatenate([pad, arr], axis=1)
    else:
        arr = arr[:, -time_window:]
    return arr[:, :, None]


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
    parser.add_argument("--use_stzi_fair_model", action="store_true")
    parser.add_argument("--adjacency_mode", type=str, default="proximity", choices=["proximity", "road", "hybrid"])
    parser.add_argument("--road_edges_csv", type=Path, default=None)
    parser.add_argument("--k_neighbors", type=int, default=6)
    parser.add_argument("--time_window", type=int, default=12)
    parser.add_argument("--fairness_col", type=str, default="median_income")
    parser.add_argument("--fairness_weight", type=float, default=0.1)
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

    stzi_metadata: Dict[str, object] = {}
    if args.use_stzi_fair_model:
        print("Training spatio-temporal ZIP+fairness model...")
        adjacency = build_adjacency_matrix(
            units_gdf=modeling_df,
            mode=args.adjacency_mode,
            k_neighbors=args.k_neighbors,
            road_edges_path=args.road_edges_csv,
        )
        sequences = build_temporal_sequences(crime_gdf=crime_gdf, units_gdf=modeling_df, time_window=args.time_window)

        fairness_values = modeling_df[args.fairness_col]
        fairness_values = fairness_values.fillna(fairness_values.median())
        cluster_ids = pd.qcut(fairness_values, q=4, labels=False, duplicates="drop").to_numpy()

        stzi_model = STZIFairRegressor(fairness_weight=args.fairness_weight)
        stzi_model.fit(
            sequence=sequences,
            adjacency=adjacency,
            y=modeling_df["crime_count"].to_numpy(dtype=float),
            cluster_ids=cluster_ids,
        )
        stzi_pred = stzi_model.predict(sequences, adjacency)
        stzi_metrics = {
            "model": "stzi_fair",
            "rmse": float(np.sqrt(mean_squared_error(modeling_df["crime_count"], stzi_pred))),
            "mae": float(mean_absolute_error(modeling_df["crime_count"], stzi_pred)),
            "r2": float(r2_score(modeling_df["crime_count"], stzi_pred)),
            "zip_fair_loss": float(stzi_model.loss_),
        }
        pd.DataFrame([stzi_metrics]).to_csv(args.output_dir / "stzi_fair_metrics.csv", index=False)
        np.save(args.output_dir / "adjacency_matrix.npy", adjacency)
        np.save(args.output_dir / "temporal_sequences.npy", sequences)
        joblib.dump(stzi_model, args.output_dir / "stzi_fair_model.joblib")
        stzi_metadata = {
            "enabled": True,
            "adjacency_mode": args.adjacency_mode,
            "k_neighbors": args.k_neighbors,
            "time_window": args.time_window,
            "fairness_col": args.fairness_col,
            "fairness_weight": args.fairness_weight,
            "zip_fair_loss": float(stzi_model.loss_),
        }
        print("STZI-fair metrics:", stzi_metrics)

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
                "stzi_fair": stzi_metadata,
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
