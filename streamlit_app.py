#!/usr/bin/env python3
"""Optional Streamlit dashboard for urban crime risk predictions."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Urban Crime Risk", layout="wide")
st.title("Urban Crime Risk Prediction Dashboard")

st.sidebar.header("Configuration")
artifacts_dir = Path(st.sidebar.text_input("Artifacts directory", "outputs"))

meta_path = artifacts_dir / "metadata.json"
model_path = artifacts_dir / "best_model.joblib"
features_path = artifacts_dir / "feature_table.geojson"

if not (meta_path.exists() and model_path.exists() and features_path.exists()):
    st.warning("Missing artifacts. Run crime_risk_pipeline.py first.")
    st.stop()

with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = joblib.load(model_path)
gdf = gpd.read_file(features_path)
feature_cols = metadata["feature_cols"]
target = metadata["target"]

crime_type_filter = st.sidebar.text_input("Crime type filter (for display only)", "All")
min_pop = st.sidebar.slider("Minimum population", 0, int(gdf["population"].fillna(0).max()), 0)

view_df = gdf[gdf["population"].fillna(0) >= min_pop].copy()
view_df["predicted_risk"] = model.predict(view_df[feature_cols])

threshold = st.sidebar.number_input("High-risk threshold", value=float(view_df["predicted_risk"].quantile(0.8)))
view_df["high_risk"] = (view_df["predicted_risk"] >= threshold).astype(int)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicted vs Actual")
    chart_df = view_df[["grid_id", target, "predicted_risk"]].melt("grid_id", var_name="series", value_name="value")
    fig = px.bar(chart_df, x="grid_id", y="value", color="series", barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("High-risk Areas")
    st.metric("Selected units", len(view_df))
    st.metric("High-risk units", int(view_df["high_risk"].sum()))

st.subheader("Spatial View")
if view_df.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}:
    map_df = view_df.to_crs("EPSG:4326")
    map_fig = px.choropleth_mapbox(
        map_df,
        geojson=map_df.__geo_interface__,
        locations=map_df.index,
        color="predicted_risk",
        mapbox_style="carto-positron",
        center={"lat": float(map_df.geometry.centroid.y.mean()), "lon": float(map_df.geometry.centroid.x.mean())},
        zoom=10,
        opacity=0.6,
        hover_data=["grid_id", target, "predicted_risk"],
    )
    st.plotly_chart(map_fig, use_container_width=True)
else:
    st.info("Geometry is not polygonal. Skipping map panel.")

st.subheader("Download")
out_csv = view_df.drop(columns="geometry", errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=out_csv, file_name="crime_risk_predictions.csv", mime="text/csv")

st.caption(f"Crime type filter selected: {crime_type_filter} (placeholder filter for custom extensions)")
