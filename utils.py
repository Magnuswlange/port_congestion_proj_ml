from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
from geopy.distance import geodesic
import glob
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.geometry import Polygon
import pandas as pd


BBOX_MAX_LAT = 39.492119
BBOX_MIN_LAT = 39.3653539
BBOX_MIN_LON = -0.343666
BBOX_MAX_LON = -0.213547
BBOX: Polygon = box(BBOX_MIN_LON, BBOX_MIN_LAT, BBOX_MAX_LON, BBOX_MAX_LAT)

def print_worst_predictions(df: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series, y_pred: np.ndarray) -> None:
    voyage_id_map = df[["voyage_id", "pos_timestamp"]].loc[X_val.index].copy()

    errors_df = pd.DataFrame(
        {
            "voyage_id": voyage_id_map["voyage_id"],
            "pos_timestamp": voyage_id_map["pos_timestamp"],
            "predicted": y_pred,
            "actual": y_val,
            "error": y_pred - y_val,
        }
    )

    # Worst voyages
    print("Worst underpredictions:")
    print(
        errors_df.nsmallest(10, "error")[["voyage_id", "error", "predicted", "actual"]]
    )

    print("\nWorst overpredictions:")
    print(
        errors_df.nlargest(10, "error")[["voyage_id", "error", "predicted", "actual"]]
    )


def split_train_val_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_model = df.copy()
    features = get_features(df)

    # 3) Build X, y
    X = features
    y = df_model["tta_hours"]

    # 4) Time-based split
    split_idx = int(len(df_model) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_val, y_train, y_val


def get_valid_voyages_pct(df: pd.DataFrame, og_df: pd.DataFrame) -> float:
    valid_voyage_ids = df["voyage_id"].unique()
    valid_voyages_pct = (
        valid_voyage_ids.shape[0] / len(og_df["voyage_id"].unique()) * 100
    )
    return valid_voyages_pct


def eval_predictions(y_val: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    mae: float = mean_absolute_error(y_val, y_pred)
    mse: float = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2: float = r2_score(y_val, y_pred)

    print(f"MAE (hours): {mae:.2f}")
    print(f"RMSE (hours): {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    errors = y_pred - y_val
    print(f"\nErrors (hours): \n{errors.describe().round(2)}")

    return errors


def export_voyage_ids_to_csv(df: pd.DataFrame, out: str) -> None:
    valid_voyage_ids = df["voyage_id"].unique()
    valid_voyage_ids_df = pd.DataFrame({"voyage_id": valid_voyage_ids})

    valid_voyage_ids_df.to_csv(
        out + ".csv",
        index=False,
        encoding="utf-8",
    )


def plot_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=["number"])
    corr = num_df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", center=0)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.show()


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "deg_off_course",
        "voyage_status",
        "navigation_status",
        "loading_pct",
        "position_accuracy",
        "delta_dist_km",
        "remaining_dist_live",
        "speed_over_ground",
        "draught",
        "build_year",
        "month",
        "length",
        "width",
        "capacity_nt",
        "capacity_dwt",
        "position_device",
    ]
    features = df[feature_cols]
    features.shape
    return features


def calc_remaining_dist(row):
    current = (row["latitude"], row["longitude"])
    arrival = (row["arrival_latitude"], row["arrival_longitude"])
    return geodesic(current, arrival).km


def plot_all_voyages(
    df: pd.DataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    poly: Polygon,
    padding: float = 2,
) -> None:
    """Plot ALL unique voyages from dataframe."""

    unique_voyages = df["voyage_id"].unique()
    print(f"Found {len(unique_voyages)} voyages.")

    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    # Get overall bounds for ALL voyages
    all_lons = df["longitude"]
    all_lats = df["latitude"]
    min_lon, max_lon = all_lons.min(), all_lons.max()
    min_lat, max_lat = all_lats.min(), all_lats.max()

    # Single figure for ALL
    _, ax = plt.subplots(figsize=(14, 14))
    coastline_gdf.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)

    # Plot each voyage with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_voyages)))

    for i, voyage_id in enumerate(unique_voyages):
        voyage_data = df[df["voyage_id"] == voyage_id].sort_values("pos_timestamp")

        ax.plot(
            voyage_data["longitude"],
            voyage_data["latitude"],
            linestyle="-",
            linewidth=0.8,
            color=colors[i],
            alpha=0.75,
        )

    # Bounding box
    poly_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=2, label="Bounding box")

    ax.set_xlim((min_lon - padding, max_lon + padding))
    ax.set_ylim((min_lat - padding, max_lat + padding))
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"All Voyages ({len(unique_voyages)}) in Bounding Box")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_voyage_on_coastline(
    voyage_df: pd.DataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    poly: Polygon,
    padding: float = 0.1,
) -> None:
    # Sort by timestamp
    voyage_df = voyage_df.sort_values(by="pos_timestamp", ascending=True, inplace=False)

    # Get voyage coord bounds
    min_lon, max_lon = voyage_df["longitude"].min(), voyage_df["longitude"].max()
    min_lat, max_lat = voyage_df["latitude"].min(), voyage_df["latitude"].max()

    # Calculate range (square window)
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    full_range = max(lon_range, lat_range)

    pad = full_range * padding
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    x_lims = (center_lon - full_range / 2 - pad, center_lon + full_range / 2 + pad)
    y_lims = (center_lat - full_range / 2 - pad, center_lat + full_range / 2 + pad)

    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    # Single figure
    _, ax = plt.subplots(figsize=(12, 12))
    coastline_gdf.plot(ax=ax, color="darkgreen", edgecolor="black", label="Coastline")

    # Voyage
    ax.plot(
        voyage_df["longitude"],
        voyage_df["latitude"],
        marker="o",
        linestyle="-",
        color="blue",
        label="Voyage Trajectory",
        markersize=2,
        markeredgecolor="black",
    )

    # BBox on SAME axes
    poly_gdf.boundary.plot(
        ax=ax, facecolor="red", edgecolor="black", alpha=0.75, label="Bounding box"
    )

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_aspect("equal")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Voyage: {voyage_df['voyage_id'].iloc[0]}")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print(
        f"Voyage: {voyage_df['voyage_id'].iloc[0]} "
        f"\nTotal traveled distance: {voyage_df['total_dist_km'].iloc[0].round(2)} km"
        f"\nDuration: {(voyage_df['duration'].iloc[0] / 3600).round(2)} hours"
        f"\nData points: {voyage_df.shape[0]}"
    )


def draw_bbox(coastline_gdf: gpd.GeoDataFrame, poly: Polygon) -> None:
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[poly], crs="EPSG:4326"
    )  # Coordinat Reference System, WGS84 lat/long in degrees

    # Plot
    _, ax = plt.subplots(figsize=(6, 6))
    coastline_gdf.plot(ax=ax, color="darkgreen", edgecolor="black")
    bbox_gdf.boundary.plot(ax=ax, facecolor="red", edgecolor="black", alpha=0.75)

    # Add padding around bbox (200%)
    pad_lon = (BBOX_MAX_LON - BBOX_MIN_LON) * 4
    pad_lat = (BBOX_MAX_LAT - BBOX_MIN_LAT) * 4

    x_lims = (BBOX_MIN_LON - pad_lon, BBOX_MAX_LON + pad_lon)
    y_lims = (BBOX_MIN_LAT - pad_lat, BBOX_MAX_LAT + pad_lat)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Berth bounding box")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def load_parquet(path_pattern: str, engine: str = "fastparquet") -> pd.DataFrame:
    """
    Loads parquet file(s).
    Supports glob patterns like "*.parquet.gzip".

    Args:
        path_pattern: File path or glob pattern.
        engine: Parquet engine.

    Returns:
        Single DataFrame, concatenated if multiple files.
    """

    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {path_pattern}")

    dfs = [pd.read_parquet(f, engine=engine) for f in files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f'Loaded {len(df):,} rows from {len(dfs)} files matching "{path_pattern}"')

    return df