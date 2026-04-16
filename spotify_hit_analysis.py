"""
Predicting the Next Hit
Music Streaming · Content Performance Analysis
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "spotify-2023-2.csv"
PLOTS_DIR = "plots"

CLUSTER_NAMES = {0: "Less-known", 1: "Phenomenal", 2: "Well-known", 3: "Normal"}

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Helpers
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


# -----------------------------
# Load data
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    numeric_cols = [
        "streams", "artist_count", "released_year", "released_month", "released_day",
        "in_spotify_playlists", "in_spotify_charts",
        "in_apple_playlists", "in_apple_charts",
        "in_deezer_playlists", "in_deezer_charts",
        "in_shazam_charts", "bpm", "danceability_%", "valence_%",
        "energy_%", "acousticness_%", "instrumentalness_%",
        "liveness_%", "speechiness_%"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "key" in df.columns:
        df["key"] = df["key"].fillna("Unknown")
    if "mode" in df.columns:
        df["mode"] = df["mode"].fillna("Unknown")
    if "in_shazam_charts" in df.columns:
        df["in_shazam_charts"] = df["in_shazam_charts"].fillna(0)

    required_cols = ["track_name", "artist(s)_name", "streams", "released_month"]
    required_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=required_cols).copy()

    return df


# -----------------------------
# 1. Streams distribution
# -----------------------------
def plot_streams_distribution(df: pd.DataFrame) -> None:
    plt.figure()
    sns.histplot(df["streams"], bins=40)
    plt.title("Distribution of Streams")
    plt.xlabel("Streams")
    plt.ylabel("Frequency")
    save_plot("streams_distribution.png")


# -----------------------------
# 2. Audio features correlation
# -----------------------------
def analyze_audio_features(df: pd.DataFrame) -> None:
    features = [
        "bpm", "danceability_%", "valence_%", "energy_%",
        "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%"
    ]
    features = [c for c in features if c in df.columns]

    corr = df[features + ["streams"]].corr(numeric_only=True)

    print("\nCorrelation with streams:")
    print(corr["streams"].sort_values(ascending=False))

    plt.figure()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Audio Features Correlation with Streams")
    save_plot("audio_features_correlation.png")


# -----------------------------
# 3. Artist-tier comparison
# -----------------------------
def analyze_artist_tiers(df: pd.DataFrame) -> None:
    artist_streams = df.groupby("artist(s)_name")["streams"].sum().sort_values(ascending=False)

    top_10 = artist_streams.head(10)
    rest = artist_streams.iloc[10:]

    t_stat, p_val = ttest_ind(top_10.values, rest.values, equal_var=False)

    print("\nTop vs Non-Top Artists")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6g}")
    print(f"Top 10 avg streams: {top_10.mean():,.0f}")
    print(f"Non-top avg streams: {rest.mean():,.0f}")

    # Visualize artist tier comparison
    avg_streams = pd.DataFrame({
        "Group": ["Top 10 Artists", "Non-Top 10 Artists"],
        "Avg Streams": [top_10.mean(), rest.mean()]
    })

    plt.figure()
    sns.barplot(data=avg_streams, x="Group", y="Avg Streams", palette="Blues_d")
    plt.title("Average Streams: Top 10 vs Non-Top 10 Artists")
    plt.ylabel("Average Total Streams")
    save_plot("artist_tier_comparison.png")


# -----------------------------
# 4. Playlist regression
# -----------------------------
def run_playlist_regression(df: pd.DataFrame):
    x_cols = [c for c in ["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists"] if c in df.columns]

    model_df = df[x_cols + ["streams"]].dropna()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_x.fit_transform(model_df[x_cols])
    y = scaler_y.fit_transform(model_df[["streams"]]).flatten()

    # Use named columns for readable regression output
    X_df = pd.DataFrame(X, columns=x_cols)
    X_df = sm.add_constant(X_df)

    model = sm.OLS(y, X_df).fit()

    print("\nPlaylist Regression Summary")
    print(model.summary())

    print("\nKey coefficients (normalized):")
    for name, coef in zip(X_df.columns, model.params):
        print(f"  {name}: {coef:.4f}")

    preds = model.predict(X_df)

    plt.figure()
    plt.scatter(y, preds, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
    plt.title("Actual vs Predicted Normalized Streams (Playlist Model)")
    plt.xlabel("Actual Normalized Streams")
    plt.ylabel("Predicted Normalized Streams")
    save_plot("playlist_vs_streams.png")

    return model


# -----------------------------
# 5. Clustering + seasonality
# -----------------------------
def analyze_seasonality(df: pd.DataFrame) -> None:
    features = [c for c in [
        "in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists", "streams"
    ] if c in df.columns]

    cluster_df = df[features + ["released_month"]].dropna().copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(scaled)
    cluster_df["cluster_name"] = cluster_df["cluster"].map(CLUSTER_NAMES)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    plt.figure()
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_df["cluster"], cmap="viridis")
    plt.colorbar(scatter, label="Cluster")
    plt.title("Song Performance Clusters (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_plot("cluster_segments.png")

    cluster_df["season"] = cluster_df["released_month"].apply(get_season)

    table = pd.crosstab(cluster_df["season"], cluster_df["cluster_name"])
    chi2, p, _, _ = chi2_contingency(table)

    print("\nSeason vs Cluster")
    print(f"Chi-square: {chi2:.4f}, p-value: {p:.6g}")
    print(table)

    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    table = table.reindex(season_order)

    pct = table.div(table.sum(axis=1), axis=0)

    pct.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Season vs Performance Tier Distribution")
    plt.xlabel("Season")
    plt.ylabel("Proportion")
    plt.legend(title="Performance Tier", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot("season_vs_performance.png")


# -----------------------------
# Run analysis
# -----------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)

    plot_streams_distribution(df)
    analyze_audio_features(df)
    analyze_artist_tiers(df)
    run_playlist_regression(df)
    analyze_seasonality(df)

    print("\nDone. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
