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
    plt.title("Audio Features Correlation")
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

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    print("\nRegression Summary")
    print(model.summary())

    preds = model.predict(X)

    plt.figure()
    plt.scatter(y, preds, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
    plt.title("Playlist vs Streams")
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

    cluster_df = df[features + ["released_month"]].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_df["cluster"])
    plt.title("Cluster Segments")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_plot("cluster_segments.png")

    cluster_df["season"] = cluster_df["released_month"].apply(get_season)

    table = pd.crosstab(cluster_df["season"], cluster_df["cluster"])
    chi2, p, _, _ = chi2_contingency(table)

    print("\nSeason vs Cluster")
    print(f"Chi-square: {chi2:.4f}, p-value: {p:.6g}")

    pct = table.div(table.sum(axis=1), axis=0)

    pct.plot(kind="bar", stacked=True)
    plt.title("Season vs Performance")
    plt.xlabel("Season")
    plt.ylabel("Proportion")
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
