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
from sklearn.metrics import silhouette_score
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
    """Save the current matplotlib figure to the plots folder."""
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def get_season(month: int) -> str:
    """Map release month to season."""
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


# -----------------------------
# Load and clean data
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load dataset and clean key fields."""
    df = pd.read_csv(path, encoding="latin-1")

    numeric_cols = [
        "streams",
        "artist_count",
        "released_year",
        "released_month",
        "released_day",
        "in_spotify_playlists",
        "in_spotify_charts",
        "in_apple_playlists",
        "in_apple_charts",
        "in_deezer_playlists",
        "in_deezer_charts",
        "in_shazam_charts",
        "bpm",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
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

    required_cols = ["track_name", "artist(s)_name", "streams", "released_year", "released_month"]
    required_cols = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=required_cols).copy()

    return df


# -----------------------------
# 1. Streams distribution
# -----------------------------
def plot_streams_distribution(df: pd.DataFrame) -> None:
    """Plot overall distribution of song streams."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["streams"], bins=40)
    plt.title("Distribution of Streams")
    plt.xlabel("Streams")
    plt.ylabel("Frequency")
    save_plot("streams_distribution.png")


# -----------------------------
# 2. Audio features vs streams
# -----------------------------
def analyze_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create correlation heatmap for audio features and streams."""
    feature_cols = [
        "bpm",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    corr_df = df[feature_cols + ["streams"]].corr(numeric_only=True)

    print("\nCorrelation with streams:")
    print(corr_df["streams"].sort_values(ascending=False))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Audio Features Correlation Heatmap")
    save_plot("audio_features_correlation.png")

    return corr_df


# -----------------------------
# 3. Artist-tier comparison
# -----------------------------
def analyze_artist_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare top 10 artists vs non-top 10 artists on average audio features.
    This supports the insight that top artists tend to have different feature profiles.
    """
    artist_streams = (
        df.groupby("artist(s)_name")["streams"]
        .sum()
        .sort_values(ascending=False)
    )

    top_10_artists = artist_streams.head(10)
    non_top_10_artists = artist_streams.iloc[10:]

    t_stat, p_val = ttest_ind(
        top_10_artists.values,
        non_top_10_artists.values,
        equal_var=False,
        nan_policy="omit",
    )

    print("\nTop 10 vs non-top 10 artist total streams:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6g}")

    df = df.copy()
    top_10_names = set(top_10_artists.index)

    df["artist_group"] = np.where(
        df["artist(s)_name"].isin(top_10_names),
        "Top 10 Artists",
        "Non-Top 10 Artists",
    )

    audio_features = [
        "danceability_%",
        "energy_%",
        "acousticness_%",
        "valence_%",
        "speechiness_%",
    ]
    audio_features = [col for col in audio_features if col in df.columns]

    summary = df.groupby("artist_group")[audio_features].mean().T

    print("\nAverage audio features by artist group:")
    print(summary)

    return summary


# -----------------------------
# 4. Playlist regression
# -----------------------------
def run_playlist_regression(df: pd.DataFrame):
    """
    Model how playlist placements relate to streams.
    Outputs a plot of actual vs predicted normalized streams.
    """
    x_cols = [
        col for col in ["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists"]
        if col in df.columns
    ]
    y_col = "streams"

    model_df = df[x_cols + [y_col]].dropna().copy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        x_scaler.fit_transform(model_df[x_cols]),
        columns=x_cols,
        index=model_df.index,
    )

    y_scaled = pd.Series(
        y_scaler.fit_transform(model_df[[y_col]]).flatten(),
        index=model_df.index,
        name="streams_scaled",
    )

    X_const = sm.add_constant(X_scaled)
    model = sm.OLS(y_scaled, X_const).fit()

    print("\nPlaylist regression summary:")
    print(model.summary())

    predictions = model.predict(X_const)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_scaled, predictions, alpha=0.6)
    plt.plot(
        [y_scaled.min(), y_scaled.max()],
        [y_scaled.min(), y_scaled.max()],
        linestyle="--",
        color="red",
    )
    plt.xlabel("Actual Normalized Streams")
    plt.ylabel("Predicted Normalized Streams")
    plt.title("Playlist Regression: Actual vs Predicted Streams")
    save_plot("playlist_vs_streams.png")

    return model


# -----------------------------
# 5. Clustering and seasonality
# -----------------------------
def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster songs into performance groups and test whether season
    is associated with performance tier.
    """
    cluster_features = [
        col for col in [
            "in_spotify_playlists",
            "in_apple_playlists",
            "in_deezer_playlists",
            "streams",
        ]
        if col in df.columns
    ]

    cluster_df = df[cluster_features + ["released_month"]].dropna().copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df[cluster_features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    cluster_df["Cluster"] = cluster_labels

    # PCA plot for visualizing clusters
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, alpha=0.7)
    plt.title("K-Means Cluster Segments")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    save_plot("cluster_segments.png")

    silhouette = silhouette_score(scaled_features, cluster_labels)
    print(f"\nSilhouette score: {silhouette:.3f}")

    cluster_summary = cluster_df.groupby("Cluster")[cluster_features].mean()
    print("\nCluster summary:")
    print(cluster_summary)

    # Label clusters by average stream performance
    stream_rank = cluster_summary["streams"].sort_values().index.tolist()
    cluster_name_map = {
        stream_rank[0]: "Less-known",
        stream_rank[1]: "Normal",
        stream_rank[2]: "Well-known",
        stream_rank[3]: "Phenomenal",
    }

    cluster_df["Cluster_Name"] = cluster_df["Cluster"].map(cluster_name_map)
    cluster_df["Season"] = cluster_df["released_month"].apply(get_season)

    season_table = pd.crosstab(cluster_df["Season"], cluster_df["Cluster_Name"])
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    season_table = season_table.reindex(season_order)

    print("\nSeason x performance tier:")
    print(season_table)

    chi2, p_val, dof, expected = chi2_contingency(season_table)
    print("\nChi-square test: season vs performance tier")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_val:.6g}")

    season_pct = season_table.div(season_table.sum(axis=1), axis=0) * 100

    season_pct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
    plt.title("Season vs Performance Tier")
    plt.xlabel("Season")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot("season_vs_performance.png")

    return cluster_df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)

    plot_streams_distribution(df)
    analyze_audio_features(df)
    analyze_artist_tiers(df)
    run_playlist_regression(df)
    analyze_seasonality(df)

    print(f"\nAnalysis complete. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
