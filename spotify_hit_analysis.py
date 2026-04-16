"""
Predicting the Next Hit
Music Streaming · Content Performance Analysis

This script analyzes Spotify's top songs of 2023 to identify variables
most strongly associated with streaming performance.

Methods used:
- t-tests
- chi-square test
- multiple linear regression
- K-means clustering
- PCA
- exploratory data analysis

Dataset:
Top Spotify Songs 2023 (Kaggle)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "spotify-2023-2.csv"
PLOTS_DIR = "plots"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Utility functions
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()


def get_season(month: int) -> str:
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

    required = ["track_name", "artist(s)_name", "streams", "released_year", "released_month"]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required).copy()

    return df


# -----------------------------
# Section 1: Artist-tier analysis
# -----------------------------
def analyze_artist_tiers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    artist_streams = (
        df.groupby("artist(s)_name")["streams"]
        .sum()
        .sort_values(ascending=False)
    )

    top_10 = artist_streams.head(10)
    non_top_10 = artist_streams.iloc[10:]

    print("\nTop 10 Artists by Total Streams")
    print(top_10)

    plt.figure()
    sns.barplot(x=top_10.values, y=top_10.index)
    plt.title("Top 10 Artists by Total Streams")
    plt.xlabel("Total Streams")
    plt.ylabel("Artist")
    save_plot("top_10_artists_bar.png")

    stream_share = (top_10 / top_10.sum()) * 100
    plt.figure(figsize=(8, 8))
    plt.pie(
        stream_share.values,
        labels=stream_share.index,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Stream Share of Top 10 Artists")
    save_plot("top_10_artists_pie.png")

    t_stat, p_val = ttest_ind(
        top_10.values,
        non_top_10.values,
        equal_var=False,
        nan_policy="omit"
    )
    print("\nT-test: Top 10 vs Non-Top 10 Artist Total Streams")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6g}")

    compare_df = pd.DataFrame({
        "streams": np.concatenate([top_10.values, non_top_10.values]),
        "group": ["Top 10 Artists"] * len(top_10) + ["Non-Top 10 Artists"] * len(non_top_10)
    })

    plt.figure()
    sns.boxplot(data=compare_df, x="group", y="streams")
    plt.title("Artist Total Streams: Top 10 vs Non-Top 10")
    plt.xlabel("")
    plt.ylabel("Total Streams")
    save_plot("artist_tier_boxplot.png")

    return top_10, non_top_10


# -----------------------------
# Section 2: Audio characteristics
# -----------------------------
def analyze_audio_features(df: pd.DataFrame, top_10: pd.Series) -> pd.DataFrame:
    df = df.copy()
    top_10_names = set(top_10.index)

    df["artist_group"] = np.where(
        df["artist(s)_name"].isin(top_10_names),
        "Top 10 Artists",
        "Non-Top 10 Artists"
    )

    audio_features = [
        "danceability_%", "energy_%", "acousticness_%",
        "valence_%", "speechiness_%"
    ]
    audio_features = [c for c in audio_features if c in df.columns]

    summary = df.groupby("artist_group")[audio_features].mean().T
    print("\nAverage Audio Features by Artist Group")
    print(summary)

    summary.plot(kind="bar")
    plt.title("Average Audio Features: Top 10 vs Non-Top 10 Artists")
    plt.xlabel("Feature")
    plt.ylabel("Average Value")
    plt.xticks(rotation=45)
    save_plot("audio_feature_comparison.png")

    feature_cols = [
        "danceability_%", "energy_%", "acousticness_%",
        "valence_%", "speechiness_%", "released_year",
        "released_month", "in_spotify_playlists"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = []
    for col in feature_cols:
        g1 = df.loc[df["artist_group"] == "Top 10 Artists", col].dropna()
        g2 = df.loc[df["artist_group"] == "Non-Top 10 Artists", col].dropna()
        if len(g1) > 1 and len(g2) > 1:
            t_stat, p_val = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
            results.append({
                "Variable": col,
                "Top10_Mean": g1.mean(),
                "NonTop10_Mean": g2.mean(),
                "T_Statistic": t_stat,
                "P_Value": p_val
            })

    results_df = pd.DataFrame(results).sort_values("P_Value")
    print("\nFeature Comparison Results")
    print(results_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=results_df, x="Variable", y="T_Statistic", ax=axes[0])
    axes[0].axhline(1.96, linestyle="--", color="red")
    axes[0].axhline(-1.96, linestyle="--", color="red")
    axes[0].set_title("T-Statistics by Feature")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=results_df, x="Variable", y="P_Value", ax=axes[1])
    axes[1].axhline(0.05, linestyle="--", color="red")
    axes[1].set_title("P-Values by Feature")
    axes[1].tick_params(axis="x", rotation=45)

    save_plot("feature_ttest_results.png")

    song_features = [
        "bpm", "danceability_%", "valence_%", "energy_%",
        "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%"
    ]
    song_features = [c for c in song_features if c in df.columns]

    corr_df = df[song_features + ["streams"]].corr(numeric_only=True)
    print("\nCorrelation with Streams")
    print(corr_df["streams"].sort_values(ascending=False))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    return results_df


# -----------------------------
# Section 3: Playlist regression
# -----------------------------
def run_playlist_regression(df: pd.DataFrame):
    x_cols = [c for c in ["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists"] if c in df.columns]
    y_col = "streams"

    reg_df = df[x_cols + [y_col]].dropna().copy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        x_scaler.fit_transform(reg_df[x_cols]),
        columns=x_cols,
        index=reg_df.index
    )
    y_scaled = pd.Series(
        y_scaler.fit_transform(reg_df[[y_col]]).flatten(),
        index=reg_df.index,
        name="streams_scaled"
    )

    X_const = sm.add_constant(X_scaled)
    model = sm.OLS(y_scaled, X_const).fit()

    print("\nPlaylist Regression Summary")
    print(model.summary())

    y_pred = model.predict(X_const)

    plt.figure()
    plt.scatter(y_scaled, y_pred, alpha=0.6)
    plt.plot(
        [y_scaled.min(), y_scaled.max()],
        [y_scaled.min(), y_scaled.max()],
        linestyle="--",
        color="red"
    )
    plt.xlabel("Actual Normalized Streams")
    plt.ylabel("Predicted Normalized Streams")
    plt.title("Actual vs Predicted Streams")
    save_plot("actual_vs_predicted_regression.png")

    return model


# -----------------------------
# Section 4: Clustering + seasonality
# -----------------------------
def analyze_seasonality(df: pd.DataFrame):
    cluster_features = [c for c in [
        "in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists", "streams"
    ] if c in df.columns]

    cluster_df = df[cluster_features + ["released_month"]].dropna().copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[cluster_features])

    inertia = []
    k_values = range(1, 11)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertia.append(km.inertia_)

    plt.figure()
    plt.plot(list(k_values), inertia, marker="o")
    plt.title("Elbow Method for KMeans")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    save_plot("kmeans_elbow.png")

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    cluster_df["Cluster"] = clusters

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled)

    plt.figure()
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, alpha=0.7)
    plt.title("KMeans Clusters (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    save_plot("kmeans_pca_clusters.png")

    sil = silhouette_score(scaled, clusters)
    print(f"\nSilhouette Score: {sil:.3f}")

    cluster_summary = cluster_df.groupby("Cluster")[cluster_features].mean()
    print("\nCluster Summary")
    print(cluster_summary)

    cluster_stream_rank = cluster_summary["streams"].sort_values().index.tolist()
    cluster_name_map = {
        cluster_stream_rank[0]: "Less-known",
        cluster_stream_rank[1]: "Normal",
        cluster_stream_rank[2]: "Well-known",
        cluster_stream_rank[3]: "Phenomenal"
    }

    cluster_df["Cluster_Name"] = cluster_df["Cluster"].map(cluster_name_map)
    cluster_df["Season"] = cluster_df["released_month"].apply(get_season)

    freq_table = pd.crosstab(cluster_df["Season"], cluster_df["Cluster_Name"])
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    freq_table = freq_table.reindex(season_order)

    print("\nSeason x Cluster Frequency Table")
    print(freq_table)

    chi2, p_val, dof, expected = chi2_contingency(freq_table)
    print("\nChi-square Test: Season vs Cluster")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_val:.6g}")

    freq_pct = freq_table.div(freq_table.sum(axis=1), axis=0) * 100
    freq_pct.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Percentage Distribution of Clusters Across Seasons")
    plt.xlabel("Season")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot("season_cluster_distribution.png")

    return cluster_df


# -----------------------------
# Main
# -----------------------------
def main():
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)

    top_10, non_top_10 = analyze_artist_tiers(df)
    _ = analyze_audio_features(df, top_10)
    _ = run_playlist_regression(df)
    _ = analyze_seasonality(df)

    print("\nAnalysis complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
