# It's Not the Product. It's the Placement.
**Multiple Linear Regression · K-Means Clustering · PCA · 952 Songs**

---

## Overview
The assumption was that audio features — danceability, energy, tempo — drive streaming success. The data said otherwise.

This project analyzes Spotify's top songs of 2023 to identify what actually predicts streaming performance across Spotify, Apple Music, and Deezer. Built a full pipeline from raw CSV to regression, clustering, and hypothesis testing.

> Full write-up available at [portfolio URL]

---

## Key Findings
- **Playlist placement explained 46.4% of variance in streams (R²=0.464)** — distribution is the lever, not the music itself
- **Spotify placements had the largest effect** (β=0.565) vs Apple Music (β=0.294) — Deezer showed a negative relationship (β=−0.171)
- **Audio features do not predict streaming success** — danceability showed a slight negative correlation (r=−0.11) and top artists did not consistently outscore others
- **Release season is significantly associated with performance tier** — winter releases showed the highest concentration of top-performing songs (χ²=55.40, p<.001)

The practical takeaway: getting on the right playlists matters more than what the song sounds like.

---

## Key Visuals

### Most Songs Receive Moderate Streaming Attention
![Streams Distribution](plots/streams_distribution.png)

Even among Spotify's top songs of 2023, most cluster in a moderate performance range — a small number break far ahead.

### Spotify Playlists Have the Strongest Impact on Streams
![Playlist Impact](plots/playlist_impact_comparison.png)

Spotify editorial placement is the single highest-leverage distribution channel — Deezer shows a negative relationship with streams.

### Songs Group into Distinct Performance Tiers by Playlist Exposure
![Clustering](plots/playlist_clustering.png)

K-means clustering separates tracks into four tiers — Low Visibility, Breaking Through, Well Known, and Phenomenal — based on playlist presence and stream volume.

### Four Distinct Song Performance Groups Confirmed
![PCA](plots/pca_clusters.png)

PCA reduces 4 variables to 2 dimensions — 92% of variance explained — confirming the clusters are genuinely distinct and not an artifact of the algorithm.

### Song Performance Mix Changes Across Release Seasons
![Seasonal Performance](plots/seasonal_performance_distribution.png)

Winter releases show the highest share of top-tier songs (χ²=55.40, p<.001) — release timing meaningfully affects a track's chance of breaking through.

### Model Captures Overall Streaming Trends (R²=0.464)
![Actual vs Predicted](plots/actual_vs_predicted_streams.png)

Predicted values track actual streams across most of the range — songs above the line outperformed what their playlist presence would predict.

### Audio Features Show No Consistent Advantage for Top Artists
![Audio Features](plots/artist_tier_audio_features.png)

Danceability was actually slightly higher among non-top artists (67.8 vs 64.6, p=0.004) — energy showed no significant difference. Audio features don't reliably separate top performers.

### Audio Features Have Near-Zero Correlation with Streams
![Stream Correlations](plots/stream_correlations.png)

All audio features show weak to negative correlations with streams — what a song sounds like is not what makes it perform well.

---

## Methods
- Exploratory Data Analysis
- Multiple Linear Regression with standardized coefficients for platform comparison
- T-tests comparing top vs non-top artist audio feature scores
- Chi-square test for release season vs performance tier
- K-Means Clustering (k=4) on playlist presence and stream volume
- PCA for cluster validation in reduced feature space (92% variance explained)

---

## Tech Stack
Python · Pandas · Statsmodels · SciPy · Scikit-learn · Matplotlib · Seaborn

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook spotify-hit-song-analysis.ipynb
```

---

## Data

**Top Spotify Songs 2023:** https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

> Dataset not included in this repo due to size. Download from Kaggle and place the CSV in a folder called `spotify` in your Google Drive.

---

## Files
- `spotify-hit-song-analysis.ipynb` — full analysis notebook
- `requirements.txt` — dependencies
- `plots/` — generated visualizations
