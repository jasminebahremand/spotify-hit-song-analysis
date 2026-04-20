# Playlist Placement Beats Talent
**Multiple Linear Regression · K-Means Clustering · PCA · 953 Songs**

---

## Overview
The assumption was that audio features — danceability, energy, tempo — drive streaming success. The data said otherwise.

This project analyzes Spotify's top songs of 2023 to identify what actually predicts streaming performance across Spotify, Apple Music, and Deezer. Built a full pipeline from raw CSV to regression, clustering, and hypothesis testing.

> Full write-up available at [portfolio URL]

---

## Key Findings
- **Playlist placement explained 72.7% of variance in streams (R²=0.727)** — distribution is the lever, not the music itself
- **Spotify placements had the largest effect** (β=0.4975) vs Apple Music (β=0.37) and Deezer (β=0.12) — Spotify editorial is the single highest-leverage channel
- **Top artists consistently scored higher on danceability and energy** — but these features showed little to no predictive power across the full dataset (r=−0.10 for danceability)
- **Release season is significantly associated with performance tier** — top-performing songs were more concentrated in winter releases (χ²=27.92, p=.001)

The practical takeaway: getting on the right playlists matters more than what the song sounds like.

---

## Key Visuals

### Spotify Playlists Have the Strongest Impact on Streams
![Playlist Impact](plots/playlist_impact_comparison.png)

### Songs Group into Distinct Performance Tiers by Playlist Exposure
![Clustering](plots/playlist_clustering.png)

### Top Artists Score Higher on Danceability and Energy
![Audio Features](plots/artist_tier_audio_features.png)

### Song Performance Mix Changes Across Release Seasons
![Seasonal Performance](plots/seasonal_performance_distribution.png)

### Model Captures Overall Streaming Trends
![Actual vs Predicted](plots/actual_vs_predicted_streams.png)

### Audio Features Show Weak Relationship with Streaming Performance
![Audio Correlation](plots/audio_feature_correlation.png)

---

## Methods
- Exploratory Data Analysis
- Multiple Linear Regression with standardized coefficients for platform comparison
- T-tests comparing top vs non-top artist audio feature scores
- Chi-square test for release season vs performance tier
- K-Means Clustering (k=4) on playlist presence and stream volume
- PCA for cluster visualization in reduced feature space

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
