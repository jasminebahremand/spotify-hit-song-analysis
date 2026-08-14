# What Actually Drives Streaming Performance?

**Multiple Linear Regression (log-log) · K-Means Clustering · PCA · 952 Songs**

## Overview

Playlist placement, audio features, release timing — which actually predicts whether a song performs? This project analyzed Spotify's top songs of 2023 to find out, testing variables across Spotify, Apple Music, and Deezer. Built a full pipeline from raw CSV to regression, clustering, and hypothesis testing, with robustness checks throughout.

Full write-up: https://jasminebahremand.my.canva.site/

## Key Findings

- **Playlist placement is the strongest driver of streaming performance**, explaining ~62.5% of variation in streams (log-log regression, R² = 0.625). The effect held after controlling for song age, artist prominence, and collaboration.
- **It's overwhelmingly a Spotify story.** Standardized effects: Spotify β = 0.66 (dominant, significant), Deezer β = 0.11 (small, positive, significant), Apple Music β = 0.05 (not significant). Artist fame appears to work *through* playlist placement rather than on top of it.
- **A song's sound barely predicts performance.** Only speechiness holds up as a reliable signal — robust across multiple cutoffs and even in a combined model with playlists (p = 0.002) — while every other audio feature, including danceability, fails to hold. Adding all audio features to the playlist model raises R² by just 0.005.
- **Release timing has a weak effect** (χ² = 55.40, p < .001, Cramér's V = 0.14). Winter releases over-index the top tier (30.5% vs ~21% in spring), but the association is driven mostly by the *bottom* tier — Fall avoids it while Spring and Summer over-index it.
- **Playlist exposure is highly concentrated.** Songs split into a large low-exposure majority (~93%) and a small high-exposure group (~7%) that streams far more.

**Bottom line:** streaming success is driven far more by *where a song is distributed* than by *what it sounds like*.

## Key Visuals

### Spotify Playlists Show the Strongest Link to Streams
![Spotify Playlists Show the Strongest Link to Streams](plots/playlist_impact_comparison.png)
Standardized effect of each platform's playlists. Spotify dominates; Apple's effect isn't statistically significant and Deezer's is small, so playlist impact is really a Spotify story.

### Winter Releases Are Most Likely to Be Top-Tier Hits
![Winter Releases Are Most Likely to Be Top-Tier Hits](plots/seasonal_performance_distribution.png)
Share of each season's songs that reach the top tier. Winter and Fall lead (30.5% and 26.3%) over Spring and Summer (~21%), though the seasonal effect is modest (Cramér's V = 0.14).

### Songs Split Into a Low-Exposure Majority and a High-Exposure Few
![Songs Split Into a Low-Exposure Majority and a High-Exposure Few](plots/playlist_clustering.png)
K-means (k=2, chosen by silhouette score) groups songs by playlist footprint into a large low-exposure majority (~93%) and a small high-exposure group (~7%) that streams far more — playlist exposure is highly concentrated.

## Methods

- Exploratory data analysis and cleaning (type conversion, missing-value handling, log transforms for right-skewed variables)
- Multiple linear regression (log-log) with robust standard errors (HC3), VIF checks for multicollinearity, and standardized coefficients for fair platform comparison
- Robustness checks: re-ran the model controlling for song age, artist prominence, and collaboration
- Audio features: Welch's t-tests (top-tier vs rest) with a Bonferroni correction and Cohen's d effect sizes; a threshold-sensitivity check across multiple cutoffs; and a combined model testing whether audio adds anything beyond playlist placement
- Chi-square test with Cramér's V and standardized residuals for release season vs performance tier
- K-Means clustering (k=2, chosen via silhouette and elbow diagnostics) with PCA for visualization

## Limitations

- **Correlation, not causation.** Playlists and streams reinforce each other (popular songs get added to more playlists), so these results show association, not proof that playlists drive streams.
- **Selection bias.** The data covers only the top songs of 2023, so findings are conditional on a song already being a hit and may not generalize to all songs.
- **Unmeasured confounders.** Song quality, marketing spend, and social-media virality all plausibly drive both playlist placement and streams, and none are in the dataset.
- **Non-independent observations.** Some artists appear multiple times, so their songs aren't fully independent; clustering standard errors by artist would be the rigorous fix.

## Next Steps

- Test the model on a broader sample that includes non-charting songs, to check whether the playlist–streams pattern holds beyond the top-song bubble.
- Add external signals (artist followers, monthly listeners, marketing spend) to control for fame and promotion directly rather than with a proxy.
- Use longitudinal data (streams and playlist adds over time) to untangle the playlist↔streams feedback loop and move toward causal claims.
- Cluster standard errors by artist to account for repeated artists, and add a train/test split if the goal shifts from explaining drivers to prediction.

## Tech Stack

Python · Pandas · Statsmodels · SciPy · Scikit-learn · Matplotlib · Seaborn

## How to Run

Locally:

pip install -r requirements.txt
jupyter notebook spotify_hit_song_analysis.ipynb
​

Or in Google Colab: open the notebook and run all cells — no setup needed. The notebook loads the dataset straight from this repo, so it works whether you run it locally or in the cloud.

## Data

Top Spotify Songs 2023: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

The CSV (spotify-2023.csv) is included in this repo and loaded automatically by the notebook, so there's no separate download from Kaggle. (Original source: Kaggle, linked above.)

## Files

- `spotify_hit_song_analysis.ipynb` — full analysis notebook
- `spotify-2023.csv` — Spotify 2023 dataset
- `requirements.txt` — dependencies
- `plots/` — generated visualizations
