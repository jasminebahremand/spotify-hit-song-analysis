# Predicting the Next Hit
Music Streaming · Content Performance Analysis

## Overview
Analyzed Spotify’s top songs of 2023 to identify factors associated with streaming success and inform data-driven music investment decisions.

## Methods
- Exploratory Data Analysis
- Hypothesis Testing (t-tests, chi-square)
- Multiple Linear Regression
- K-Means Clustering
- PCA

## Key Findings
- Playlist placement was the strongest predictor of streams
- Audio features did not significantly predict performance across songs
- Release timing was associated with performance differences

## Tech Stack
Python · Pandas · Statsmodels · SciPy · Scikit-learn · Matplotlib · Seaborn

## Files
- spotify_hit_analysis.py — main analysis script
- requirements.txt — project dependencies

## Plots
- streams_distribution.png
- playlist_vs_streams.png
- audio_features_correlation.png
- cluster_segments.png
- season_vs_performance.png

## How to Run
pip install -r requirements.txt  
python spotify_hit_analysis.py
