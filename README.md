# Premier League 2023-24 — Player Statistics Analysis

A full data-science pipeline applied to Fantasy Premier League data: exploratory analysis, linear regression with variable selection, PCA, and a player similarity recommender — all on 488 Premier League players from the 2023-24 season.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Analysis Steps](#analysis-steps)
- [Figures](#figures)
- [Getting Started](#getting-started)
- [Usage](#usage)

---

## Overview

| Item | Detail |
|---|---|
| **Season** | Premier League 2023-24 |
| **Players** | 488 (≥ 90 minutes played) |
| **Source** | [Fantasy Premier League — vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) |
| **Response variable** | `total_points` (FPL season points) |
| **Key techniques** | EDA · OLS regression · backward elimination (VIF + p-values) · PCA · cosine/Euclidean similarity |

---

## Project Structure

```
football_project/
├── data/
│   └── players.csv          # Auto-downloaded on first run
├── figures/                 # All generated plots (PNG)
├── notebooks/
│   └── analysis.ipynb       # End-to-end walkthrough notebook
├── src/
│   ├── load_data.py         # Download & clean FPL data
│   ├── eda.py               # EDA: pairplots, heatmap, distributions
│   ├── regression.py        # OLS regression + backward elimination
│   ├── pca.py               # PCA, scree plot, correlation circle, projection
│   └── player_similarity.py # Nearest-neighbour recommender in PCA space
├── report/
│   └── report.tex           # LaTeX report
├── requirements.txt
└── README.md
```

---

## Dataset

Data is fetched automatically from the public FPL GitHub repository the first time you run `load_data.py`. Players with fewer than 90 minutes played are excluded.

**Features used in analysis:**

| Column | Description |
|---|---|
| `goals` / `assists` | Goals scored and assists |
| `xg` / `xa` | Expected goals / expected assists |
| `influence` / `creativity` / `threat` / `ict_index` | FPL ICT Index components |
| `minutes` | Total minutes played |
| `yellow_cards` / `clean_sheets` | Discipline and defensive contribution |
| `total_points` | **Response variable** — total FPL season points |
| `value` | Market value (× 0.1 M£) |
| `position` | GK · DEF · MID · FWD |

---

## Analysis Steps

### 1. Data Loading & Cleaning — `src/load_data.py`
Downloads the raw FPL CSV, selects and renames relevant columns, maps position codes to readable labels, and removes players with missing or insufficient data.

### 2. Exploratory Data Analysis — `src/eda.py`
- **Dataset description**: shape, position breakdown, numeric summary, missing values
- **Grouped pairplots**: variables split into three meaningful groups (Scoring, Performance, Involvement) so each pairplot stays readable
- **Correlation heatmap**: full pairwise Pearson correlations across all 11 features
- **Distribution plots**: histograms + KDE per feature, coloured by position

### 3. Linear Regression — `src/regression.py`
- **Full model**: OLS with all 11 candidate predictors; shows heavy multicollinearity (VIF > 100 for some variables)
- **Backward elimination**: iteratively removes the predictor with the highest VIF (threshold 5) and highest p-value until all remaining predictors are significant and VIF-clean
- **Final model**: 6 predictors — `goals`, `assists`, `influence`, `creativity`, `yellow_cards`, `clean_sheets`
- **Diagnostics**: residual plots, Q-Q plot, scale-location, leverage

### 4. PCA — `src/pca.py`
- Standardises the 6 selected features and fits full PCA
- **Scree plot**: variance explained per component
- **Correlation circle**: biplot of feature loadings on PC1 & PC2
- **Projection plot**: all 488 players in PC1-PC2 space, coloured by position

### 5. Player Similarity — `src/player_similarity.py`
Given 1–3 player names, finds the *k* most statistically similar players using Euclidean distance in the first 3 principal components. Includes fuzzy name matching and a scatter-plot visualisation of the query player(s) and their neighbours.

---

## Figures

| Figure | Description |
|---|---|
| `pairplot_scoring.png` | Pairplot — Goals, xG, Assists, xA |
| `pairplot_performance.png` | Pairplot — ICT Index components |
| `pairplot_involvement.png` | Pairplot — Minutes, Cards, Clean Sheets, Points |
| `correlation_heatmap.png` | Pearson correlation heatmap |
| `distributions.png` | Per-feature histograms + KDE |
| `regression_coefficients.png` | Final model coefficient plot |
| `regression_diagnostics.png` | Residual diagnostics (4-panel) |
| `pca_scree.png` | Scree / explained-variance plot |
| `correlation_circle.png` | PCA biplot (feature loadings) |
| `pca_projection.png` | Player projections on PC1 & PC2 |
| `player_similarity.png` | Similarity recommender scatter plot |

---

## Getting Started

### Prerequisites

Python 3.11+ is recommended.

```bash
# Clone the repo
git clone https://github.com/<your-username>/football_project.git
cd football_project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# 1. Download & cache the data (auto-runs on import, but can be run standalone)
python src/load_data.py

# 2. EDA plots
python src/eda.py

# 3. Regression
python src/regression.py

# 4. PCA
python src/pca.py

# 5. Player similarity
python src/player_similarity.py
```

Or open the notebook for the full walkthrough:

```bash
jupyter lab notebooks/analysis.ipynb
```

---

## Usage

### Find similar players

```python
from src.player_similarity import find_similar_players

# Single query
find_similar_players(["Erling Haaland"], n_neighbors=5)

# Multi-query (finds players similar to all three)
find_similar_players(["Kevin De Bruyne", "Martin Odegaard"], n_neighbors=5)
```

### Load the dataset directly

```python
from src.load_data import load_data

df = load_data()          # loads from cache, downloads if not present
df = load_data(force_download=True)   # forces a fresh download
```

### Run PCA and inspect loadings

```python
from src.load_data import load_data
from src.pca import fit_pca, SELECTED_FEATURES

df = load_data()
pca, X_pca, sub_df = fit_pca(df, SELECTED_FEATURES)
print(pca.explained_variance_ratio_)
```
