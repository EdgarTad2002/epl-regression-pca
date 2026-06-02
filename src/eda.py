"""
eda.py
------
Exploratory Data Analysis for the football player dataset.

Covers project steps 2–4:
  2. Describe the dataset
  3. Pairwise scatter plots (each variable vs response) + correlation heatmap
  4. Assess whether variables follow a Gaussian distribution
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import load_data

# ── output folder ─────────────────────────────────────────────────────────────
_ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR  = os.path.abspath(os.path.join(_ROOT, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# All 11 candidate features (used in full regression model)
ALL_FEATURES = [
    "goals", "assists", "xg", "xa",
    "influence", "creativity", "threat", "ict_index",
    "minutes", "yellow_cards", "clean_sheets",
]

# Alias kept for backward compatibility
FEATURES = ALL_FEATURES

# ── Step 2 – describe dataset ─────────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame) -> None:
    """Print a structured textual description of the dataset."""
    print("=" * 60)
    print("DATASET DESCRIPTION")
    print("=" * 60)
    print(f"  Rows (players)  : {len(df):,}")
    print(f"  Columns         : {df.shape[1]}")
    print(f"  Season          : Premier League 2023-24")
    print(f"  Source          : Fantasy Premier League (public GitHub)")
    print()

    print("── Position breakdown ───────────────────────────────────")
    print(df["position"].value_counts().to_string())
    print()

    print("── Numeric summary (core features) ─────────────────────")
    feats = [f for f in FEATURES if f in df.columns]
    summary = df[feats].describe().T
    summary.columns = ["n", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(summary.round(2).to_string())
    print()

    print("── Missing values ───────────────────────────────────────")
    missing = df[feats].isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values in core features.")
    else:
        print(missing.to_string())
    print("=" * 60)


# ── Step 3 – grouped pairplots ────────────────────────────────────────────────

# Variables split into meaningful groups so each pairplot stays readable
GROUPS = {
    "scoring":     ["goals", "xg", "assists", "xa"],
    "performance": ["influence", "creativity", "threat", "ict_index"],
    "involvement": ["minutes", "yellow_cards", "clean_sheets", "total_points"],
}

PALETTE = {"GK": "#4c72b0", "DEF": "#55a868", "MID": "#c44e52", "FWD": "#dd8452"}


def plot_grouped_pairplots(df: pd.DataFrame) -> list[str]:
    """
    Create one pairplot per variable group (4×4 each — easy to read).
    Each figure is saved as a separate PNG.

    Returns list of saved file paths.
    """
    paths = []
    for group_name, feats in GROUPS.items():
        feats = [f for f in feats if f in df.columns]

        g = sns.pairplot(
            df[feats + ["position"]].dropna(),
            hue="position",
            palette=PALETTE,
            diag_kind="kde",
            plot_kws=dict(alpha=0.5, s=22, linewidth=0),
            diag_kws=dict(linewidth=1.8),
            corner=True,       # lower triangle only — cleaner
        )
        title_map = {
            "scoring":     "Scoring & Expected Goals",
            "performance": "Performance Metrics (Influence / Creativity / Threat)",
            "involvement": "Involvement, Discipline & Points",
        }
        g.figure.suptitle(
            f"Pairplot — {title_map[group_name]}\nPremier League 2023-24",
            y=1.02, fontsize=12, fontweight="bold",
        )

        path = os.path.join(FIG_DIR, f"pairplot_{group_name}.png")
        g.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"Saved → {path}")
        paths.append(path)

    return paths


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Correlation heatmap of all numeric features + response.
    Shows exact correlation coefficients — quick overview of all relationships.
    Saves → figures/correlation_heatmap.png
    """
    cols = [f for f in ALL_FEATURES + ["total_points"] if f in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        square=True, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Correlation Heatmap — Premier League 2023-24 Player Stats",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


def plot_scatterplot_matrix(df: pd.DataFrame) -> list[str]:
    """Kept for backward compatibility — redirects to grouped pairplots."""
    return plot_grouped_pairplots(df)


# ── Step 4 – Gaussian assessment ─────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> str:
    """
    For each core feature, plot:
      left  : histogram + fitted normal curve
      right : Q-Q plot (quantile-quantile vs normal)

    A Shapiro-Wilk p-value is printed on each histogram.
    Saves → figures/distributions.png
    """
    feats = [f for f in FEATURES if f in df.columns]
    ncols = 2
    nrows = len(feats)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, nrows * 2.8),
        constrained_layout=True,
    )
    fig.suptitle(
        "Distribution Assessment — Histogram & Q-Q Plot\n"
        "Premier League 2023-24 Players",
        fontsize=13, fontweight="bold",
    )

    for i, feat in enumerate(feats):
        col = df[feat].dropna()

        # ── histogram + normal fit ──
        ax_hist = axes[i, 0]
        ax_hist.hist(col, bins=30, density=True, color="#4c72b0",
                     alpha=0.6, edgecolor="white", linewidth=0.4)

        # overlay fitted normal
        mu, sigma = col.mean(), col.std()
        x = np.linspace(col.min(), col.max(), 200)
        ax_hist.plot(x, stats.norm.pdf(x, mu, sigma),
                     color="#c44e52", linewidth=2, label="Normal fit")

        # Shapiro-Wilk test (max 5000 samples)
        sample = col.sample(min(len(col), 5000), random_state=42)
        _, p = stats.shapiro(sample)
        normal_flag = "≈ Normal" if p > 0.05 else "NOT Normal"
        ax_hist.set_title(f"{feat}  —  {normal_flag}  (Shapiro p={p:.3f})",
                          fontsize=9)
        ax_hist.set_xlabel(feat, fontsize=8)
        ax_hist.set_ylabel("Density", fontsize=8)
        ax_hist.legend(fontsize=7)

        # ── Q-Q plot ──
        ax_qq = axes[i, 1]
        (osm, osr), (slope, intercept, r) = stats.probplot(col, dist="norm")
        ax_qq.scatter(osm, osr, s=8, alpha=0.5, color="#4c72b0")
        ax_qq.plot(
            [osm[0], osm[-1]],
            [slope * osm[0] + intercept, slope * osm[-1] + intercept],
            color="#c44e52", linewidth=1.8, label=f"R²={r**2:.3f}",
        )
        ax_qq.set_title(f"Q-Q: {feat}", fontsize=9)
        ax_qq.set_xlabel("Theoretical quantiles", fontsize=8)
        ax_qq.set_ylabel("Sample quantiles", fontsize=8)
        ax_qq.legend(fontsize=7)

    path = os.path.join(FIG_DIR, "distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


def print_gaussian_summary(df: pd.DataFrame) -> None:
    """Print a table summarising normality test results for each feature."""
    feats = [f for f in FEATURES if f in df.columns]
    print("\n── Normality Summary (Shapiro-Wilk) ────────────────────")
    print(f"{'Feature':<22} {'Skewness':>10} {'Kurtosis':>10} {'p-value':>10} {'Normal?':>9}")
    print("-" * 65)
    for feat in feats:
        col = df[feat].dropna()
        sample = col.sample(min(len(col), 5000), random_state=42)
        _, p   = stats.shapiro(sample)
        skew   = col.skew()
        kurt   = col.kurtosis()
        flag   = "YES" if p > 0.05 else "no"
        print(f"{feat:<22} {skew:>10.3f} {kurt:>10.3f} {p:>10.4f} {flag:>9}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    # Step 2
    describe_dataset(df)

    # Step 4 summary (text)
    print_gaussian_summary(df)

    # Step 3 — grouped pairplots + correlation heatmap
    print("Plotting grouped pairplots…")
    plot_grouped_pairplots(df)
    print("Plotting correlation heatmap…")
    plot_correlation_heatmap(df)

    # Step 4 — histograms + Q-Q plots
    print("Plotting distributions…")
    plot_distributions(df)

    print("\nAll figures saved to  figures/")
