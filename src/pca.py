"""
pca.py
------
Principal Component Analysis for the football player dataset.

Covers project steps 8–10 (+ bonus):
  8.  Run PCA, interpret components
  9.  Plot the correlation circle (biplot of loadings)
  10. Plot data projection on the first 2 principal directions
  Bonus: colour projection by player position
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import load_data

_ROOT   = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.abspath(os.path.join(_ROOT, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# Variables selected after backward elimination in regression.py
SELECTED_FEATURES = [
    "goals", "assists", "influence",
    "creativity", "yellow_cards", "clean_sheets",
]

POSITION_COLORS = {
    "GK":  "#4c72b0",
    "DEF": "#55a868",
    "MID": "#c44e52",
    "FWD": "#dd8452",
}


# ── Step 8 – fit PCA ──────────────────────────────────────────────────────────

def fit_pca(df: pd.DataFrame,
            features: list[str] = SELECTED_FEATURES
            ) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    """
    Standardise the data and fit PCA.

    Returns
    -------
    pca     : fitted sklearn PCA object
    X_pca   : projected data (n_players × n_components)
    sub_df  : cleaned sub-DataFrame aligned with X_pca
    """
    sub    = df[features + ["position", "player"]].dropna().reset_index(drop=True)
    X      = sub[features].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca   = PCA()
    X_pca = pca.fit_transform(X_sc)

    print("=" * 60)
    print("STEP 8 — PCA RESULTS")
    print("=" * 60)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    print(f"\n  {'PC':<6} {'Eigenvalue':>12} {'Var %':>8} {'Cumulative %':>14}")
    print("  " + "-" * 44)
    for i, (ev, vr, cv) in enumerate(
        zip(pca.explained_variance_, exp_var, cum_var), 1
    ):
        print(f"  PC{i:<4} {ev:>12.4f} {vr*100:>7.2f}% {cv*100:>13.2f}%")

    print("\n── Loadings (PC1 & PC2) ─────────────────────────────")
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=features,
        columns=["PC1", "PC2"],
    )
    print(loadings.round(4).to_string())
    print()

    return pca, X_pca, sub


# ── Step 8 – scree plot ───────────────────────────────────────────────────────

def plot_scree(pca: PCA) -> str:
    """Bar chart of explained variance per component + cumulative line."""
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    n       = len(exp_var)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(range(1, n + 1), exp_var, color="#4c72b0", alpha=0.8,
           label="Individual variance %")
    ax.plot(range(1, n + 1), cum_var, "o-", color="#c44e52",
            linewidth=2, markersize=6, label="Cumulative variance %")
    ax.axhline(80, color="grey", linewidth=0.8, linestyle="--", label="80 % threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot — PCA Explained Variance", fontweight="bold")
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([f"PC{i}" for i in range(1, n + 1)])
    ax.legend(fontsize=8)

    path = os.path.join(FIG_DIR, "pca_scree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


# ── Step 9 – correlation circle ───────────────────────────────────────────────

def plot_correlation_circle(pca: PCA, features: list[str]) -> str:
    """
    Classic correlation circle: arrows show how each original variable
    projects onto PC1 (x-axis) and PC2 (y-axis).
    Saves → figures/correlation_circle.png
    """
    loadings = pca.components_[:2].T   # shape (n_features, 2)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    # unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="grey",
            linewidth=0.8, linestyle="--", alpha=0.6)

    # arrows + labels
    for i, feat in enumerate(features):
        x, y = loadings[i, 0], loadings[i, 1]
        ax.annotate(
            "",
            xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#4c72b0",
                            lw=2, mutation_scale=18),
        )
        offset_x = 0.06 * np.sign(x)
        offset_y = 0.06 * np.sign(y)
        ax.text(x + offset_x, y + offset_y, feat,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#4c72b0")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1  ({var1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var2:.1f}% variance)", fontsize=12)
    ax.set_title("Correlation Circle — PCA Loadings\n"
                 "Premier League 2023-24 Player Stats",
                 fontsize=13, fontweight="bold")

    path = os.path.join(FIG_DIR, "correlation_circle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


# ── Step 10 + Bonus – data projection ────────────────────────────────────────

def plot_pca_projection(X_pca: np.ndarray, sub_df: pd.DataFrame, pca: PCA) -> str:
    """
    Scatter plot of all players on PC1 vs PC2, coloured by position.
    BONUS: position legend clearly shows how players separate by role.
    Saves → figures/pca_projection.png
    """
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for pos, color in POSITION_COLORS.items():
        mask = sub_df["position"] == pos
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, label=pos, alpha=0.6, s=35, edgecolors="white", linewidths=0.3,
        )

    # Annotate a few standout players
    top_idx = np.argsort(X_pca[:, 0])[-5:]   # top 5 on PC1
    for i in top_idx:
        ax.annotate(
            sub_df["player"].iloc[i],
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=6.5, alpha=0.85,
            xytext=(4, 4), textcoords="offset points",
        )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"PC1  ({var1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var2:.1f}% variance)", fontsize=12)
    ax.set_title("PCA Projection — Players on First Two Principal Components\n"
                 "Coloured by Position (Bonus)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Position", fontsize=10, title_fontsize=10)

    path = os.path.join(FIG_DIR, "pca_projection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def run_pca(df: pd.DataFrame | None = None,
            features: list[str] = SELECTED_FEATURES):
    """Full PCA pipeline. Returns (pca, X_pca, sub_df)."""
    if df is None:
        df = load_data()

    pca, X_pca, sub_df = fit_pca(df, features)
    plot_scree(pca)
    plot_correlation_circle(pca, features)
    plot_pca_projection(X_pca, sub_df, pca)

    print("All PCA figures saved to  figures/")
    return pca, X_pca, sub_df


if __name__ == "__main__":
    run_pca()
