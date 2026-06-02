"""
player_similarity.py
--------------------
Player Similarity Recommendation System based on PCA coordinates.

Given 1, 2, or 3 player names, finds the most statistically similar
players in the dataset using Euclidean distance in PCA space.

Usage:
    python src/player_similarity.py

    or inside a notebook / script:

        from src.player_similarity import find_similar_players, plot_similarity_projection
        find_similar_players(["Erling Haaland"], n_neighbors=5)
        find_similar_players(["Kevin De Bruyne", "Martin Odegaard"], n_neighbors=5)
"""

import os
import sys
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from difflib import get_close_matches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import load_data
from src.pca import fit_pca, SELECTED_FEATURES, POSITION_COLORS

_ROOT   = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.abspath(os.path.join(_ROOT, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# Number of PCs used for distance computation
N_COMPONENTS = 3


# ── Step 1 – build PCA table ─────────────────────────────────────────────────

def build_player_pca_table(
    df: pd.DataFrame | None = None,
    features: list[str] = SELECTED_FEATURES,
) -> pd.DataFrame:
    """
    Run the PCA pipeline and return a DataFrame with one row per player
    containing their coordinates in PCA space.

    Returns
    -------
    pd.DataFrame with columns: player, position, PC1, PC2, PC3, ...
    """
    if df is None:
        df = load_data()

    pca, X_pca, sub_df = fit_pca(df, features)

    n_pcs = X_pca.shape[1]
    pc_cols = {f"PC{i + 1}": X_pca[:, i] for i in range(n_pcs)}

    table = sub_df[["player", "position"]].copy().reset_index(drop=True)
    for col, vals in pc_cols.items():
        table[col] = vals

    return table


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Lowercase and remove diacritics for fuzzy comparison.
    Handles standard accents (é→e, ü→u) AND Scandinavian letters
    that are not decomposable via NFKD (Ø→o, Å→a, Æ→ae).

    Examples:
        'Ødegaard'  → 'odegaard'
        'Fernández' → 'fernandez'
        'Håvertz'   → 'havertz'
    """
    _MANUAL = str.maketrans({
        "Ø": "o", "ø": "o",
        "Å": "a", "å": "a",
        "Æ": "ae","æ": "ae",
        "Đ": "d", "đ": "d",
        "Ł": "l", "ł": "l",
        "ß": "ss",
    })
    text = text.translate(_MANUAL)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _resolve_names(
    player_names: list[str],
    table: pd.DataFrame,
) -> list[str]:
    """
    Resolve player names with three fallback levels:
      1. Exact case-insensitive match
      2. Unicode-normalised match  (handles Ø → O, é → e, etc.)
      3. Fuzzy match via difflib   (typo tolerance)

    Returns a list of canonical player names from the dataset.
    Raises ValueError with suggestions if a name cannot be resolved.
    """
    all_names      = table["player"].tolist()
    all_lower      = [n.lower() for n in all_names]
    all_normalized = [_normalize(n) for n in all_names]
    resolved       = []

    for name in player_names:
        name_lower = name.lower().strip()
        name_norm  = _normalize(name)

        # 1. exact case-insensitive
        if name_lower in all_lower:
            resolved.append(all_names[all_lower.index(name_lower)])
            continue

        # 2. unicode-normalised exact match
        if name_norm in all_normalized:
            resolved.append(all_names[all_normalized.index(name_norm)])
            continue

        # 3. fuzzy match on normalised names
        close = get_close_matches(name_norm, all_normalized, n=3, cutoff=0.6)
        if close:
            suggestions = [all_names[all_normalized.index(c)] for c in close]
            raise ValueError(
                f"Player '{name}' not found.\n"
                f"  Did you mean: {suggestions}?"
            )
        raise ValueError(
            f"Player '{name}' not found and no close match exists in the dataset."
        )

    return resolved


def _compute_centroid(
    player_names: list[str],
    table: pd.DataFrame,
    n_components: int = N_COMPONENTS,
) -> np.ndarray:
    """Return the average PCA coordinate of the given players."""
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    coords  = table[table["player"].isin(player_names)][pc_cols].values
    return coords.mean(axis=0)


# ── core API ──────────────────────────────────────────────────────────────────

def find_similar_players(
    player_names: list[str],
    n_neighbors: int = 5,
    same_position_only: bool = False,
    df: pd.DataFrame | None = None,
    features: list[str] = SELECTED_FEATURES,
    n_components: int = N_COMPONENTS,
    _table: pd.DataFrame | None = None,   # pass pre-built table to avoid recomputing
) -> pd.DataFrame:
    """
    Find the most statistically similar players using Euclidean distance
    in PCA space.

    Parameters
    ----------
    player_names : list[str]
        1, 2, or 3 player names (case-insensitive).
    n_neighbors : int
        Number of similar players to return (default 5).
    same_position_only : bool
        If True, only return players from the same position group as the
        majority position of the input players.
    df : pd.DataFrame, optional
        Pre-loaded dataset. Loaded automatically if not provided.
    features : list[str]
        PCA features to use (default SELECTED_FEATURES).
    n_components : int
        Number of PCA dimensions to use for distance (default 3).
    _table : pd.DataFrame, optional
        Pre-built PCA table. Built automatically if not provided.

    Returns
    -------
    pd.DataFrame with columns: player, position, distance
        Sorted by distance ascending (most similar first).
    """
    if not 1 <= len(player_names) <= 3:
        raise ValueError("Provide between 1 and 3 player names.")

    table   = _table if _table is not None else build_player_pca_table(df, features)
    names   = _resolve_names(player_names, table)
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]

    centroid = _compute_centroid(names, table, n_components)

    # compute Euclidean distance for every player
    coords    = table[pc_cols].values
    distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))

    result = table[["player", "position"]].copy()
    result["distance"] = distances

    # remove the queried players themselves
    result = result[~result["player"].isin(names)]

    # optional position filter
    if same_position_only:
        positions = table[table["player"].isin(names)]["position"].mode()
        if not positions.empty:
            result = result[result["position"] == positions.iloc[0]]

    result = (result
              .sort_values("distance")
              .head(n_neighbors)
              .reset_index(drop=True))
    result.index += 1   # rank starts at 1

    print_similarity_report(names, centroid, result)
    return result


# ── report ────────────────────────────────────────────────────────────────────

def print_similarity_report(
    player_names: list[str],
    centroid: np.ndarray,
    result: pd.DataFrame,
) -> None:
    """Print a formatted similarity report to stdout."""
    print()
    print("=" * 52)
    print("PLAYER SIMILARITY REPORT")
    print("=" * 52)

    print("\nInput Players:")
    for name in player_names:
        print(f"  - {name}")

    print(f"\nPCA centroid ({len(centroid)} components):")
    for i, val in enumerate(centroid, 1):
        print(f"  PC{i} = {val:+.4f}")

    print(f"\nMost Similar Players (top {len(result)}):\n")
    print(f"  {'Rank':<6} {'Player':<28} {'Position':<10} {'Distance':>8}")
    print("  " + "-" * 56)
    for rank, row in result.iterrows():
        print(f"  {rank:<6} {row['player']:<28} {row['position']:<10} {row['distance']:>8.4f}")
    print()


# ── visualization ─────────────────────────────────────────────────────────────

def plot_similarity_projection(
    player_names: list[str],
    n_neighbors: int = 5,
    same_position_only: bool = False,
    df: pd.DataFrame | None = None,
    features: list[str] = SELECTED_FEATURES,
    n_components: int = N_COMPONENTS,
) -> str:
    """
    PC1 vs PC2 scatterplot showing:
      - all players in light grey
      - queried players in red
      - recommended players in blue
      - centroid as a large black star

    Saves → figures/player_similarity.png

    Returns the saved file path.
    """
    table    = build_player_pca_table(df, features)
    names    = _resolve_names(player_names, table)
    centroid = _compute_centroid(names, table, n_components)
    result   = find_similar_players(
        player_names,
        n_neighbors=n_neighbors,
        same_position_only=same_position_only,
        features=features,
        n_components=n_components,
        _table=table,
    )
    recommended = result["player"].tolist()

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # ── all players (background) ──
    bg = table[~table["player"].isin(names + recommended)]
    ax.scatter(bg["PC1"], bg["PC2"],
               c="lightgrey", s=22, alpha=0.6,
               edgecolors="white", linewidths=0.2,
               zorder=1, label="Other players")

    # ── recommended players (blue) ──
    rec_df = table[table["player"].isin(recommended)]
    ax.scatter(rec_df["PC1"], rec_df["PC2"],
               c="#4c72b0", s=90, alpha=0.9,
               edgecolors="white", linewidths=0.5,
               zorder=3, label="Recommended")
    for _, row in rec_df.iterrows():
        ax.annotate(
            row["player"].split()[-1],
            (row["PC1"], row["PC2"]),
            fontsize=8, color="#4c72b0", fontweight="bold",
            xytext=(5, 5), textcoords="offset points",
        )

    # ── queried players (red) ──
    q_df = table[table["player"].isin(names)]
    ax.scatter(q_df["PC1"], q_df["PC2"],
               c="#c44e52", s=140, alpha=1.0,
               edgecolors="white", linewidths=0.8,
               zorder=4, label="Query player(s)")
    for _, row in q_df.iterrows():
        ax.annotate(
            row["player"].split()[-1],
            (row["PC1"], row["PC2"]),
            fontsize=9.5, color="#c44e52", fontweight="bold",
            xytext=(7, 5), textcoords="offset points",
        )

    # ── centroid (black star) ──
    ax.scatter(centroid[0], centroid[1],
               marker="*", c="black", s=400, zorder=5,
               label="Centroid")

    # ── draw lines from centroid to recommended ──
    for _, row in rec_df.iterrows():
        ax.plot([centroid[0], row["PC1"]], [centroid[1], row["PC2"]],
                color="#4c72b0", linewidth=0.6, alpha=0.4, zorder=2)

    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--")

    n_shown = n_components
    var_explained = ""   # filled if we have the pca object available
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)

    title_names = " + ".join(names)
    ax.set_title(
        f"Player Similarity — {title_names}\n"
        f"Top {n_neighbors} most similar players"
        + (" (same position only)" if same_position_only else ""),
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")

    path = os.path.join(FIG_DIR, "player_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


# ── main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    print("\n" + "─" * 52)
    print("DEMO 1 — Single player")
    print("─" * 52)
    find_similar_players(["Erling Haaland"], n_neighbors=5, df=df)

    print("\n" + "─" * 52)
    print("DEMO 2 — Two players (centroid)")
    print("─" * 52)
    # Note: unicode variants are resolved automatically (Ø → O accepted)
    find_similar_players(["Kevin De Bruyne", "Martin Odegaard"], n_neighbors=5, df=df)

    print("\n" + "─" * 52)
    print("DEMO 3 — Three players")
    print("─" * 52)
    find_similar_players(
        ["Kevin De Bruyne", "Martin Odegaard", "Cole Palmer"],
        n_neighbors=5, df=df,
    )

    print("\n" + "─" * 52)
    print("DEMO 4 — Same position only (FWD)")
    print("─" * 52)
    find_similar_players(
        ["Erling Haaland"],
        n_neighbors=5,
        same_position_only=True,
        df=df,
    )

    print("\n" + "─" * 52)
    print("DEMO 5 — Visualization")
    print("─" * 52)
    plot_similarity_projection(
        ["Kevin De Bruyne", "Martin Odegaard"],
        n_neighbors=5,
        df=df,
    )

    print("\n" + "─" * 52)
    print("DEMO 6 — Error handling (typo)")
    print("─" * 52)
    try:
        find_similar_players(["Erling Haland"], df=df)   # one 'a' missing
    except ValueError as e:
        print(f"  Caught: {e}")

    print("\n" + "─" * 52)
    print("DEMO 7 — Who is statistically similar to Alisson?")
    print("─" * 52)
    find_similar_players(["Alisson Ramses Becker"], n_neighbors=5, df=df)
    plot_similarity_projection(
        ["Alisson Ramses Becker"],
        n_neighbors=5,
        df=df,
    )
