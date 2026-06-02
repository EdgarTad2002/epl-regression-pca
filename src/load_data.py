"""
load_data.py
------------
Downloads Premier League 2023-24 player statistics from the publicly hosted
Fantasy Premier League dataset (GitHub) and saves them to data/players.csv.
If the file already exists it is loaded directly.

Source:
    https://github.com/vaastav/Fantasy-Premier-League

Key numeric columns kept for the project:
    minutes, goals_scored, assists, clean_sheets,
    yellow_cards, red_cards, bonus,
    expected_goals (xG), expected_assists (xA),
    expected_goal_involvements,
    influence, creativity, threat, ict_index,
    total_points, value (market value × 0.1 M£)
"""

import os

import numpy as np
import pandas as pd
import requests

# ── paths ──────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, "..")
DATA_DIR  = os.path.abspath(os.path.join(_ROOT_DIR, "data"))
DATA_PATH = os.path.join(DATA_DIR, "players.csv")

# Public GitHub raw CSV — no authentication required
DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
    "/master/data/2023-24/players_raw.csv"
)

# Columns to keep (metadata + numeric features)
META_COLS = ["first_name", "second_name", "team", "element_type"]

NUMERIC_COLS = {
    "minutes":                    "minutes",
    "goals_scored":               "goals",
    "assists":                    "assists",
    "clean_sheets":               "clean_sheets",
    "yellow_cards":               "yellow_cards",
    "red_cards":                  "red_cards",
    "bonus":                      "bonus",
    "expected_goals":             "xg",
    "expected_assists":           "xa",
    "expected_goal_involvements": "xgi",
    "influence":                  "influence",
    "creativity":                 "creativity",
    "threat":                     "threat",
    "ict_index":                  "ict_index",
    "total_points":               "total_points",
    "value_season":               "value",
}

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _download() -> pd.DataFrame:
    """Fetch the FPL 2023-24 raw player CSV from GitHub."""
    print("Downloading Premier League 2023-24 player stats from GitHub…")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    print(f"  Raw shape: {df.shape}")
    return df


def _select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, and clean the raw FPL DataFrame."""
    # Keep only columns we need
    available_meta    = [c for c in META_COLS    if c in df.columns]
    available_numeric = {k: v for k, v in NUMERIC_COLS.items() if k in df.columns}

    df = df[available_meta + list(available_numeric.keys())].copy()
    df = df.rename(columns=available_numeric)

    # Convert numeric columns
    for col in available_numeric.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Readable player name and position
    df["player"] = df["first_name"] + " " + df["second_name"]
    df["position"] = df["element_type"].map(POSITION_MAP)
    df = df.drop(columns=["first_name", "second_name", "element_type"])

    # Keep players with at least 90 minutes played (meaningful sample)
    df = df[df["minutes"] >= 90].copy()

    # Drop rows missing more than 3 numeric features
    numeric_cols = list(available_numeric.values())
    df = df.dropna(subset=numeric_cols, thresh=len(numeric_cols) - 3)
    df = df.drop_duplicates(subset=["player"])
    df = df.reset_index(drop=True)

    # Reorder: player name first
    front = ["player", "position", "team"]
    rest  = [c for c in df.columns if c not in front]
    df    = df[front + rest]

    return df


# ── public API ───────────────────────────────────────────────────────────────

def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Return the football player dataset as a DataFrame.

    Parameters
    ----------
    force_download : bool
        If True, always re-download even if data/players.csv exists.

    Returns
    -------
    pd.DataFrame
        One row per player, with numeric and categorical columns.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH) and not force_download:
        print(f"Loading existing data from  {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"  {len(df):,} players  ×  {df.shape[1]} columns")
        return df

    raw = _download()
    df  = _select_and_clean(raw)

    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} players → {DATA_PATH}")
    print(f"Columns: {list(df.columns)}")
    return df


# ── quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print("\n── Head ──────────────────────────────────────")
    print(df[["player", "position", "goals", "assists", "xg", "ict_index", "total_points"]].head(10))
    print("\n── Numeric summary ───────────────────────────")
    numeric = df.select_dtypes(include="number")
    print(numeric.describe().round(2))
    print(f"\nPositions:\n{df['position'].value_counts()}")
