"""
regression.py
-------------
Linear regression analysis for the football player dataset.

Covers project steps 5–7:
  5. Choose and justify the response variable
  6. Fit linear regression, interpret coefficients
  7. Variable selection using backward elimination (VIF + p-values)
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import load_data

_ROOT   = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.abspath(os.path.join(_ROOT, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# ── variable sets ─────────────────────────────────────────────────────────────

# All candidate predictors (full model — shows multicollinearity)
ALL_FEATURES = [
    "goals", "assists", "xg", "xa",
    "influence", "creativity", "threat", "ict_index",
    "minutes", "yellow_cards", "clean_sheets",
]

RESPONSE = "total_points"   # Step 5: response variable


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (with constant) and y, dropping rows with any NaN."""
    cols = features + [RESPONSE]
    sub  = df[cols].dropna()
    X    = sub[features]
    y    = sub[RESPONSE]
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of VIF scores for each predictor."""
    Xc  = sm.add_constant(X)
    vif = pd.DataFrame({
        "feature": X.columns,
        "VIF":     [variance_inflation_factor(Xc.values, i + 1)
                    for i in range(X.shape[1])],
    })
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


# ── Step 6 – full regression model ───────────────────────────────────────────

def fit_full_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS with all candidate predictors.
    Returns the fitted model and prints a summary.
    """
    X, y = _prepare(df, ALL_FEATURES)
    model = sm.OLS(y, sm.add_constant(X)).fit()
    print("=" * 62)
    print("STEP 6 — FULL LINEAR REGRESSION MODEL")
    print("  Response : total_points")
    print(f"  Predictors: {ALL_FEATURES}")
    print("=" * 62)
    print(model.summary())

    print("\n── VIF (full model) ─────────────────────────────────────")
    vif = compute_vif(X)
    print(vif.to_string(index=False))
    print()
    return model


# ── Step 7 – variable selection (backward elimination) ───────────────────────

def variable_selection(df: pd.DataFrame,
                        vif_thresh: float = 10.0,
                        p_thresh:   float = 0.05
                        ) -> tuple[list[str], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Backward elimination:
      1. Remove the predictor with the highest VIF (> vif_thresh) first.
      2. Then remove the predictor with the largest p-value (> p_thresh).
      3. Repeat until both conditions are satisfied.

    Returns (selected_features, final_model).
    """
    features = ALL_FEATURES.copy()
    step = 0

    print("=" * 62)
    print("STEP 7 — BACKWARD VARIABLE SELECTION")
    print("=" * 62)

    while True:
        X, y = _prepare(df, features)
        model = sm.OLS(y, sm.add_constant(X)).fit()
        vif   = compute_vif(X)

        # Check VIF
        worst_vif = vif.iloc[0]
        if worst_vif["VIF"] > vif_thresh:
            drop = worst_vif["feature"]
            step += 1
            print(f"  Step {step}: Drop '{drop}'  "
                  f"(VIF = {worst_vif['VIF']:.1f} > {vif_thresh})")
            features.remove(drop)
            continue

        # Check p-values (skip constant)
        pvals = model.pvalues.drop("const", errors="ignore")
        worst_p_feat = pvals.idxmax()
        worst_p_val  = pvals.max()
        if worst_p_val > p_thresh:
            step += 1
            print(f"  Step {step}: Drop '{worst_p_feat}'  "
                  f"(p = {worst_p_val:.4f} > {p_thresh})")
            features.remove(worst_p_feat)
            continue

        break   # nothing left to remove

    print(f"\n  Final features ({len(features)}): {features}")
    print()
    return features, model


# ── Final model summary & diagnostics ────────────────────────────────────────

def fit_final_model(df: pd.DataFrame,
                    features: list[str]
                    ) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit and summarise the final selected model."""
    X, y  = _prepare(df, features)
    model = sm.OLS(y, sm.add_constant(X)).fit()

    print("=" * 62)
    print("FINAL REGRESSION MODEL")
    print("=" * 62)
    print(model.summary())

    print("\n── VIF (final model) ────────────────────────────────────")
    vif = compute_vif(X)
    print(vif.to_string(index=False))
    print()
    return model


def plot_regression_diagnostics(model, df: pd.DataFrame, features: list[str]) -> str:
    """
    Four-panel diagnostic plot:
      1. Fitted vs Residuals
      2. Scale-Location (√|residuals| vs fitted)
      3. Q-Q of residuals
      4. Actual vs Predicted
    Saves → figures/regression_diagnostics.png
    """
    X, y = _prepare(df, features)
    fitted   = model.fittedvalues
    residuals = model.resid
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("Regression Diagnostics — Final Model", fontsize=14, fontweight="bold")

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.45, s=18, color="#4c72b0")
    ax.axhline(0, color="#c44e52", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. Scale-Location
    ax = axes[0, 1]
    ax.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.45, s=18, color="#55a868")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("√|Standardised residuals|")
    ax.set_title("Scale-Location")

    # 3. Q-Q of residuals
    from scipy import stats
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.45, s=18, color="#dd8452")
    ax.plot([osm[0], osm[-1]],
            [slope * osm[0] + intercept, slope * osm[-1] + intercept],
            color="#c44e52", linewidth=1.8)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q Plot of Residuals")

    # 4. Actual vs Predicted
    ax = axes[1, 1]
    ax.scatter(y, fitted, alpha=0.45, s=18, color="#9467bd")
    lims = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.plot(lims, lims, color="#c44e52", linewidth=1.8, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual total_points"); ax.set_ylabel("Predicted total_points")
    ax.set_title(f"Actual vs Predicted  (R² = {model.rsquared:.3f})")
    ax.legend(fontsize=8)

    path = os.path.join(FIG_DIR, "regression_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


def plot_coefficients(model) -> str:
    """Bar chart of standardised coefficients with 95 % CI.
    Saves → figures/regression_coefficients.png
    """
    coef  = model.params.drop("const")
    ci    = model.conf_int().drop("const")
    err_lo = coef - ci[0]
    err_hi = ci[1] - coef
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in coef]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(coef) + 1.5), constrained_layout=True)
    ax.barh(coef.index, coef.values, xerr=[err_lo, err_hi],
            color=colors, alpha=0.8, capsize=4, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("Regression Coefficients with 95 % Confidence Intervals",
                 fontweight="bold")

    path = os.path.join(FIG_DIR, "regression_coefficients.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved → {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def run_regression(df: pd.DataFrame | None = None):
    """
    Full regression pipeline.
    Returns (selected_features, final_model) for use in the notebook.
    """
    if df is None:
        df = load_data()

    # Step 6 — full model (shows multicollinearity)
    fit_full_model(df)

    # Step 7 — backward elimination
    selected_features, _ = variable_selection(df)

    # Final clean model
    final_model = fit_final_model(df, selected_features)

    # Diagnostic plots
    plot_regression_diagnostics(final_model, df, selected_features)
    plot_coefficients(final_model)

    return selected_features, final_model


if __name__ == "__main__":
    selected, model = run_regression()
    print(f"\nSelected features for PCA: {selected}")
