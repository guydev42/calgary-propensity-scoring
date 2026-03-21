"""Propensity scoring pipeline with calibrated models and decile analysis."""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, brier_score_loss, log_loss,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_models(seed=42):
    """Return dict of (name, model) pairs."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=seed, class_weight="balanced", n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=7, random_state=seed, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
        )
    return models


def train_and_calibrate(models, X_train, y_train, seed=42):
    """Train each model and wrap with CalibratedClassifierCV.

    Returns dict of {name: calibrated_model}.
    """
    calibrated = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
        cal.fit(X_train, y_train)
        calibrated[name] = cal
        print(f"    Done.")
    return calibrated


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):
    """Compute key metrics for a single model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"\n--- {name} ---")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Avg precision:     {ap:.4f}")
    print(f"  Brier score:       {brier:.4f}")
    print(f"  Log loss:          {ll:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    return {"name": name, "auc": auc, "avg_precision": ap, "brier": brier,
            "log_loss": ll, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# Decile analysis  (THE key deliverable for marketing)
# ---------------------------------------------------------------------------

def decile_analysis(y_test, y_prob, n_deciles=10):
    """Split customers into deciles by predicted probability.

    Returns a DataFrame with columns:
        decile, n_customers, n_responders, response_rate,
        cumulative_responders, cumulative_response_pct,
        expected_revenue
    """
    tmp = pd.DataFrame({"y_true": y_test, "prob": y_prob})
    tmp["decile"] = pd.qcut(tmp["prob"], n_deciles, labels=False, duplicates="drop")
    # Decile 9 = highest probability
    tmp["decile"] = n_deciles - 1 - tmp["decile"]  # so decile 0 = top

    agg = (
        tmp.groupby("decile")
        .agg(
            n_customers=("y_true", "count"),
            n_responders=("y_true", "sum"),
            avg_prob=("prob", "mean"),
        )
        .reset_index()
    )
    agg["response_rate"] = (agg["n_responders"] / agg["n_customers"]).round(4)
    agg["cumulative_responders"] = agg["n_responders"].cumsum()
    total_responders = agg["n_responders"].sum()
    agg["cumulative_response_pct"] = (
        agg["cumulative_responders"] / total_responders
    ).round(4)

    # Expected revenue: n_customers * response_rate * $15/month upsell
    upsell_value = 15.0
    agg["expected_revenue"] = (
        agg["n_customers"] * agg["response_rate"] * upsell_value
    ).round(2)

    return agg


def print_decile_table(decile_df):
    """Pretty-print the decile analysis table."""
    print("\nDecile analysis (decile 0 = highest propensity):")
    print("-" * 90)
    print(f"{'Decile':>6}  {'Customers':>9}  {'Responders':>10}  "
          f"{'Rate':>6}  {'Cum Resp%':>9}  {'Avg Prob':>8}  {'Exp Rev ($)':>11}")
    print("-" * 90)
    for _, row in decile_df.iterrows():
        print(f"{int(row['decile']):>6}  {int(row['n_customers']):>9}  "
              f"{int(row['n_responders']):>10}  {row['response_rate']:>6.3f}  "
              f"{row['cumulative_response_pct']:>9.3f}  {row['avg_prob']:>8.4f}  "
              f"{row['expected_revenue']:>11.2f}")
    print("-" * 90)


# ---------------------------------------------------------------------------
# Campaign ROI
# ---------------------------------------------------------------------------

def campaign_roi(decile_df, cost_per_contact=2.0, upsell_monthly=15.0, months=12):
    """Compare targeted (top 3 deciles) vs mass campaign."""
    total_customers = decile_df["n_customers"].sum()
    total_responders = decile_df["n_responders"].sum()
    overall_rate = total_responders / total_customers

    # Mass campaign
    mass_cost = total_customers * cost_per_contact
    mass_revenue = total_responders * upsell_monthly * months
    mass_roi = (mass_revenue - mass_cost) / mass_cost * 100

    # Targeted: top 3 deciles (decile 0, 1, 2)
    top3 = decile_df[decile_df["decile"] <= 2]
    tgt_customers = top3["n_customers"].sum()
    tgt_responders = top3["n_responders"].sum()
    tgt_rate = tgt_responders / tgt_customers
    tgt_cost = tgt_customers * cost_per_contact
    tgt_revenue = tgt_responders * upsell_monthly * months
    tgt_roi = (tgt_revenue - tgt_cost) / tgt_cost * 100

    cost_savings = mass_cost - tgt_cost
    conversion_lift = (tgt_rate / overall_rate - 1) * 100

    results = {
        "mass_customers": total_customers,
        "mass_responders": total_responders,
        "mass_rate": round(overall_rate, 4),
        "mass_cost": round(mass_cost, 2),
        "mass_revenue": round(mass_revenue, 2),
        "mass_roi_pct": round(mass_roi, 1),
        "targeted_customers": tgt_customers,
        "targeted_responders": tgt_responders,
        "targeted_rate": round(tgt_rate, 4),
        "targeted_cost": round(tgt_cost, 2),
        "targeted_revenue": round(tgt_revenue, 2),
        "targeted_roi_pct": round(tgt_roi, 1),
        "cost_savings": round(cost_savings, 2),
        "conversion_lift_pct": round(conversion_lift, 1),
    }

    print("\nCampaign ROI comparison:")
    print("=" * 55)
    print(f"  Mass campaign:")
    print(f"    Contacts:     {results['mass_customers']:,}")
    print(f"    Responders:   {results['mass_responders']:,}")
    print(f"    Response rate: {results['mass_rate']:.2%}")
    print(f"    Cost:         ${results['mass_cost']:,.2f}")
    print(f"    Revenue:      ${results['mass_revenue']:,.2f}")
    print(f"    ROI:          {results['mass_roi_pct']:.1f}%")
    print()
    print(f"  Targeted (top 3 deciles):")
    print(f"    Contacts:     {results['targeted_customers']:,}")
    print(f"    Responders:   {results['targeted_responders']:,}")
    print(f"    Response rate: {results['targeted_rate']:.2%}")
    print(f"    Cost:         ${results['targeted_cost']:,.2f}")
    print(f"    Revenue:      ${results['targeted_revenue']:,.2f}")
    print(f"    ROI:          {results['targeted_roi_pct']:.1f}%")
    print()
    print(f"  Cost savings:      ${results['cost_savings']:,.2f}")
    print(f"  Conversion lift:   {results['conversion_lift_pct']:.1f}%")
    print("=" * 55)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_lift_chart(decile_df, save_path=None):
    """Cumulative lift chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = np.arange(1, len(decile_df) + 1) / len(decile_df) * 100
    cum_resp = decile_df["cumulative_response_pct"].values * 100

    ax.plot(pcts, cum_resp, "b-o", linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("Percentage of customers contacted")
    ax.set_ylabel("Cumulative percentage of responders captured")
    ax.set_title("Lift chart")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_decile_response(decile_df, save_path=None):
    """Bar chart of response rate by decile."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if d <= 2 else "#90CAF9" for d in decile_df["decile"]]
    ax.bar(decile_df["decile"].astype(int), decile_df["response_rate"] * 100, color=colors)
    ax.set_xlabel("Decile (0 = highest propensity)")
    ax.set_ylabel("Response rate (%)")
    ax.set_title("Response rate by propensity decile")
    ax.set_xticks(range(len(decile_df)))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_calibration(models_dict, X_test, y_test, save_path=None):
    """Calibration curve for each model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_predicted = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_predicted, fraction_pos, "s-", label=name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """Feature importance from the best model's base estimator."""
    # Try to get feature importances from the underlying estimator
    importances = None
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
        elif hasattr(base, "coef_"):
            importances = np.abs(base.coef_[0])

    if importances is None:
        return None

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], color="#2196F3")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    """Execute the full propensity scoring pipeline."""
    # Add project root to path
    sys.path.insert(0, PROJECT_DIR)
    from src.data_loader import prepare_full_pipeline

    print("=" * 60)
    print("  Customer propensity / upsell scoring pipeline")
    print("=" * 60)

    # Load and prepare data
    print("\n[1/5] Loading and engineering features...")
    data = prepare_full_pipeline()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Response rate: {y_train.mean():.3f} (train), {y_test.mean():.3f} (test)")

    # Build and train models
    print("\n[2/5] Training models with calibration...")
    raw_models = build_models()
    calibrated_models = train_and_calibrate(raw_models, X_train, y_train)

    # Evaluate
    print("\n[3/5] Evaluating models...")
    results = {}
    best_name = None
    best_auc = 0
    for name, model in calibrated_models.items():
        r = evaluate_model(name, model, X_test, y_test)
        results[name] = r
        if r["auc"] > best_auc:
            best_auc = r["auc"]
            best_name = name

    print(f"\n  Best model: {best_name} (AUC = {best_auc:.4f})")

    # Decile analysis with best model
    print("\n[4/5] Decile analysis and campaign ROI...")
    best_prob = results[best_name]["y_prob"]
    decile_df = decile_analysis(y_test, best_prob)
    print_decile_table(decile_df)
    roi = campaign_roi(decile_df)

    # Plots
    print("\n[5/5] Generating plots...")
    plot_lift_chart(decile_df, os.path.join(FIGURES_DIR, "lift_chart.png"))
    plot_decile_response(decile_df, os.path.join(FIGURES_DIR, "decile_response.png"))
    plot_calibration(calibrated_models, X_test, y_test,
                     os.path.join(FIGURES_DIR, "calibration_curves.png"))
    plot_feature_importance(
        calibrated_models[best_name], feature_names,
        save_path=os.path.join(FIGURES_DIR, "feature_importance.png")
    )
    print("  Plots saved to figures/")

    # Save artifacts
    joblib.dump(calibrated_models[best_name], os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(data["scaler"], os.path.join(MODELS_DIR, "scaler.pkl"))
    decile_df.to_csv(os.path.join(MODELS_DIR, "decile_analysis.csv"), index=False)
    print(f"  Model and scaler saved to models/")

    print("\n  Pipeline complete.")
    return {
        "calibrated_models": calibrated_models,
        "results": results,
        "decile_df": decile_df,
        "roi": roi,
        "best_name": best_name,
        "data": data,
    }


if __name__ == "__main__":
    run_pipeline()
