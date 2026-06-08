"""
step7_uncertainty.py
====================
Conformal prediction uncertainty quantification for drug response predictions.

Instead of just predicting "IC50 = 2.3", we output:
    "IC50 = 2.3 ± 0.4  (90% prediction interval)"

Why conformal prediction instead of dropout/ensembles?
  - Distribution-free: no assumptions about the data distribution
  - Rigorous coverage guarantee: 90% intervals ACTUALLY contain the true
    value 90% of the time on test data (provable, not just hoped for)
  - Uses the validation set for calibration — no extra training needed

How it works (simple version):
  1. Run trained GAT on val set → get residuals (errors)
  2. Compute the 90th percentile of |residuals| → this is your "quantile"
  3. For any new prediction: interval = [pred - quantile, pred + quantile]
  4. Evaluate: what % of test targets fall inside their interval?
     That % should be ≥ 90% if calibration worked.

Run:
    python src/step7_uncertainty.py

Output:
    results/uncertainty/
        {drug}_intervals.csv     — predictions + intervals for test set
        {drug}_calibration.png   — calibration plot
        coverage_summary.csv     — coverage % per drug
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step4_dataset import DrugResponseDataset, TARGET_DRUGS
from step5_model import DrugResponseGAT

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results/uncertainty")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COVERAGE_TARGET = 0.90   # we want 90% prediction intervals


# ── Load trained GAT ─────────────────────────────────────────────────────────

def load_gat(drug_name: str, n_features: int) -> DrugResponseGAT:
    model = DrugResponseGAT(
        in_channels = n_features,
        hidden_dim  = 64,
        heads_1     = 4,
        heads_2     = 2,
        dropout     = 0.0,   # no dropout at inference
    ).to(DEVICE)
    ckpt = MODELS_DIR / f"gat_{drug_name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt}. Run step6_train.py first.")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


# ── Get predictions ──────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predictions, targets) as numpy arrays."""
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch).squeeze().cpu().numpy()
        tgt  = batch.y.squeeze().cpu().numpy()
        # handle single-sample batches
        preds.append(np.atleast_1d(pred))
        targets.append(np.atleast_1d(tgt))
    return np.concatenate(preds), np.concatenate(targets)


# ── Conformal calibration ────────────────────────────────────────────────────

def calibrate_conformal(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
    coverage: float = COVERAGE_TARGET,
) -> float:
    """
    Compute the conformal quantile from validation residuals.

    The conformal quantile q is defined as the ceil((n+1)*(1-alpha))/n
    quantile of the absolute residuals on the calibration (val) set.

    This gives a coverage guarantee: P(y_test in [pred-q, pred+q]) >= 1-alpha
    on exchangeable data.

    Args:
        val_preds   : model predictions on validation set
        val_targets : true values on validation set
        coverage    : desired coverage (e.g. 0.90 for 90% intervals)

    Returns:
        q : the conformal quantile (half-width of prediction interval)
    """
    residuals = np.abs(val_preds - val_targets)
    n = len(residuals)
    # Finite-sample corrected quantile level
    alpha = 1 - coverage
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = np.quantile(residuals, quantile_level)
    return float(q)


# ── Evaluate coverage ────────────────────────────────────────────────────────

def evaluate_coverage(
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    q: float,
) -> dict:
    """
    Check what fraction of test targets fall inside [pred-q, pred+q].
    Should be >= COVERAGE_TARGET if calibration worked.
    """
    lower = test_preds - q
    upper = test_preds + q
    covered = ((test_targets >= lower) & (test_targets <= upper))
    coverage = covered.mean()
    avg_width = 2 * q   # interval width is symmetric

    return {
        "coverage":       float(coverage),
        "target_coverage": COVERAGE_TARGET,
        "interval_width": float(avg_width),
        "conformal_q":    float(q),
        "n_test":         len(test_targets),
        "n_covered":      int(covered.sum()),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_intervals(
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    q: float,
    drug_name: str,
    coverage_result: dict,
):
    """
    Two-panel plot:
    Left:  sorted predictions with uncertainty bands
    Right: calibration — residual CDF vs uniform
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: prediction intervals ────────────────────────────────────
    sort_idx = np.argsort(test_targets)
    sorted_targets = test_targets[sort_idx]
    sorted_preds   = test_preds[sort_idx]
    lower = sorted_preds - q
    upper = sorted_preds + q
    covered = (sorted_targets >= lower) & (sorted_targets <= upper)

    x = np.arange(len(sorted_targets))
    axes[0].fill_between(x, lower, upper, alpha=0.25, color="#7F77DD", label="90% interval")
    axes[0].plot(x, sorted_preds,   color="#7F77DD", linewidth=1.5, label="Predicted", zorder=3)
    axes[0].scatter(x[covered],  sorted_targets[covered],
                    color="#1D9E75", s=30, zorder=4, label=f"Covered ({covered.sum()})")
    axes[0].scatter(x[~covered], sorted_targets[~covered],
                    color="#D85A30", s=40, marker="x", zorder=5,
                    linewidths=1.5, label=f"Missed ({(~covered).sum()})")

    actual_cov = coverage_result["coverage"]
    axes[0].set_xlabel("Test samples (sorted by true IC50)")
    axes[0].set_ylabel("LN(IC50)")
    axes[0].set_title(
        f"{drug_name} — Prediction Intervals\n"
        f"Coverage: {actual_cov:.1%} (target: {COVERAGE_TARGET:.0%}) | "
        f"Width: ±{q:.3f}"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── Panel 2: residual calibration plot ───────────────────────────────
    # A well-calibrated model: the CDF of normalised residuals should
    # follow a uniform distribution (diagonal line)
    residuals = np.abs(test_preds - test_targets)
    sorted_res = np.sort(residuals)
    empirical_cdf = np.arange(1, len(sorted_res) + 1) / len(sorted_res)

    # Ideal: for a perfectly calibrated model with quantile q,
    # fraction p of residuals should be < q_p
    axes[1].plot(sorted_res, empirical_cdf,
                 color="#7F77DD", linewidth=2, label="Empirical CDF")
    axes[1].axvline(q, color="#D85A30", linestyle="--", linewidth=1.5,
                    label=f"Conformal q = {q:.3f}")
    axes[1].axhline(COVERAGE_TARGET, color="#1D9E75", linestyle=":",
                    linewidth=1.5, label=f"Target coverage = {COVERAGE_TARGET:.0%}")
    axes[1].set_xlabel("|Residual| = |predicted - true|")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title(f"{drug_name} — Residual Calibration")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Conformal Prediction — {drug_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{drug_name}_calibration.png", dpi=120)
    plt.close()
    print(f"    Saved: results/uncertainty/{drug_name}_calibration.png")


# ── Save interval CSV ────────────────────────────────────────────────────────

def save_intervals(
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    q: float,
    drug_name: str,
    test_ds,
):
    df = pd.DataFrame({
        "cell_line":   test_ds.cell_lines,
        "cancer_type": [test_ds.meta.loc[cl, "lineage"] for cl in test_ds.cell_lines],
        "true_ln_ic50":  test_targets,
        "pred_ln_ic50":  test_preds,
        "interval_lower": test_preds - q,
        "interval_upper": test_preds + q,
        "covered": (
            (test_targets >= test_preds - q) &
            (test_targets <= test_preds + q)
        ),
        "abs_error": np.abs(test_preds - test_targets),
    })
    df = df.sort_values("abs_error")
    out = RESULTS_DIR / f"{drug_name}_intervals.csv"
    df.to_csv(out, index=False)
    print(f"    Saved: results/uncertainty/{drug_name}_intervals.csv")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 7: Conformal Uncertainty")
    print("=" * 55)

    all_coverage = []

    for drug in TARGET_DRUGS:
        print(f"\n── {drug} ──────────────────────────────────────")

        try:
            # Load datasets
            val_ds  = DrugResponseDataset(drug_name=drug, split="val")
            test_ds = DrugResponseDataset(drug_name=drug, split="test")

            if len(val_ds) == 0 or len(test_ds) == 0:
                print(f"  Skipping — no val/test data")
                continue

            # Load trained model
            model = load_gat(drug, n_features=val_ds.n_features)

            # Get predictions
            val_preds,  val_targets  = get_predictions(model, val_ds)
            test_preds, test_targets = get_predictions(model, test_ds)

            print(f"  Val samples:  {len(val_preds)}")
            print(f"  Test samples: {len(test_preds)}")

            # Calibrate conformal quantile on val set
            q = calibrate_conformal(val_preds, val_targets)
            print(f"  Conformal quantile q = {q:.4f}  (interval half-width)")
            print(f"  Prediction interval width = ±{q:.4f} LN(IC50) units")

            # Evaluate coverage on test set
            cov = evaluate_coverage(test_preds, test_targets, q)
            print(f"  Test coverage: {cov['coverage']:.1%}  (target: {COVERAGE_TARGET:.0%})")
            print(f"  Covered: {cov['n_covered']}/{cov['n_test']} test samples")

            if cov["coverage"] >= COVERAGE_TARGET:
                print(f"  ✓ Coverage guarantee MET")
            else:
                print(f"  ✗ Coverage below target — likely due to small val set size")

            # Save outputs
            save_intervals(test_preds, test_targets, q, drug, test_ds)
            plot_intervals(test_preds, test_targets, q, drug, cov)

            all_coverage.append({
                "drug":            drug,
                "conformal_q":     q,
                "interval_width":  cov["interval_width"],
                "test_coverage":   cov["coverage"],
                "target_coverage": COVERAGE_TARGET,
                "n_covered":       cov["n_covered"],
                "n_test":          cov["n_test"],
                "coverage_met":    cov["coverage"] >= COVERAGE_TARGET,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────
    if all_coverage:
        df = pd.DataFrame(all_coverage)
        df.to_csv(Path("results") / "coverage_summary.csv", index=False)

        print("\n\n" + "=" * 55)
        print("  CONFORMAL PREDICTION SUMMARY")
        print("=" * 55)
        print(f"\n  {'Drug':<12} {'Coverage':>10} {'Target':>8} {'Width':>8} {'Met?':>6}")
        print(f"  {'-'*46}")
        for _, row in df.iterrows():
            met = "✓" if row["coverage_met"] else "✗"
            print(
                f"  {row['drug']:<12} "
                f"{row['test_coverage']:>9.1%} "
                f"{row['target_coverage']:>7.0%} "
                f"{row['interval_width']:>8.4f} "
                f"{met:>6}"
            )

        print(f"\n  Saved: results/coverage_summary.csv")
        print(f"  Saved: results/uncertainty/")
        print(f"\nDone. Run step8_explain.py next (SHAP).")


if __name__ == "__main__":
    main()