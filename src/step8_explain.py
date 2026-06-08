"""
step8_explain.py
================
SHAP-based explainability for the Drug Response GNN.

For each drug, identifies which proteins most strongly drive
the model's IC50 predictions — i.e. which proteins, when highly
expressed, push the predicted drug response up or down.

Why this matters:
  - Biologically validates the model (EGFR should matter for Erlotinib)
  - Makes predictions interpretable for clinicians
  - Each SHAP plot is a figure you can put in your README/portfolio

How it works:
  We use SHAP's KernelExplainer on a "flattened" version of the model —
  input = protein expression vector, output = predicted LN(IC50).
  KernelExplainer is model-agnostic: works on any black-box function.

Run:
    python src/step8_explain.py

Output:
    results/shap/
        {drug}_shap_bar.png       — mean |SHAP| per protein (bar chart)
        {drug}_shap_beeswarm.png  — SHAP value distribution per protein
        {drug}_shap_values.csv    — raw SHAP values for all test samples
        shap_top_proteins.csv     — top 5 proteins per drug summary
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step4_dataset import DrugResponseDataset, TARGET_DRUGS
from step5_model import DrugResponseGAT
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results/shap")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")   # SHAP runs on CPU always


# ── Load model ───────────────────────────────────────────────────────────────

def load_gat(drug_name: str, n_features: int) -> DrugResponseGAT:
    model = DrugResponseGAT(
        in_channels = n_features,
        hidden_dim  = 64,
        heads_1     = 4,
        heads_2     = 2,
        dropout     = 0.0,
    ).to(DEVICE)
    ckpt = MODELS_DIR / f"gat_{drug_name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


# ── Build prediction function for SHAP ───────────────────────────────────────

def make_predict_fn(model, dataset):
    """
    Returns a function: np.ndarray [n_samples, n_features] -> np.ndarray [n_samples]
    that SHAP can call repeatedly.

    We rebuild a Data object for each sample using the fixed graph topology
    but substituting the input protein expression values.
    """
    edge_index = dataset.edge_index
    edge_attr  = dataset.edge_attr
    n_proteins = dataset.n_proteins
    protein_idx = dataset.protein_idx
    protein_order = dataset.protein_order

    def predict(X: np.ndarray) -> np.ndarray:
        """X shape: [n_samples, n_features]"""
        preds = []
        for i in range(len(X)):
            expr = X[i]   # [n_features]

            # Rebuild node feature matrix
            x = torch.zeros(n_proteins, len(protein_order), dtype=torch.float)
            for feat_idx, protein in enumerate(protein_order):
                node_idx = protein_idx[protein]
                x[node_idx, feat_idx] = float(expr[feat_idx])

            data = Data(
                x          = x,
                edge_index = edge_index,
                edge_attr  = edge_attr,
                batch      = torch.zeros(n_proteins, dtype=torch.long),
            )
            with torch.no_grad():
                pred = model(data).squeeze().item()
            preds.append(pred)
        return np.array(preds)

    return predict


# ── Run SHAP ──────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    train_ds,
    test_ds,
    drug_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute SHAP values using KernelExplainer.

    Background: mean protein expression over training set (summarised to 10
    samples via k-means to keep runtime fast).
    Explain: all test samples.

    Returns:
        shap_values  : [n_test, n_features]
        test_X       : [n_test, n_features]
        feature_names: list of protein names
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    feature_names = train_ds.protein_order

    # Build training matrix [n_train, n_features]
    train_X = train_ds.protein_expr[feature_names].values.astype(np.float32)
    test_X  = test_ds.protein_expr[feature_names].values.astype(np.float32)

    predict_fn = make_predict_fn(model, train_ds)

    print(f"    Computing SHAP background (k-means summary)...")
    # Summarise background to 10 samples — keeps KernelExplainer fast
    background = shap.kmeans(train_X, min(10, len(train_X)))

    print(f"    Running KernelExplainer on {len(test_X)} test samples...")
    print(f"    (This takes ~1-3 min per drug on CPU)")
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_X, nsamples=100, silent=True)

    return np.array(shap_values), test_X, feature_names


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    drug_name: str,
    top_n: int = 15,
):
    """Mean absolute SHAP value per protein — shows overall importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]

    proteins = [feature_names[i] for i in sorted_idx]
    values   = mean_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdPu(np.linspace(0.3, 0.9, len(proteins)))[::-1]
    bars = ax.barh(range(len(proteins)), values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|  (impact on LN IC50 prediction)")
    ax.set_title(
        f"{drug_name} — Top {top_n} Proteins by SHAP Importance\n"
        f"Higher = stronger influence on predicted drug response",
        fontsize=11
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{drug_name}_shap_bar.png", dpi=130)
    plt.close()
    print(f"    Saved: results/shap/{drug_name}_shap_bar.png")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    test_X: np.ndarray,
    feature_names: list[str],
    drug_name: str,
    top_n: int = 15,
):
    """
    Beeswarm-style dot plot: each dot = one test sample.
    Position on x = SHAP value (positive = pushes IC50 up).
    Colour = feature value (red = high expression, blue = low).
    """
    mean_abs   = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]
    proteins   = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))

    for plot_rank, feat_idx in enumerate(sorted_idx[::-1]):
        sv   = shap_values[:, feat_idx]   # SHAP values for this protein
        fval = test_X[:, feat_idx]        # actual expression values

        # Normalise feature values for colour mapping
        fval_norm = (fval - fval.min()) / (fval.max() - fval.min() + 1e-8)
        colors = plt.cm.RdBu_r(fval_norm)

        # Jitter y-position for readability
        y = np.full(len(sv), plot_rank) + np.random.uniform(-0.2, 0.2, len(sv))
        ax.scatter(sv, y, c=colors, s=25, alpha=0.8, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(proteins[::-1], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value  (positive = higher IC50 = more resistant to drug)")
    ax.set_title(
        f"{drug_name} — SHAP Value Distribution per Protein\n"
        f"Colour: red = high expression, blue = low expression",
        fontsize=11
    )
    ax.grid(axis="x", alpha=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Normalised expression", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{drug_name}_shap_beeswarm.png", dpi=130)
    plt.close()
    print(f"    Saved: results/shap/{drug_name}_shap_beeswarm.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 8: SHAP Explainability")
    print("=" * 55)

    all_top_proteins = []

    for drug in TARGET_DRUGS:
        print(f"\n── {drug} ──────────────────────────────────────")

        try:
            train_ds = DrugResponseDataset(drug_name=drug, split="train")
            test_ds  = DrugResponseDataset(drug_name=drug, split="test")

            if len(test_ds) == 0:
                print(f"  Skipping — no test data")
                continue

            model = load_gat(drug, n_features=train_ds.n_features)

            shap_values, test_X, feature_names = compute_shap_values(
                model, train_ds, test_ds, drug
            )

            # Save raw SHAP values
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_df.insert(0, "cell_line", test_ds.cell_lines)
            shap_df.to_csv(RESULTS_DIR / f"{drug}_shap_values.csv", index=False)

            # Plots
            plot_shap_bar(shap_values, feature_names, drug)
            plot_shap_beeswarm(shap_values, test_X, feature_names, drug)

            # Top 5 proteins for summary
            mean_abs   = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)[::-1][:5]
            top5 = [feature_names[i] for i in sorted_idx]
            top5_vals = mean_abs[sorted_idx]

            print(f"\n  Top 5 resistance-driving proteins for {drug}:")
            for rank, (prot, val) in enumerate(zip(top5, top5_vals), 1):
                print(f"    {rank}. {prot:<20} SHAP = {val:.4f}")

            for rank, (prot, val) in enumerate(zip(top5, top5_vals), 1):
                all_top_proteins.append({
                    "drug": drug,
                    "rank": rank,
                    "protein": prot,
                    "mean_abs_shap": val,
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────────────────
    if all_top_proteins:
        df = pd.DataFrame(all_top_proteins)
        df.to_csv(Path("results") / "shap_top_proteins.csv", index=False)

        print("\n\n" + "=" * 55)
        print("  TOP RESISTANCE PROTEINS PER DRUG")
        print("=" * 55)
        for drug in df["drug"].unique():
            sub = df[df["drug"] == drug]
            proteins = ", ".join(sub["protein"].tolist())
            print(f"  {drug:<12}: {proteins}")

        print(f"\n  Saved: results/shap_top_proteins.csv")
        print(f"  Saved: results/shap/*.png")
        print(f"\nWeek 3 complete. Run app.py next (week 4 — Streamlit demo).")


if __name__ == "__main__":
    main()