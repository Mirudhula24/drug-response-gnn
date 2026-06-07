"""
step6_train.py
==============
Training loop for the Drug Response GNN.

Trains one GAT model per drug, compares against baseline MLP.
Saves best checkpoint per drug based on validation RMSE.

Run:
    python src/step6_train.py

Output:
    models/gat_{drug}.pt          — best GAT checkpoint per drug
    models/mlp_{drug}.pt          — best MLP checkpoint per drug
    results/metrics.csv           — RMSE + MAE per drug per model
    results/loss_curves/          — train/val loss plots per drug
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from step4_dataset import DrugResponseDataset, TARGET_DRUGS
from step5_model import DrugResponseGAT, BaselineMLP

# ── Config ───────────────────────────────────────────────────────────────────

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results/loss_curves")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "batch_size":    16,      # small — we only have ~85 train samples
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "epochs":        200,
    "patience":      30,      # early stopping patience
    "hidden_dim":    64,
    "heads_1":       4,       # reduced from 8 — better for small datasets
    "heads_2":       2,
    "dropout":       0.3,
}


# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(pred, target)).item()

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return nn.functional.l1_loss(pred, target).item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze()
        target = batch.y.squeeze()
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze()
        preds.append(pred.cpu())
        targets.append(batch.y.squeeze().cpu())
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    return rmse(preds, targets), mae(preds, targets), preds, targets


# ── Train one drug ────────────────────────────────────────────────────────────

def train_drug(drug_name: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training: {drug_name}")
    print(f"{'='*55}")

    # ── Datasets ──────────────────────────────────────────────
    train_ds = DrugResponseDataset(drug_name=drug_name, split="train")
    val_ds   = DrugResponseDataset(drug_name=drug_name, split="val")
    test_ds  = DrugResponseDataset(drug_name=drug_name, split="test")

    if len(train_ds) == 0:
        print(f"  Skipping {drug_name} — no training data")
        return {}

    print(f"  Samples — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"  Device: {DEVICE}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False)

    n_features = train_ds.n_features

    # ── GAT model ──────────────────────────────────────────────
    gat = DrugResponseGAT(
        in_channels = n_features,
        hidden_dim  = CFG["hidden_dim"],
        heads_1     = CFG["heads_1"],
        heads_2     = CFG["heads_2"],
        dropout     = CFG["dropout"],
    ).to(DEVICE)

    gat_optimizer = optim.Adam(
        gat.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )
    gat_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        gat_optimizer, patience=15, factor=0.5
    )

    # ── MLP baseline ───────────────────────────────────────────
    mlp = BaselineMLP(
        in_features = n_features * train_ds.n_proteins,
        hidden_dim  = 128,
        dropout     = CFG["dropout"],
    ).to(DEVICE)

    mlp_optimizer = optim.Adam(
        mlp.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )

    # ── Training ───────────────────────────────────────────────
    best_gat_val_rmse = float("inf")
    best_mlp_val_rmse = float("inf")
    gat_patience_counter = 0
    mlp_patience_counter = 0
    gat_stopped = False
    mlp_stopped = False

    gat_train_losses, gat_val_rmses = [], []
    mlp_train_losses, mlp_val_rmses = [], []

    print(f"\n  {'Epoch':>5} | {'GAT val RMSE':>12} | {'MLP val RMSE':>12} | {'Best GAT':>8}")
    print(f"  {'-'*50}")

    for epoch in range(1, CFG["epochs"] + 1):

        # GAT step
        if not gat_stopped:
            gat_loss = train_epoch(gat, train_loader, gat_optimizer, DEVICE)
            gat_val_rmse, _, _, _ = evaluate(gat, val_loader, DEVICE)
            gat_scheduler.step(gat_val_rmse)
            gat_train_losses.append(gat_loss)
            gat_val_rmses.append(gat_val_rmse)

            if gat_val_rmse < best_gat_val_rmse:
                best_gat_val_rmse = gat_val_rmse
                torch.save(gat.state_dict(), MODELS_DIR / f"gat_{drug_name}.pt")
                gat_patience_counter = 0
            else:
                gat_patience_counter += 1
                if gat_patience_counter >= CFG["patience"]:
                    gat_stopped = True
                    print(f"  GAT early stopping at epoch {epoch}")

        # MLP step
        if not mlp_stopped:
            mlp_loss = train_epoch(mlp, train_loader, mlp_optimizer, DEVICE)
            mlp_val_rmse, _, _, _ = evaluate(mlp, val_loader, DEVICE)
            mlp_train_losses.append(mlp_loss)
            mlp_val_rmses.append(mlp_val_rmse)

            if mlp_val_rmse < best_mlp_val_rmse:
                best_mlp_val_rmse = mlp_val_rmse
                torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_{drug_name}.pt")
                mlp_patience_counter = 0
            else:
                mlp_patience_counter += 1
                if mlp_patience_counter >= CFG["patience"]:
                    mlp_stopped = True
                    print(f"  MLP early stopping at epoch {epoch}")

        if gat_stopped and mlp_stopped:
            break

        if epoch % 20 == 0 or epoch == 1:
            g_rmse = gat_val_rmses[-1] if gat_val_rmses else float("nan")
            m_rmse = mlp_val_rmses[-1] if mlp_val_rmses else float("nan")
            print(f"  {epoch:>5} | {g_rmse:>12.4f} | {m_rmse:>12.4f} | {best_gat_val_rmse:>8.4f}")

    # ── Test evaluation ────────────────────────────────────────
    # Load best checkpoints
    gat.load_state_dict(torch.load(MODELS_DIR / f"gat_{drug_name}.pt", map_location=DEVICE))
    mlp.load_state_dict(torch.load(MODELS_DIR / f"mlp_{drug_name}.pt", map_location=DEVICE))

    gat_test_rmse, gat_test_mae, gat_preds, gat_targets = evaluate(gat, test_loader, DEVICE)
    mlp_test_rmse, mlp_test_mae, mlp_preds, mlp_targets = evaluate(mlp, test_loader, DEVICE)

    improvement = (mlp_test_rmse - gat_test_rmse) / mlp_test_rmse * 100

    print(f"\n  ── Test Results ──────────────────────────────")
    print(f"  GAT  RMSE: {gat_test_rmse:.4f} | MAE: {gat_test_mae:.4f}")
    print(f"  MLP  RMSE: {mlp_test_rmse:.4f} | MAE: {mlp_test_mae:.4f}")
    print(f"  GAT improvement over MLP: {improvement:+.1f}%")

    # ── Loss curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gat_val_rmses, label="GAT val RMSE", color="#7F77DD", linewidth=2)
    axes[0].plot(mlp_val_rmses, label="MLP val RMSE", color="#D85A30", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"{drug_name} — Validation RMSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        gat_targets.numpy(), gat_preds.numpy(),
        alpha=0.7, color="#7F77DD", s=40, label="GAT predictions"
    )
    axes[1].scatter(
        mlp_targets.numpy(), mlp_preds.numpy(),
        alpha=0.5, color="#D85A30", s=40, marker="^", label="MLP predictions"
    )
    mn = min(gat_targets.min(), mlp_targets.min()).item()
    mx = max(gat_targets.max(), mlp_targets.max()).item()
    axes[1].plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="Perfect prediction")
    axes[1].set_xlabel("True LN(IC50)")
    axes[1].set_ylabel("Predicted LN(IC50)")
    axes[1].set_title(f"{drug_name} — Predicted vs True")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Drug Response GNN — {drug_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{drug_name}_curves.png", dpi=120)
    plt.close()

    return {
        "drug":             drug_name,
        "gat_val_rmse":     best_gat_val_rmse,
        "gat_test_rmse":    gat_test_rmse,
        "gat_test_mae":     gat_test_mae,
        "mlp_val_rmse":     best_mlp_val_rmse,
        "mlp_test_rmse":    mlp_test_rmse,
        "mlp_test_mae":     mlp_test_mae,
        "improvement_pct":  improvement,
        "train_samples":    len(train_ds),
        "test_samples":     len(test_ds),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 6: Training")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    all_results = []

    for drug in TARGET_DRUGS:
        try:
            result = train_drug(drug)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR training {drug}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(Path("results") / "metrics.csv", index=False)

        print("\n\n" + "=" * 55)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 55)
        print(f"\n  {'Drug':<12} {'GAT RMSE':>10} {'MLP RMSE':>10} {'Improvement':>12}")
        print(f"  {'-'*46}")
        for _, row in df.iterrows():
            print(f"  {row['drug']:<12} {row['gat_test_rmse']:>10.4f} {row['mlp_test_rmse']:>10.4f} {row['improvement_pct']:>11.1f}%")

        avg_improvement = df["improvement_pct"].mean()
        print(f"\n  Average GAT improvement over MLP: {avg_improvement:+.1f}%")
        print(f"\n  Saved: results/metrics.csv")
        print(f"  Saved: results/loss_curves/*.png")
        print(f"  Saved: models/gat_*.pt")
        print(f"\nDone. Run step7_uncertainty.py next (week 3).")


if __name__ == "__main__":
    main()