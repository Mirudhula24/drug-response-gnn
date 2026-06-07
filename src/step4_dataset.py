"""
step4_dataset.py
================
PyTorch Geometric Dataset for the drug response GNN.

Each sample is one (cell_line, drug) pair:
  - Data.x         : node features = protein expression values [n_proteins, 1]
  - Data.edge_index: PPI graph topology [2, n_edges]       — same for all samples
  - Data.edge_attr : edge weights (STRING confidence)      — same for all samples
  - Data.y         : LN(IC50) target value [1]
  - Data.cell_line : str — for tracking predictions
  - Data.drug      : str — which drug this sample is for
  - Data.split     : str — "train" / "val" / "test"

Run as script to verify:
    python src/step4_dataset.py

Output:
    Prints dataset statistics per drug per split.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Optional

PROCESSED_DIR = Path("data/processed")

TARGET_DRUGS = [
    "Erlotinib",
    "Lapatinib",
    "Sorafenib",
    "Imatinib",
    "Paclitaxel",
]


class DrugResponseDataset(Dataset):
    """
    PyTorch Geometric dataset for cancer drug response prediction.

    One graph per (cell line, drug) pair.
    The PPI graph topology is fixed; only node features change per cell line.

    Args:
        drug_name : one of TARGET_DRUGS
        split     : "train", "val", or "test" (None = all)
        data_dir  : path to processed data folder
    """

    def __init__(
        self,
        drug_name: str,
        split: Optional[str] = None,
        data_dir: Path = PROCESSED_DIR,
    ):
        self.drug_name = drug_name
        self.split     = split
        self.data_dir  = data_dir

        # Load all processed data BEFORE calling super().__init__()
        self._load_data()
        super().__init__()

    def _load_data(self):
        """Load and align all data sources."""
        # Protein expression: [n_cell_lines x n_proteins]
        protein_df = pd.read_csv(
            self.data_dir / "protein_expr.csv", index_col=0
        )
        # Drug response: [n_cell_lines x n_drugs]
        drug_df = pd.read_csv(
            self.data_dir / "drug_response.csv", index_col=0
        )
        # Metadata: cell line -> lineage + split
        meta_df = pd.read_csv(
            self.data_dir / "cell_metadata.csv", index_col=0
        )
        # Protein index: protein_name -> node_id
        with open(self.data_dir / "protein_index.json") as f:
            self.protein_idx = json.load(f)

        # Graph structure (shared across all samples)
        self.edge_index = torch.from_numpy(
            np.load(self.data_dir / "edge_index.npy")
        ).long()
        self.edge_attr = torch.from_numpy(
            np.load(self.data_dir / "edge_weight.npy")
        ).float().unsqueeze(1)   # [n_edges, 1]

        # Check drug exists
        if self.drug_name not in drug_df.columns:
            available = [c for c in drug_df.columns if c in TARGET_DRUGS]
            raise ValueError(
                f"Drug '{self.drug_name}' not found. "
                f"Available: {available}"
            )

        # Keep only cell lines with IC50 for this drug
        drug_col = drug_df[self.drug_name].dropna()
        shared   = (
            protein_df.index
            .intersection(drug_col.index)
            .intersection(meta_df.index)
        )

        # Filter by split if specified
        if self.split is not None:
            split_mask = meta_df.loc[shared, "split"] == self.split
            shared = shared[split_mask]

        self.cell_lines   = shared.tolist()
        self.protein_expr = protein_df.loc[shared]
        self.ic50_values  = drug_col.loc[shared]
        self.meta         = meta_df.loc[shared]

        # Order protein columns to match protein_idx
        ordered_proteins   = sorted(self.protein_idx.keys(), key=lambda p: self.protein_idx[p])
        available_proteins = [p for p in ordered_proteins if p in self.protein_expr.columns]
        self.protein_order = available_proteins
        self.n_proteins    = len(self.protein_idx)
        self.n_features    = len(available_proteins)

    # ── PyG required methods ─────────────────────────────────────────────────

    def len(self) -> int:
        return len(self.cell_lines)

    def get(self, idx: int) -> Data:
        """Build a PyG Data object for one (cell_line, drug) pair."""
        cell_line   = self.cell_lines[idx]
        expr_values = self.protein_expr.loc[cell_line, self.protein_order].to_numpy().flatten()

        # Node feature matrix: [n_proteins, n_features]
        x = torch.zeros(self.n_proteins, self.n_features, dtype=torch.float)
        for feat_idx, protein in enumerate(self.protein_order):
            node_idx = self.protein_idx[protein]
            x[node_idx, feat_idx] = float(expr_values[feat_idx])

        y = torch.tensor([self.ic50_values[cell_line]], dtype=torch.float)

        return Data(
            x           = x,
            edge_index  = self.edge_index,
            edge_attr   = self.edge_attr,
            y           = y,
            cell_line   = cell_line,
            drug        = self.drug_name,
            cancer_type = self.meta.loc[cell_line, "lineage"],
        )

    @property
    def num_node_features(self) -> int:
        return self.n_features

    def summary(self) -> str:
        split_label = self.split or "all"
        return (
            f"DrugResponseDataset | drug={self.drug_name} | split={split_label} | "
            f"n={len(self)} | proteins={self.n_proteins} | features={self.n_features}"
        )


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    drug_name: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """Returns {"train": ..., "val": ..., "test": ...} DataLoaders for one drug."""
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = DrugResponseDataset(drug_name=drug_name, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


# ── Verification ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 4: Dataset Verification")
    print("=" * 55)

    for drug in TARGET_DRUGS:
        print(f"\n── {drug} ──")
        total = 0
        for split in ["train", "val", "test"]:
            try:
                ds = DrugResponseDataset(drug_name=drug, split=split)
                print(f"  {split:5}: {len(ds):4} samples")
                total += len(ds)

                if split == "train" and len(ds) > 0:
                    sample = ds[0]
                    print(f"         sample[0]:")
                    print(f"           x.shape          = {sample.x.shape}")
                    print(f"           edge_index.shape = {sample.edge_index.shape}")
                    print(f"           edge_attr.shape  = {sample.edge_attr.shape}")
                    print(f"           y                = {sample.y.item():.3f} (LN IC50)")
                    print(f"           cell_line        = {sample.cell_line}")
                    print(f"           cancer_type      = {sample.cancer_type}")
            except Exception as e:
                print(f"  {split:5}: ERROR — {e}")

        print(f"  total: {total} samples")

    print("\nDataset looks good. Run step5_model.py next.")
