"""
step5_model.py
==============
Graph Attention Network (GAT) for drug response prediction.

Architecture:
  Input: protein expression node features [n_proteins × n_features]
    ↓
  GAT Layer 1: multi-head attention (8 heads, hidden_dim=64) + ELU + dropout
    ↓
  GAT Layer 2: multi-head attention (4 heads, hidden_dim=64) + ELU + dropout
    ↓
  GAT Layer 3: single-head attention (out_dim=64) + ELU
    ↓
  Global mean pooling: [batch × 64]    ← collapses graph to fixed vector
    ↓
  MLP head: 64 → 32 → 1               ← predicts LN(IC50)

Why GAT over GCN?
  GCN aggregates neighbours with fixed normalised weights.
  GAT learns attention weights — it discovers WHICH neighbours matter for
  each protein's role in drug response. This is biologically meaningful:
  in the EGFR pathway, attention should weight EGFR neighbours highly
  when predicting Erlotinib (an EGFR inhibitor) response.

Run:
    python src/step5_model.py

Output:
    Prints model architecture and parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch


class DrugResponseGAT(nn.Module):
    """
    Graph Attention Network for predicting LN(IC50) drug response.

    Args:
        in_channels   : number of input features per protein node
        hidden_dim    : hidden dimension for GAT layers
        heads_1       : attention heads in layer 1
        heads_2       : attention heads in layer 2
        dropout       : dropout probability (applied in training)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        heads_1: int    = 8,
        heads_2: int    = 4,
        dropout: float  = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # ── GAT layers ────────────────────────────────────────────────────
        # Layer 1: in_channels → hidden_dim × heads_1
        self.gat1 = GATv2Conv(
            in_channels  = in_channels,
            out_channels = hidden_dim,
            heads        = heads_1,
            dropout      = dropout,
            edge_dim     = 1,        # edge features: STRING confidence score
            concat       = True,     # concatenate heads
        )
        # After layer 1: node dim = hidden_dim * heads_1

        # Layer 2: (hidden_dim * heads_1) → hidden_dim × heads_2
        self.gat2 = GATv2Conv(
            in_channels  = hidden_dim * heads_1,
            out_channels = hidden_dim,
            heads        = heads_2,
            dropout      = dropout,
            edge_dim     = 1,
            concat       = True,
        )
        # After layer 2: node dim = hidden_dim * heads_2

        # Layer 3: (hidden_dim * heads_2) → hidden_dim (single head, no concat)
        self.gat3 = GATv2Conv(
            in_channels  = hidden_dim * heads_2,
            out_channels = hidden_dim,
            heads        = 1,
            dropout      = dropout,
            edge_dim     = 1,
            concat       = False,
        )
        # After layer 3: node dim = hidden_dim

        # ── Batch normalisation ───────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads_1)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads_2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # ── Global pooling → graph-level representation ───────────────────
        # We concatenate mean and max pooling for richer representation
        pool_dim = hidden_dim * 2

        # ── MLP regression head ───────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),   # single output: LN(IC50)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for linear layers."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data : PyG Batch object with .x, .edge_index, .edge_attr, .batch

        Returns:
            predictions : [batch_size, 1] — predicted LN(IC50)
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # ── GAT layer 1 ───────────────────────────────────────────────────
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── GAT layer 2 ───────────────────────────────────────────────────
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── GAT layer 3 ───────────────────────────────────────────────────
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.elu(x)

        # ── Readout: mean + max pooling concatenated ──────────────────────
        x_mean = global_mean_pool(x, batch)   # [batch, hidden_dim]
        x_max  = global_max_pool(x, batch)    # [batch, hidden_dim]
        x_pool = torch.cat([x_mean, x_max], dim=1)   # [batch, hidden_dim*2]

        # ── MLP head → LN(IC50) ───────────────────────────────────────────
        out = self.mlp(x_pool)   # [batch, 1]
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Baseline MLP (for comparison) ────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """
    Flat MLP baseline: ignores graph structure entirely.
    Treats protein expression as a flat feature vector.
    Used to demonstrate that GNN outperforms structure-agnostic model.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        # Flatten all node features per graph sample
        # data.x: [total_nodes, n_features] with data.batch indicating which graph
        batch_size = data.batch.max().item() + 1
        flat = []
        for i in range(batch_size):
            mask = data.batch == i
            node_feats = data.x[mask]             # [n_nodes, n_features]
            flat.append(node_feats.flatten())     # [n_nodes * n_features]
        x = torch.stack(flat)
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Verification ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 5: Model Architecture")
    print("=" * 55)

    # Dummy forward pass
    from torch_geometric.data import Data

    n_nodes    = 200    # proteins
    n_features = 200    # same as nodes (1 feature per protein)
    n_edges    = 2400   # ~12k directed edges for undirected graph

    dummy_data = Data(
        x          = torch.randn(n_nodes, n_features),
        edge_index = torch.randint(0, n_nodes, (2, n_edges)),
        edge_attr  = torch.rand(n_edges, 1),
        y          = torch.randn(1),
        batch      = torch.zeros(n_nodes, dtype=torch.long),
    )

    model = DrugResponseGAT(in_channels=n_features)
    model.eval()

    with torch.no_grad():
        out = model(dummy_data)

    print(f"\n  Model: DrugResponseGAT")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Input node features: {n_features}")
    print(f"  Dummy output shape:  {out.shape}   ← should be [1, 1]")
    print(f"\n  Architecture:")
    print(f"    GAT Layer 1: {n_features} → 64×8 heads  = 512 per node")
    print(f"    GAT Layer 2: 512          → 64×4 heads  = 256 per node")
    print(f"    GAT Layer 3: 256          → 64×1 head   =  64 per node")
    print(f"    Pooling:     mean + max                  = 128 graph repr")
    print(f"    MLP:         128 → 64 → 32 → 1")

    baseline = BaselineMLP(in_features=n_nodes * n_features)
    print(f"\n  Baseline MLP parameters: {baseline.count_parameters():,}")
    print(f"\nModel ready. Run step6_train.py next.")
