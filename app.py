"""
app.py
======
Streamlit demo app for the Drug Response GNN project.

Features:
  - Select cancer type + drug
  - Get predicted LN(IC50) + 90% conformal prediction interval
  - SHAP bar chart showing top resistance proteins
  - Model performance summary table
  - Clean, professional UI

Run locally:
    streamlit run app.py

Deploy to Hugging Face Spaces:
    1. Create a new Space (Streamlit SDK)
    2. Upload: app.py, requirements.txt, models/, data/processed/, results/
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, "src")
from step4_dataset import DrugResponseDataset
from step5_model import DrugResponseGAT
from torch_geometric.data import Data

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drug Response GNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")

AVAILABLE_DRUGS = ["Erlotinib", "Lapatinib", "Sorafenib", "Paclitaxel"]

DRUG_INFO = {
    "Erlotinib":  {"mechanism": "EGFR inhibitor",        "cancer": "Lung cancer",          "color": "#7F77DD"},
    "Lapatinib":  {"mechanism": "EGFR/HER2 inhibitor",   "cancer": "Breast cancer",        "color": "#1D9E75"},
    "Sorafenib":  {"mechanism": "RAF/VEGFR inhibitor",   "cancer": "Liver/Kidney cancer",  "color": "#D85A30"},
    "Paclitaxel": {"mechanism": "Microtubule stabiliser","cancer": "Breast/Ovarian cancer","color": "#378ADD"},
}

CONFORMAL_Q = {
    "Erlotinib":  2.8584,
    "Lapatinib":  3.8064,
    "Sorafenib":  2.3002,
    "Paclitaxel": 2.8971,
}


# ── Cache model loading ───────────────────────────────────────────────────────

@st.cache_resource
def load_model(drug_name: str, n_features: int) -> DrugResponseGAT:
    model = DrugResponseGAT(
        in_channels = n_features,
        hidden_dim  = 64,
        heads_1     = 4,
        heads_2     = 2,
        dropout     = 0.0,
    )
    ckpt = MODELS_DIR / f"gat_{drug_name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_dataset(drug_name: str):
    return DrugResponseDataset(drug_name=drug_name, split="test")


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_shap_top() -> pd.DataFrame:
    path = RESULTS_DIR / "shap_top_proteins.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_protein_index() -> dict:
    with open(PROCESSED_DIR / "protein_index.json") as f:
        return json.load(f)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_single(
    model: DrugResponseGAT,
    expr_values: np.ndarray,
    dataset: DrugResponseDataset,
) -> float:
    """Run model on a single protein expression vector."""
    protein_idx   = dataset.protein_idx
    protein_order = dataset.protein_order
    edge_index    = dataset.edge_index
    edge_attr     = dataset.edge_attr
    n_proteins    = dataset.n_proteins

    x = torch.zeros(n_proteins, len(protein_order), dtype=torch.float)
    for feat_idx, protein in enumerate(protein_order):
        node_idx = protein_idx[protein]
        x[node_idx, feat_idx] = float(expr_values[feat_idx])

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        batch      = torch.zeros(n_proteins, dtype=torch.long),
    )
    with torch.no_grad():
        pred = model(data).squeeze().item()
    return pred


# ── SHAP bar chart ────────────────────────────────────────────────────────────

def plot_shap_bar_streamlit(drug_name: str, color: str):
    shap_path = RESULTS_DIR / "shap" / f"{drug_name}_shap_values.csv"
    if not shap_path.exists():
        return None

    df = pd.read_csv(shap_path)
    proteins = [c for c in df.columns if c != "cell_line"]
    mean_abs = df[proteins].abs().mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        range(len(mean_abs)),
        mean_abs.values[::-1],
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels(mean_abs.index[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(f"Top resistance-driving proteins — {drug_name}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧬 Drug Response GNN")
        st.markdown(
            "Predicts cancer cell line drug response (IC50) "
            "from protein expression using a **Graph Attention Network** "
            "over protein–protein interaction networks."
        )
        st.divider()

        st.markdown("**Select drug**")
        drug = st.selectbox(
            "Drug",
            AVAILABLE_DRUGS,
            label_visibility="collapsed",
        )

        info = DRUG_INFO[drug]
        st.markdown(f"**Mechanism:** {info['mechanism']}")
        st.markdown(f"**Primary use:** {info['cancer']}")
        st.divider()

        st.markdown("**Select cell line**")
        try:
            test_ds = load_dataset(drug)
            cell_lines = sorted(test_ds.cell_lines)
            selected_cell = st.selectbox(
                "Cell line",
                cell_lines,
                label_visibility="collapsed",
            )
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
            return

        st.divider()
        st.markdown(
            "**Data sources**\n\n"
            "- Protein expression: [CCLE](https://depmap.org)\n"
            "- Drug response: [GDSC2](https://www.cancerrxgene.org)\n"
            "- PPI graph: [STRING DB](https://string-db.org)\n"
        )
        st.markdown(
            "**Model**\n\n"
            "GATv2 · 3 layers · 515K params\n\n"
            "Uncertainty: conformal prediction\n\n"
            "Explainability: SHAP KernelExplainer"
        )

    # ── Main panel ─────────────────────────────────────────────────────────────
    st.title("🧬 Cancer Drug Response Predictor")
    st.markdown(
        "Built with **PyTorch Geometric + STRING PPI graph + conformal prediction**. "
        "Predicts how sensitive a cancer cell line is to a drug — with uncertainty intervals."
    )

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader(f"Prediction — {drug} on {selected_cell}")

        try:
            # Load model and get expression for selected cell line
            model = load_model(drug, n_features=test_ds.n_features)
            cancer_type = test_ds.meta.loc[selected_cell, "lineage"]
            expr_values = test_ds.protein_expr.loc[
                selected_cell, test_ds.protein_order
            ].to_numpy().flatten()

            pred   = predict_single(model, expr_values, test_ds)
            q      = CONFORMAL_Q[drug]
            lower  = pred - q
            upper  = pred + q
            true_y = test_ds.ic50_values.get(selected_cell, None)

            # ── Prediction display ─────────────────────────────────────────
            st.markdown(f"**Cancer type:** `{cancer_type}`")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted LN(IC50)",
                f"{pred:.3f}",
                help="Log-transformed IC50. Higher = more resistant to drug."
            )
            m2.metric(
                "90% Interval",
                f"[{lower:.2f}, {upper:.2f}]",
                help="Conformal prediction interval — guaranteed to contain the true value 90% of the time."
            )
            if true_y is not None:
                error = abs(pred - true_y)
                m3.metric(
                    "True LN(IC50)",
                    f"{true_y:.3f}",
                    delta=f"Error: {error:.3f}",
                    delta_color="inverse",
                )

            # ── Interval visualisation ─────────────────────────────────────
            st.markdown("**Prediction interval**")
            fig_int, ax_int = plt.subplots(figsize=(7, 1.8))
            ax_int.barh(
                0, upper - lower, left=lower,
                height=0.4, color=info["color"], alpha=0.3,
                label="90% interval"
            )
            ax_int.axvline(pred,  color=info["color"], linewidth=2.5, label="Prediction")
            if true_y is not None:
                ax_int.axvline(true_y, color="#333333", linewidth=2,
                               linestyle="--", label="True value")
            ax_int.set_yticks([])
            ax_int.set_xlabel("LN(IC50)")
            ax_int.legend(loc="upper right", fontsize=9)
            ax_int.grid(axis="x", alpha=0.3)
            ax_int.spines["top"].set_visible(False)
            ax_int.spines["right"].set_visible(False)
            ax_int.spines["left"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_int, use_container_width=True)
            plt.close()

            # ── Interpretation ─────────────────────────────────────────────
            if pred > 3.0:
                sensitivity = "🔴 **Resistant** — high IC50 suggests this cell line does not respond well to this drug"
            elif pred > 1.5:
                sensitivity = "🟡 **Moderate sensitivity** — intermediate IC50"
            else:
                sensitivity = "🟢 **Sensitive** — low IC50 suggests strong drug response"
            st.info(sensitivity)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())

    with col2:
        st.subheader("Resistance proteins (SHAP)")
        fig_shap = plot_shap_bar_streamlit(drug, info["color"])
        if fig_shap:
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

            shap_top = load_shap_top()
            if not shap_top.empty:
                top5 = shap_top[shap_top["drug"] == drug]["protein"].tolist()
                st.markdown(f"**Top drivers:** {', '.join(top5[:3])}")
        else:
            st.info("Run step8_explain.py to generate SHAP values")

    # ── Model performance ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Model performance across drugs")

    metrics = load_metrics()
    if not metrics.empty:
        col_a, col_b = st.columns([1.5, 1])

        with col_a:
            display = metrics[["drug", "gat_test_rmse", "mlp_test_rmse", "improvement_pct"]].copy()
            display.columns = ["Drug", "GAT RMSE", "MLP RMSE", "GAT vs MLP"]
            display["GAT RMSE"]   = display["GAT RMSE"].round(4)
            display["MLP RMSE"]   = display["MLP RMSE"].round(4)
            display["GAT vs MLP"] = display["GAT vs MLP"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(display, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("""
**Key results:**
- GAT outperforms flat MLP by up to **+12.6%** (Paclitaxel)
- Conformal intervals achieve **≥90% coverage** on all 4 drugs
- SHAP identifies known biology: **HER2** top driver for Lapatinib, **PTEN** for Sorafenib
- Model trained on **CCLE × GDSC2** with cell-line stratified splits to avoid data leakage
            """)

    # ── Architecture ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("Model architecture"):
        st.markdown("""
```
Input: protein expression node features  [158 proteins × 35 features]
         ↓
GAT Layer 1:  multi-head attention (4 heads, dim=64)  +  ELU  +  dropout
         ↓
GAT Layer 2:  multi-head attention (2 heads, dim=64)  +  ELU  +  dropout
         ↓
GAT Layer 3:  single-head attention       (dim=64)    +  ELU
         ↓
Global mean + max pooling  →  [128-dim graph representation]
         ↓
MLP: 128 → 64 → 32 → 1   (predicted LN IC50)
         ↓
Conformal calibration on val set  →  90% prediction interval
```
**Graph:** 158 protein nodes · 328 directed edges · STRING confidence ≥ 700  
**Training:** Adam · lr=1e-3 · early stopping (patience=30) · cell-line stratified splits  
**Parameters:** 515,905 (GAT) vs 5,128,449 (MLP baseline)
        """)

    st.caption(
        "Built by Mimi · "
        "[GitHub](https://github.com/Mirudhula24/drug-response-gnn) · "
        "Stack: PyTorch Geometric · STRING DB · CCLE · GDSC2 · Conformal Prediction · SHAP"
    )


if __name__ == "__main__":
    main()