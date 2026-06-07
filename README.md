# Drug Response GNN
### Predicting cancer drug response from protein–protein interaction networks

> Built by Mimi | Portfolio project — ML Engineer track

---

## What this does

Given a cancer cell line's **protein expression profile**, this model predicts
how much a specific drug will slow cell growth — expressed as **LN(IC50)**.

Instead of treating proteins as an unordered flat vector, we model them as a
**graph**: nodes are proteins, edges are known biological interactions from the
STRING database. A Graph Attention Network (GAT) learns which protein
neighbourhoods are most predictive of drug response.

Uniquely, predictions come with **conformal prediction intervals** — so instead
of "IC50 = 2.3", you get "IC50 = 2.3 ± 0.4 (90% coverage)".

---

## Paper references

- **GATv2**: Brody et al., *How Attentive are Graph Attention Networks?* ICLR 2022
- **STRING DB**: Szklarczyk et al., *Nucleic Acids Research* 2021
- **CCLE**: Ghandi et al., *Nature* 2019
- **GDSC**: Yang et al., *Nucleic Acids Research* 2013
- **Conformal prediction**: Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* 2022

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install PyTorch (CPU — change index URL for CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Run week 1 (data + graph)

```bash
# Step 1: Download CCLE protein data + GDSC drug response
python src/step1_download_data.py

# Step 2: Preprocess, impute, split
python src/step2_preprocess.py

# Step 3: Build PPI graph from STRING
python src/step3_build_graph.py

# Step 4: Verify dataset
python src/step4_dataset.py

# Step 5: Verify model architecture
python src/step5_model.py
```

---

## Project structure

```
drug-response-gnn/
├── data/
│   ├── raw/               ← downloaded files (gitignored)
│   └── processed/         ← cleaned, aligned data + graph files
├── src/
│   ├── step1_download_data.py
│   ├── step2_preprocess.py     ← KNN imputation + stratified split
│   ├── step3_build_graph.py    ← STRING PPI graph construction
│   ├── step4_dataset.py        ← PyTorch Geometric Dataset class
│   ├── step5_model.py          ← GATv2 architecture + baseline MLP
│   ├── step6_train.py          ← training loop (week 2)
│   ├── step7_uncertainty.py    ← conformal prediction (week 3)
│   └── step8_explain.py        ← SHAP node attribution (week 3)
├── models/                ← saved model checkpoints
├── results/               ← metrics, plots
├── app.py                 ← Streamlit demo (week 4)
└── requirements.txt
```

---

## Key design decisions (interview talking points)

| Decision | Why it matters |
|---|---|
| GATv2 over GCN | Learns which protein neighbours matter — biologically interpretable |
| Cell-line stratified split | Prevents data leakage — most papers get this wrong |
| KNN imputation within cancer type | Biologically meaningful; global imputation adds noise |
| 5 curated drugs only | Focused scope with good data coverage; clinically important |
| Conformal prediction | Rigorous uncertainty — not just dropout variance |
| SHAP on node features | Per-protein attribution → which proteins drive resistance |

---

## Target drugs

| Drug | Mechanism | Cancer type |
|---|---|---|
| Erlotinib | EGFR inhibitor | Lung |
| Lapatinib | EGFR/HER2 inhibitor | Breast |
| Sorafenib | RAF inhibitor | Liver/Kidney |
| Imatinib | BCR-ABL inhibitor | Leukemia |
| Paclitaxel | Microtubule stabiliser | Breast/Ovarian |

---

## Resume bullet

> Built protein–protein interaction GAT (PyTorch Geometric + STRING DB,
> ~200 nodes, ~12k edges) on CCLE/GDSC to predict cancer drug response (IC50);
> used cell-line-stratified splits to avoid data leakage; added MAPIE conformal
> prediction intervals (90% coverage); SHAP node attribution identified
> top resistance-driving proteins per drug — deployed on Hugging Face Spaces
