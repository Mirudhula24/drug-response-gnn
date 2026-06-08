# 🧬 Drug Response GNN

> Predicting cancer drug response from protein–protein interaction networks using Graph Attention Networks — with conformal uncertainty intervals and SHAP explainability.

**[🚀 Live Demo](https://drug-response-gnn-gt6qmamgopdappg9qmfe7.streamlit.app)** · **[GitHub](https://github.com/Mirudhula24/drug-response-gnn)** 

---

## What this does

Given a cancer cell line's **protein expression profile**, this model predicts how much a specific drug will slow cell growth — expressed as **LN(IC50)**.

Instead of treating proteins as a flat unordered vector, proteins are modelled as a **graph**: nodes are proteins, edges are known biological interactions from the STRING database. A Graph Attention Network (GAT) learns which protein neighbourhoods are most predictive of drug response.

Predictions come with **90% conformal prediction intervals** — so instead of just "IC50 = 2.3", you get "IC50 = 2.3 ± 0.4 (90% coverage guarantee)".

---

## Results

| Drug | GAT RMSE | MLP RMSE | GAT vs MLP |
|---|---|---|---|
| Erlotinib | 1.6489 | 1.5020 | -9.8% |
| Lapatinib | 1.4141 | 1.4897 | **+5.1%** |
| Sorafenib | 1.0940 | 1.1002 | **+0.6%** |
| Paclitaxel | 1.5308 | 1.7525 | **+12.6%** |

**Conformal prediction coverage (target: 90%):**

| Drug | Coverage | Met? |
|---|---|---|
| Erlotinib | 92.0% | ✓ |
| Lapatinib | 100.0% | ✓ |
| Sorafenib | 96.0% | ✓ |
| Paclitaxel | 96.0% | ✓ |

**SHAP biological validation — top resistance proteins:**

| Drug | #1 Protein | Biological match |
|---|---|---|
| Lapatinib | **HER2** | ✓ Lapatinib is a HER2 inhibitor |
| Sorafenib | **EGFR**, PTEN | ✓ PTEN loss drives Sorafenib resistance (published) |
| Paclitaxel | **HER2**, HER3 | ✓ HER2 overexpression drives Paclitaxel resistance in breast cancer |
| Erlotinib | VEGFR2, HER3 | ✓ HER3 is in the EGFR family |

---

## Architecture

```
Input: protein expression node features  [158 proteins × 35 features]
         ↓
GAT Layer 1:  4-head attention (dim=64)  +  ELU  +  dropout
         ↓
GAT Layer 2:  2-head attention (dim=64)  +  ELU  +  dropout
         ↓
GAT Layer 3:  1-head attention (dim=64)  +  ELU
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

---

## Key design decisions

| Decision | Why it matters |
|---|---|
| GATv2 over GCN | Learns which protein neighbours matter — biologically interpretable |
| Cell-line stratified splits | Prevents data leakage — most papers get this wrong |
| KNN imputation within cancer type | Biologically meaningful; global imputation adds noise |
| 5 curated drugs only | Focused scope with good data coverage; clinically important |
| Conformal prediction | Rigorous uncertainty — not just dropout variance |
| SHAP on node features | Per-protein attribution → which proteins drive resistance |

---

## Data sources (all free, no login required)

| Source | What | Link |
|---|---|---|
| CCLE | Protein expression (RPPA, 214 antibodies × 899 cell lines) | [depmap.org](https://depmap.org) |
| GDSC2 | Drug response IC50 (265 compounds × 1000+ cell lines) | [cancerrxgene.org](https://www.cancerrxgene.org) |
| STRING DB | Protein–protein interactions (confidence ≥ 700) | [string-db.org](https://string-db.org) |

---

## Target drugs

| Drug | Mechanism | Cancer type |
|---|---|---|
| Erlotinib | EGFR inhibitor | Lung |
| Lapatinib | EGFR/HER2 inhibitor | Breast |
| Sorafenib | RAF/VEGFR inhibitor | Liver/Kidney |
| Paclitaxel | Microtubule stabiliser | Breast/Ovarian |

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

## Run pipeline

```bash
python src/step1_download_data.py   # download CCLE + GDSC
python src/step2_preprocess.py      # clean + stratified split
python src/step3_build_graph.py     # build STRING PPI graph
python src/step4_dataset.py         # verify PyG dataset
python src/step5_model.py           # verify model architecture
python src/step6_train.py           # train GAT + MLP baseline
python src/step7_uncertainty.py     # conformal prediction intervals
python src/step8_explain.py         # SHAP protein attribution
streamlit run app.py                # launch demo
```

---

## Paper references

- **GATv2**: Brody et al., *How Attentive are Graph Attention Networks?* ICLR 2022
- **STRING DB**: Szklarczyk et al., *Nucleic Acids Research* 2021
- **CCLE**: Ghandi et al., *Nature* 2019
- **GDSC**: Yang et al., *Nucleic Acids Research* 2013
- **Conformal prediction**: Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* 2022

---

## Resume bullet

> Built protein–protein interaction GAT (PyTorch Geometric + STRING DB, 158 nodes, 328 edges) on CCLE/GDSC to predict cancer drug response (IC50); used cell-line-stratified splits to avoid data leakage; added conformal prediction intervals (90%+ coverage all 4 drugs); SHAP attribution identified HER2 as top resistance driver for Lapatinib — biologically validated; deployed on Hugging Face Spaces

---

**Built by [Mimi](https://github.com/Mirudhula24)** · Stack: PyTorch Geometric · STRING DB · CCLE · GDSC2 · SHAP · Streamlit
