"""
step3_build_graph.py
====================
Constructs the protein–protein interaction (PPI) graph from the STRING database.

What we build:
  - Nodes  = proteins present in CCLE RPPA data (~200 proteins)
  - Edges  = known interactions from STRING with confidence ≥ 700
  - Node features = protein expression values (from step2, one vector per cell line)
  - Edge weights  = STRING combined confidence score (0–1000 → 0.0–1.0)

The graph is SHARED across all cell lines — what changes per cell line
is the node feature vector (expression values). The topology stays fixed.

Run:
    python src/step3_build_graph.py

Output:
    data/processed/string_edges.csv      — filtered PPI edge list
    data/processed/protein_index.json    — protein name → node index mapping
    data/processed/graph_stats.txt       — graph summary
"""

import json
import io
import gzip
import requests
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

STRING_MIN_SCORE = 700     # confidence threshold (out of 1000)
STRING_TAXON_ID  = 9606    # Homo sapiens


# ── 1. Get our protein list from CCLE data ──────────────────────────────────

def get_ccle_proteins() -> list[str]:
    """
    Read column names from the preprocessed protein expression matrix.
    These are HGNC gene symbols (e.g. EGFR, AKT1, TP53).
    
    CCLE RPPA antibodies often have suffixes like _pY1068 for phospho-sites.
    We strip those to get the base gene symbol for STRING lookup.
    """
    df = pd.read_csv(PROCESSED_DIR / "protein_expr.csv", index_col=0, nrows=0)
    raw_names = df.columns.tolist()

    # Strip phospho-site suffixes (e.g. "EGFR_pY1068" → "EGFR")
    # and isoform suffixes (e.g. "AKT1_p" → "AKT1")
    clean_names = []
    for name in raw_names:
        base = name.split("_")[0].split("-")[0].upper()
        clean_names.append(base)

    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in clean_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    print(f"  CCLE RPPA proteins: {len(raw_names)} antibodies → {len(unique_names)} unique gene symbols")
    return unique_names, raw_names


# ── 2. Map gene symbols to STRING protein IDs ───────────────────────────────

def map_to_string_ids(gene_symbols: list[str]) -> dict[str, str]:
    """
    Query STRING API to map HGNC gene symbols to STRING protein IDs.
    Returns {gene_symbol: string_id} for successfully mapped proteins.
    
    STRING IDs look like: "9606.ENSP00000275493"
    """
    print(f"\n  Mapping {len(gene_symbols)} gene symbols to STRING IDs...")

    url = "https://string-db.org/api/json/get_string_ids"
    params = {
        "identifiers": "\r".join(gene_symbols),
        "species":      STRING_TAXON_ID,
        "limit":        1,
        "echo_query":   1,
    }

    resp = requests.post(url, data=params, timeout=60)
    resp.raise_for_status()
    results = resp.json()

    mapping = {}
    for entry in results:
        query = entry.get("queryItem", "").upper()
        string_id = entry.get("stringId", "")
        if query and string_id and query not in mapping:
            mapping[query] = string_id

    print(f"  Mapped: {len(mapping)}/{len(gene_symbols)} proteins")
    unmapped = [g for g in gene_symbols if g not in mapping]
    if unmapped:
        print(f"  Unmapped (will be isolated nodes): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")

    return mapping


# ── 3. Download PPI interactions from STRING ─────────────────────────────────

def fetch_string_interactions(string_ids: list[str]) -> pd.DataFrame:
    """
    Fetch all pairwise interactions between our proteins from STRING API.
    Filters to combined_score ≥ STRING_MIN_SCORE.

    Returns DataFrame with columns:
        protein_a, protein_b, combined_score (0–1000)
    """
    print(f"\n  Fetching STRING interactions for {len(string_ids)} proteins...")
    print(f"  (minimum confidence score: {STRING_MIN_SCORE})")

    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers":    "%0d".join(string_ids),
        "species":        STRING_TAXON_ID,
        "required_score": STRING_MIN_SCORE,
        "network_type":   "functional",
    }

    resp = requests.post(url, data=params, timeout=120)
    resp.raise_for_status()
    interactions = resp.json()

    if not interactions:
        print("  No interactions returned — check your STRING IDs")
        return pd.DataFrame(columns=["protein_a", "protein_b", "score"])

    df = pd.DataFrame(interactions)

    # Rename columns to standard names
    df = df.rename(columns={
        "preferredName_A": "protein_a",
        "preferredName_B": "protein_b",
        "score":           "combined_score"
    })[["protein_a", "protein_b", "combined_score"]]

    df["protein_a"] = df["protein_a"].str.upper()
    df["protein_b"] = df["protein_b"].str.upper()

    print(f"  Interactions returned: {len(df)}")
    print(f"  Score distribution:")
    for threshold in [700, 800, 900]:
        n = (df["combined_score"] >= threshold).sum()
        print(f"    score ≥ {threshold}: {n} edges")

    return df


# ── 4. Build NetworkX graph ──────────────────────────────────────────────────

def build_networkx_graph(
    proteins: list[str],
    edges_df: pd.DataFrame
) -> nx.Graph:
    """
    Build undirected weighted graph.
    All CCLE proteins are nodes (even if isolated — no STRING edges).
    Edges are filtered PPI interactions, weighted by confidence / 1000.
    """
    G = nx.Graph()
    G.add_nodes_from(proteins)

    for _, row in edges_df.iterrows():
        a, b = row["protein_a"], row["protein_b"]
        if a in G and b in G and a != b:
            weight = row["combined_score"] / 1000.0
            G.add_edge(a, b, weight=weight)

    # Graph statistics
    degrees = [d for _, d in G.degree()]
    print(f"\n  Graph summary:")
    print(f"    Nodes:            {G.number_of_nodes()}")
    print(f"    Edges:            {G.number_of_edges()}")
    print(f"    Avg degree:       {np.mean(degrees):.1f}")
    print(f"    Max degree:       {max(degrees)}")
    print(f"    Isolated nodes:   {sum(1 for d in degrees if d == 0)}")
    print(f"    Connected components: {nx.number_connected_components(G)}")

    return G


# ── 5. Create protein→index mapping ─────────────────────────────────────────

def create_protein_index(proteins: list[str], G: nx.Graph) -> dict[str, int]:
    """
    Assigns a stable integer index to each protein node.
    Proteins are sorted alphabetically for reproducibility.
    This index is what PyTorch Geometric uses for node IDs.
    """
    sorted_proteins = sorted(G.nodes())
    idx_map = {p: i for i, p in enumerate(sorted_proteins)}
    return idx_map


# ── 6. Build PyG-ready edge index ────────────────────────────────────────────

def build_edge_index(G: nx.Graph, protein_idx: dict[str, int]) -> np.ndarray:
    """
    Convert NetworkX graph to PyTorch Geometric edge_index format.
    edge_index shape: [2, num_edges * 2] — both directions for undirected graph.
    Also returns edge_attr (weights) aligned to edge_index.
    """
    src, dst, weights = [], [], []
    for u, v, data in G.edges(data=True):
        i, j = protein_idx[u], protein_idx[v]
        w = data.get("weight", 1.0)
        # Both directions (undirected)
        src.extend([i, j])
        dst.extend([j, i])
        weights.extend([w, w])

    edge_index  = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    return edge_index, edge_weight


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Drug Response GNN — Step 3: Build PPI Graph")
    print("=" * 55)

    print("\n[1/5] Reading CCLE protein list...")
    gene_symbols, raw_column_names = get_ccle_proteins()

    print("\n[2/5] Mapping gene symbols to STRING IDs...")
    gene_to_string = map_to_string_ids(gene_symbols)

    print("\n[3/5] Fetching PPI interactions from STRING...")
    string_ids = list(gene_to_string.values())
    edges_df = fetch_string_interactions(string_ids)

    # Map STRING preferred names back to our gene symbols
    # STRING returns preferred names (usually same as HGNC) — we'll just use those
    edges_df.to_csv(PROCESSED_DIR / "string_edges.csv", index=False)
    print(f"  Saved: data/processed/string_edges.csv")

    print("\n[4/5] Building NetworkX graph...")
    G = build_networkx_graph(gene_symbols, edges_df)

    print("\n[5/5] Creating protein index and edge tensors...")
    protein_idx = create_protein_index(gene_symbols, G)
    edge_index, edge_weight = build_edge_index(G, protein_idx)

    # Save protein index
    with open(PROCESSED_DIR / "protein_index.json", "w") as f:
        json.dump(protein_idx, f, indent=2)

    # Save edge index as numpy arrays (loaded in step4 by PyG)
    np.save(PROCESSED_DIR / "edge_index.npy", edge_index)
    np.save(PROCESSED_DIR / "edge_weight.npy", edge_weight)

    # Save human-readable graph stats
    stats = {
        "n_nodes":              G.number_of_nodes(),
        "n_edges":              G.number_of_edges(),
        "avg_degree":           float(np.mean([d for _, d in G.degree()])),
        "string_min_score":     STRING_MIN_SCORE,
        "proteins":             sorted(G.nodes()),
        "n_proteins_in_ccle":   len(gene_symbols),
        "n_proteins_in_string": len([p for p in gene_symbols if G.degree(p) > 0]),
    }
    with open(PROCESSED_DIR / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved:")
    print(f"    protein_index.json  — {len(protein_idx)} proteins indexed")
    print(f"    edge_index.npy      — shape {edge_index.shape}")
    print(f"    edge_weight.npy     — shape {edge_weight.shape}")
    print(f"    graph_stats.json")

    # Visualise degree distribution
    try:
        import matplotlib.pyplot as plt
        degrees = [d for _, d in G.degree() if d > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(degrees, bins=30, color="#7F77DD", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Node degree (number of PPI connections)")
        ax.set_ylabel("Count")
        ax.set_title("Degree distribution of CCLE proteins in STRING PPI network")
        plt.tight_layout()
        plt.savefig(PROCESSED_DIR / "degree_distribution.png", dpi=120)
        print(f"    degree_distribution.png")
    except Exception:
        pass

    print(f"\nDone. Run step4_dataset.py next.")


if __name__ == "__main__":
    main()
