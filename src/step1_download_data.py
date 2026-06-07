"""
step1_download_data.py
======================
Downloads all raw data needed for the Drug Response GNN project.

Data sources (all free, no login required):
  - CCLE protein expression (RPPA):  DepMap portal
  - GDSC drug response (IC50):       GDSC2 release
  - CCLE cell line metadata:         DepMap portal

Run:
    python src/step1_download_data.py

Output:
    data/raw/ccle_proteins.csv        — protein expression per cell line
    data/raw/gdsc_drug_response.csv   — IC50 values per cell line × drug
    data/raw/ccle_metadata.csv        — cancer type per cell line
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── URLs ────────────────────────────────────────────────────────────────────
URLS = {
    # CCLE RPPA protein expression — 214 antibodies × ~900 cell lines
    "ccle_proteins.csv": (
        "https://depmap.org/portal/api/download/files?"
        "file_name=CCLE_RPPA_20181003.csv&release=DepMap+Public+22Q2"
    ),
    # Fallback direct link (DepMap sometimes changes URLs)
    "_ccle_proteins_fallback.csv": (
        "https://ndownloader.figshare.com/files/34989926"
    ),
    # GDSC2 drug sensitivity — IC50 for ~200 drugs × ~1000 cell lines
    "gdsc_drug_response.csv": (
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
        "GDSC2_fitted_dose_response_27Oct23.xlsx"
    ),
    # CCLE sample info — maps cell line name → cancer type (lineage)
    "ccle_metadata.csv": (
        "https://ndownloader.figshare.com/files/34989940"
    ),
}

# Curated list of drugs with good data coverage in GDSC2
TARGET_DRUGS = [
    "Erlotinib",    # EGFR inhibitor — lung cancer
    "Lapatinib",    # EGFR/HER2 inhibitor — breast cancer
    "Sorafenib",    # RAF inhibitor — liver/kidney cancer
    "Imatinib",     # BCR-ABL inhibitor — leukemia
    "Paclitaxel",   # Microtubule stabiliser — breast/ovarian
]


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=desc or dest.name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [error] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_ccle_proteins():
    """
    CCLE RPPA protein expression.
    Tries DepMap direct link first, falls back to figshare mirror.
    """
    dest = RAW_DIR / "ccle_proteins.csv"
    if dest.exists():
        print("  [skip] ccle_proteins.csv already exists")
        return

    print("\n→ Downloading CCLE protein expression (RPPA)...")

    # Primary: DepMap figshare
    primary_url = "https://ndownloader.figshare.com/files/34989926"
    if not download_file(primary_url, dest, "CCLE RPPA proteins"):
        # Fallback: public depmap S3 mirror
        fallback_url = (
            "https://depmap.org/portal/api/download/files?"
            "file_name=CCLE_RPPA_20181003.csv"
        )
        download_file(fallback_url, dest, "CCLE RPPA (fallback)")


def download_gdsc_response():
    """
    GDSC2 drug response IC50 values.
    Note: this is an xlsx file — we save it as .xlsx and convert in step 2.
    """
    dest = RAW_DIR / "gdsc_drug_response.xlsx"
    if dest.exists():
        print("  [skip] gdsc_drug_response.xlsx already exists")
        return

    print("\n→ Downloading GDSC2 drug response...")
    url = (
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
        "GDSC2_fitted_dose_response_27Oct23.xlsx"
    )
    if not download_file(url, dest, "GDSC2 IC50"):
        # Mirror on figshare
        mirror = "https://ndownloader.figshare.com/files/43757175"
        download_file(mirror, dest, "GDSC2 IC50 (mirror)")


def download_ccle_metadata():
    """Cell line metadata — cancer lineage, primary site."""
    dest = RAW_DIR / "ccle_metadata.csv"
    if dest.exists():
        print("  [skip] ccle_metadata.csv already exists")
        return

    print("\n→ Downloading CCLE cell line metadata...")
    url = "https://ndownloader.figshare.com/files/34989940"
    download_file(url, dest, "CCLE metadata")


def verify_downloads():
    """Quick sanity check on downloaded files."""
    import pandas as pd

    print("\n── Verification ───────────────────────────────────────")
    checks = {
        "ccle_proteins.csv": ("Cell lines (rows)", "Proteins (cols)"),
        "ccle_metadata.csv": ("Cell lines (rows)", "Columns"),
    }
    for fname, (row_label, col_label) in checks.items():
        path = RAW_DIR / fname
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, nrows=5)
                full = pd.read_csv(path, index_col=0)
                print(f"  {fname}: {row_label}={len(full)}, {col_label}={len(full.columns)}")
            except Exception as e:
                print(f"  {fname}: could not read — {e}")
        else:
            print(f"  {fname}: NOT FOUND")

    gdsc = RAW_DIR / "gdsc_drug_response.xlsx"
    if gdsc.exists():
        size_mb = gdsc.stat().st_size / 1e6
        print(f"  gdsc_drug_response.xlsx: {size_mb:.1f} MB")
    else:
        print("  gdsc_drug_response.xlsx: NOT FOUND")

    print(f"\nTarget drugs for this project: {TARGET_DRUGS}")
    print("─" * 55)


if __name__ == "__main__":
    print("=" * 55)
    print("  Drug Response GNN — Step 1: Data Download")
    print("=" * 55)
    download_ccle_proteins()
    download_gdsc_response()
    download_ccle_metadata()
    verify_downloads()
    print("\nDone. Run step2_preprocess.py next.")
