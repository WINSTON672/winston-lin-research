#!/usr/bin/env python3
"""
cox2_additions.py — Three additions to make the COX-2 paper publishable:
  1. COX-1 selectivity model (COX-2 pIC50 - COX-1 pIC50 for each candidate)
  2. Scaffold analysis (Murcko scaffolds, top chemical series)
  3. Applicability domain (Tanimoto similarity to training set)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS   = Path(__file__).parent
CHEMBL2   = SCRIPTS / "cox2_chembl.csv"
COX1_CSV  = SCRIPTS / "cox1_chembl.csv"
CAND_CSV  = SCRIPTS / "cox2_results" / "candidates.csv"
OUT_DIR   = SCRIPTS / "cox2_results"
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def to_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        pass
    return None

def fp_matrix(smiles_list, desc="fps"):
    fps, idx = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, leave=False)):
        fp = to_fp(smi)
        if fp is not None:
            fps.append(list(fp))
            idx.append(i)
    return np.array(fps, dtype=np.float32), idx

def fp_objects(smiles_list, desc="fps"):
    """Return rdkit ExplicitBitVect objects (for Tanimoto)."""
    fps, idx = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, leave=False)):
        fp = to_fp(smi)
        if fp is not None:
            fps.append(fp)
            idx.append(i)
    return fps, idx

def train_rf(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15/0.85, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X_tr_s, y_tr)
    y_pred = rf.predict(X_te_s)
    r, _ = pearsonr(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    print(f"  R={r:.3f}  MSE={mse:.3f}  RMSE={mse**0.5:.3f}")
    return rf, scaler, X_tr

# ── 1. Retrain COX-2 model ────────────────────────────────────────────────────

print("=== COX-2 model ===")
df2 = pd.read_csv(CHEMBL2)
print(f"  {len(df2)} compounds")
X2, idx2 = fp_matrix(df2["SMILES"].tolist(), "cox2-fps")
y2 = df2["pIC50"].iloc[idx2].values
rf2, scaler2, X2_train = train_rf(X2, y2)
print(f"  training set size: {len(X2_train)}")

# ── 2. COX-1 model ────────────────────────────────────────────────────────────

print("\n=== COX-1 model ===")

if not COX1_CSV.exists():
    print("  downloading COX-1 data from ChEMBL (CHEMBL221)...")
    from chembl_webresource_client.new_client import new_client
    activity = new_client.activity
    records = activity.filter(
        target_chembl_id="CHEMBL221",  # COX-1 / Prostaglandin G/H synthase 1
        standard_type="IC50",
        relation="=",
        assay_type="B",
    ).only(["canonical_smiles", "standard_value", "standard_units", "pchembl_value"])
    rows = []
    for r in tqdm(records, desc="chembl-cox1"):
        smi = r.get("canonical_smiles")
        pval = r.get("pchembl_value")
        if smi and pval:
            try:
                rows.append({"SMILES": smi, "pIC50": float(pval)})
            except (ValueError, TypeError):
                pass
    df1 = pd.DataFrame(rows).drop_duplicates("SMILES")
    df1 = df1[df1["pIC50"].between(3, 12)]
    df1.to_csv(COX1_CSV, index=False)
    print(f"  saved {len(df1)} compounds → {COX1_CSV}")
else:
    df1 = pd.read_csv(COX1_CSV)
    print(f"  loaded {len(df1)} compounds from cache")

X1, idx1 = fp_matrix(df1["SMILES"].tolist(), "cox1-fps")
y1 = df1["pIC50"].iloc[idx1].values
rf1, scaler1, _ = train_rf(X1, y1)

# ── 3. Predict on candidates ──────────────────────────────────────────────────

print("\n=== Predicting selectivity on candidates ===")
cand = pd.read_csv(CAND_CSV)
cand["SMILES"] = cand["SMILES"].str.strip()

cand_fps, cand_idx = fp_matrix(cand["SMILES"].tolist(), "cand-fps")
cand_valid = cand.iloc[cand_idx].copy().reset_index(drop=True)

cand_valid["pIC50_cox2"] = rf2.predict(scaler2.transform(cand_fps))
cand_valid["pIC50_cox1"] = rf1.predict(scaler1.transform(cand_fps))
cand_valid["selectivity"] = cand_valid["pIC50_cox2"] - cand_valid["pIC50_cox1"]

print(f"  Candidates with selectivity > 0: {(cand_valid['selectivity'] > 0).sum()}")
print(f"  Candidates with selectivity > 1: {(cand_valid['selectivity'] > 1).sum()}")
print(f"  Mean selectivity: {cand_valid['selectivity'].mean():.3f}")

# Selectivity scatter plot
plt.figure(figsize=(5, 5))
plt.scatter(cand_valid["pIC50_cox2"], cand_valid["pIC50_cox1"],
            c=cand_valid["selectivity"], cmap="RdYlGn", s=30, alpha=0.8)
plt.colorbar(label="Selectivity (COX-2 − COX-1 pIC50)")
plt.axline((7.2, 7.2), slope=1, color="gray", ls="--", lw=0.8)
plt.xlabel("Predicted COX-2 pIC50")
plt.ylabel("Predicted COX-1 pIC50")
plt.title("COX-2 vs. COX-1 predicted activity")
plt.tight_layout()
plt.savefig(OUT_DIR / "selectivity_scatter.png", dpi=150)
plt.close()
print(f"  selectivity scatter → {OUT_DIR/'selectivity_scatter.png'}")

# Selectivity histogram
plt.figure(figsize=(5, 3.5))
plt.hist(cand_valid["selectivity"], bins=25, color="steelblue", edgecolor="white", lw=0.4)
plt.axvline(0, color="red", ls="--", lw=1)
plt.xlabel("Selectivity (COX-2 − COX-1 pIC50)")
plt.ylabel("Count")
plt.title("Predicted selectivity distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "selectivity_hist.png", dpi=150)
plt.close()

# ── 4. Applicability domain ───────────────────────────────────────────────────

print("\n=== Applicability domain ===")

# training set bit vectors
train_smiles = df2["SMILES"].iloc[idx2].tolist()
# use a random subsample of 500 training compounds to keep it fast
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(train_smiles), size=min(500, len(train_smiles)), replace=False)
train_bvs, _ = fp_objects([train_smiles[i] for i in sample_idx], "train-ad")

cand_bvs = []
for smi in tqdm(cand_valid["SMILES"], desc="cand-ad"):
    fp = to_fp(smi)
    cand_bvs.append(fp)

max_tanimoto = []
for cfp in tqdm(cand_bvs, desc="tanimoto"):
    if cfp is None:
        max_tanimoto.append(0.0)
        continue
    sims = DataStructs.BulkTanimotoSimilarity(cfp, train_bvs)
    max_tanimoto.append(max(sims))

cand_valid["max_tanimoto"] = max_tanimoto

# Threshold: in-domain if max Tanimoto >= 0.4
AD_THRESHOLD = 0.4
cand_valid["in_domain"] = cand_valid["max_tanimoto"] >= AD_THRESHOLD
n_in = cand_valid["in_domain"].sum()
print(f"  In-domain (Tanimoto ≥ {AD_THRESHOLD}): {n_in}/{len(cand_valid)} ({100*n_in/len(cand_valid):.1f}%)")

# Tanimoto distribution
plt.figure(figsize=(5, 3.5))
plt.hist(cand_valid["max_tanimoto"], bins=25, color="darkorange", edgecolor="white", lw=0.4)
plt.axvline(AD_THRESHOLD, color="red", ls="--", lw=1, label=f"threshold = {AD_THRESHOLD}")
plt.xlabel("Max Tanimoto similarity to training set")
plt.ylabel("Count")
plt.title("Applicability domain")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "applicability_domain.png", dpi=150)
plt.close()
print(f"  AD histogram → {OUT_DIR/'applicability_domain.png'}")

# ── 5. Scaffold analysis ──────────────────────────────────────────────────────

print("\n=== Scaffold analysis ===")

scaffolds = []
for smi in cand_valid["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf))
        except Exception:
            scaffolds.append("")
    else:
        scaffolds.append("")

cand_valid["scaffold"] = scaffolds
scaf_counts = cand_valid["scaffold"].value_counts()

print(f"  Unique scaffolds: {(scaf_counts > 0).sum()}")
print(f"  Singletons: {(scaf_counts == 1).sum()}")
print(f"  Top 5 scaffolds:")
for scaf, cnt in scaf_counts.head(5).items():
    print(f"    n={cnt:3d}  {scaf[:80]}")

# Scaffold size distribution
plt.figure(figsize=(5, 3.5))
scaf_sizes = scaf_counts.values
plt.bar(range(1, min(16, len(scaf_sizes)+1)), scaf_sizes[:15], color="mediumpurple", edgecolor="white")
plt.xlabel("Scaffold rank (by frequency)")
plt.ylabel("Number of candidates")
plt.title("Top 15 Murcko scaffold series")
plt.tight_layout()
plt.savefig(OUT_DIR / "scaffold_distribution.png", dpi=150)
plt.close()
print(f"  scaffold plot → {OUT_DIR/'scaffold_distribution.png'}")

# ── 6. Save enriched candidate list ─────────────────────────────────────────

cand_valid.to_csv(OUT_DIR / "candidates_enriched.csv", index=False)
print(f"\nEnriched candidates → {OUT_DIR/'candidates_enriched.csv'}")

# ── 7. Summary stats for paper ───────────────────────────────────────────────

print("\n=== Summary for paper ===")
n = len(cand_valid)
print(f"Total candidates analyzed: {n}")
print(f"Unique Murcko scaffolds: {scaf_counts.notna().sum()} (singletons: {(scaf_counts==1).sum()})")
print(f"Top scaffold count: {scaf_counts.iloc[0]} compounds")
sel = cand_valid["selectivity"]
print(f"Selectivity: mean={sel.mean():.2f}, median={sel.median():.2f}, >0: {(sel>0).sum()}, >1: {(sel>1).sum()}")
mt = cand_valid["max_tanimoto"]
print(f"Max Tanimoto: mean={mt.mean():.3f}, median={mt.median():.3f}, in-domain: {n_in}/{n}")

# Top 5 by selectivity
print("\nTop 5 by selectivity (>0, in-domain preferred):")
top_sel = cand_valid[cand_valid["selectivity"] > 0].sort_values("selectivity", ascending=False).head(5)
print(top_sel[["SMILES", "pIC50_pred", "pIC50_cox2", "pIC50_cox1", "selectivity", "max_tanimoto"]].to_string())

print("\nDone.")
