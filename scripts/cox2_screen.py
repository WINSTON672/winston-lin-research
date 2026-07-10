#!/usr/bin/env python3
"""
cox2_screen.py — COX-2 virtual screening pipeline.
Morgan fingerprint MLP trained on ChEMBL IC50 data, screened against ZINC250k.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

# Build PAINS filter once at module level
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(_pains_params)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).parent
CHEMBL_CSV  = SCRIPTS / "cox2_chembl.csv"
ZINC_CSV    = SCRIPTS / "zinc250k.csv"
OUT_DIR     = SCRIPTS / "cox2_results"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Fingerprint helper ─────────────────────────────────────────────────────

def to_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol:
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except Exception:
        pass
    return None


def fp_matrix(smiles_list):
    fps, idx = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc="fingerprints", leave=False)):
        fp = to_fp(smi)
        if fp is not None:
            fps.append(fp)
            idx.append(i)
    return np.array(fps, dtype=np.float32), idx


# ── 2. Load training data ─────────────────────────────────────────────────────

print("Loading ChEMBL COX-2 data...")
df = pd.read_csv(CHEMBL_CSV)
print(f"  {len(df)} compounds, pIC50 {df['pIC50'].min():.2f}–{df['pIC50'].max():.2f}")

print("Computing training fingerprints...")
X_all, valid_idx = fp_matrix(df["SMILES"].tolist())
y_all = df["pIC50"].iloc[valid_idx].values
print(f"  {len(X_all)} valid fingerprints")

# ── 3. Train/val/test split and model ────────────────────────────────────────

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.15, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15/0.85, random_state=42)

scaler_X = StandardScaler()
X_tr_s  = scaler_X.fit_transform(X_tr)
X_val_s = scaler_X.transform(X_val)
X_te_s  = scaler_X.transform(X_te)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(X_tr_s, y_tr)

y_pred_te = rf.predict(X_te_s)
r, _ = pearsonr(y_te, y_pred_te)
mse   = mean_squared_error(y_te, y_pred_te)

print(f"\nTest set  Pearson R = {r:.3f}  MSE = {mse:.3f}")

TRAIN_MAX_PIC50 = float(y_all.max())
print(f"  Training pIC50 max = {TRAIN_MAX_PIC50:.3f} (predictions capped here)")

# Predicted vs actual scatter
plt.figure(figsize=(5, 5))
plt.scatter(y_te, y_pred_te, alpha=0.4, s=10)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=1)
plt.xlabel("Measured pIC50"); plt.ylabel("Predicted pIC50")
plt.title(f"Test set  R={r:.3f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curve.png", dpi=150)
plt.close()
print(f"  scatter plot → {OUT_DIR/'loss_curve.png'}")

# ── 4. Load ZINC and download if needed ──────────────────────────────────────

if not ZINC_CSV.exists():
    print("\nDownloading ZINC250k...")
    import urllib.request
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    urllib.request.urlretrieve(url, ZINC_CSV)
    print(f"  saved → {ZINC_CSV}")

print("\nLoading ZINC250k...")
zinc = pd.read_csv(ZINC_CSV)
# column might be 'smiles' or 'SMILES'
smi_col = "smiles" if "smiles" in zinc.columns else "SMILES"
zinc = zinc.rename(columns={smi_col: "SMILES"})
print(f"  {len(zinc)} compounds")

# ── 5. Virtual screening ──────────────────────────────────────────────────────

print("\nComputing ZINC fingerprints...")
X_zinc, zinc_idx = fp_matrix(zinc["SMILES"].tolist())
X_zinc_s = scaler_X.transform(X_zinc)

print("Screening...")
preds = rf.predict(X_zinc_s)

# RF prediction uncertainty: std of individual tree predictions
tree_preds = np.array([t.predict(X_zinc_s) for t in rf.estimators_])
pred_std = tree_preds.std(axis=0)

zinc_valid = zinc.iloc[zinc_idx].copy().reset_index(drop=True)
zinc_valid["pIC50_pred"] = np.clip(preds, None, TRAIN_MAX_PIC50)
zinc_valid["pIC50_std"]  = pred_std

# ── 6. Property filters ───────────────────────────────────────────────────────

def props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mw    = Descriptors.MolWt(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    lp    = Descriptors.MolLogP(mol)
    q     = qed(mol)
    pains = PAINS_CATALOG.HasMatch(mol)
    return mw, hbd, hba, lp, q, pains

print("Computing properties + PAINS...")
prop_rows = []
for smi in tqdm(zinc_valid["SMILES"], desc="props", leave=False):
    prop_rows.append(props(smi.strip()))

zinc_valid[["MW", "HBD", "HBA", "logP", "QED", "PAINS"]] = prop_rows

n_before = (
    (zinc_valid["MW"]  <= 500) &
    (zinc_valid["HBD"] <= 5)   &
    (zinc_valid["HBA"] <= 10)  &
    (zinc_valid["logP"] <= 5)  &
    (zinc_valid["logP"] >= -1) &
    (zinc_valid["logP"] <= 3)  &
    (zinc_valid["pIC50_pred"] > 7.2)
).sum()

mask = (
    (zinc_valid["MW"]  <= 500) &
    (zinc_valid["HBD"] <= 5)   &
    (zinc_valid["HBA"] <= 10)  &
    (zinc_valid["logP"] <= 5)  &
    (zinc_valid["logP"] >= -1) &
    (zinc_valid["logP"] <= 3)  &
    (zinc_valid["pIC50_pred"] > 7.2) &
    (~zinc_valid["PAINS"])
)
print(f"  Before PAINS removal: {n_before}  After: {mask.sum()}")

candidates = zinc_valid[mask].sort_values("pIC50_pred", ascending=False).reset_index(drop=True)
print(f"\nCandidates passing all filters: {len(candidates)}")

candidates.to_csv(OUT_DIR / "candidates.csv", index=False)
print(f"  saved → {OUT_DIR/'candidates.csv'}")

if len(candidates):
    top = candidates.iloc[0]
    print(f"\nTop candidate:")
    print(f"  SMILES:  {top['SMILES'].strip()}")
    print(f"  pIC50:   {top['pIC50_pred']:.3f} ± {top['pIC50_std']:.3f}")
    print(f"  logP:    {top['logP']:.3f}")
    print(f"  QED:     {top['QED']:.3f}")
    print(f"\nTop 10:")
    cols = ["SMILES", "pIC50_pred", "pIC50_std", "logP", "QED"]
    print(candidates[cols].head(10).to_string())

# ── 7. Molecule grid figure ───────────────────────────────────────────────────

try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    import io

    top12 = candidates.head(12)
    mols  = [Chem.MolFromSmiles(s) for s in top12["SMILES"]]
    mols  = [m for m in mols if m]

    img = Draw.MolsToGridImage(
        mols, molsPerRow=4, subImgSize=(400, 300),
        legends=[f"pIC50={p:.2f}" for p in top12["pIC50_pred"][:len(mols)]]
    )
    img.save(OUT_DIR / "figure2_molecules.png")
    print(f"\nMolecule grid → {OUT_DIR/'figure2_molecules.png'}")
except Exception as e:
    print(f"Grid image skipped: {e}")

print("\nDone.")
