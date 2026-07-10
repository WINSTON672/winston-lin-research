#!/usr/bin/env python3
"""Recompute the model-comparison table (Table 1) on corrected CHEMBL230 data.
Same protocol as main.tex: 2048-bit Morgan r2, z-score, 70/15/15 split (seed 42),
500-tree RF/XGB, report held-out TEST metrics."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor

SCRIPTS = Path("/Users/winstonlin/Desktop/scripts")

def fps(smiles):
    X, idx = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s).strip())
        if m:
            X.append(list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))); idx.append(i)
    return np.array(X, np.float32), idx

def evaluate(csv, label):
    df = pd.read_csv(csv)
    X, idx = fps(df["SMILES"].tolist()); y = df["pIC50"].iloc[idx].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.15/0.85, random_state=42)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    out = {"label": label, "n_total": len(y), "n_test": len(yte)}
    # mean baseline
    yp = np.full_like(yte, ytr.mean()); out["mean"] = (np.nan, mean_squared_error(yte, yp))
    for name, mdl in [("ridge", Ridge()),
                      ("xgb", XGBRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
                      ("rf", RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42))]:
        mdl.fit(Xtr_s, ytr); yp = mdl.predict(Xte_s)
        out[name] = (pearsonr(yte, yp)[0], mean_squared_error(yte, yp))
    return out

res = {}
for csv, lab in [(SCRIPTS/"cox2_chembl.csv", "COX-2"), (SCRIPTS/"cox1_chembl.csv", "COX-1")]:
    r = evaluate(csv, lab); res[lab] = r
    print(f"\n=== {lab}  (n_total={r['n_total']}, n_test={r['n_test']}) ===")
    for k in ("mean", "ridge", "xgb", "rf"):
        R, mse = r[k]; rmse = mse**0.5
        print(f"  {k:6s}  R={R if isinstance(R,float) and not np.isnan(R) else 'NA':<6}  MSE={mse:.3f}  RMSE={rmse:.3f}"
              if not (isinstance(R,float) and np.isnan(R)) else f"  {k:6s}  R=---     MSE={mse:.3f}  RMSE={rmse:.3f}")
Path("/private/tmp/claude-501/-Users-winstonlin/782364e6-dbdf-4c5e-af88-46ab8dded67b/scratchpad/cox2_rerun/model_table.json").write_text(
    json.dumps({k: {m: list(v[m]) if isinstance(v[m], tuple) else v[m] for m in v} for k, v in res.items()}, indent=2, default=str))
print("\nDONE_MODEL_TABLE")
