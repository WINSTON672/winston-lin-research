#!/usr/bin/env python3
"""
pull_chembl.py — Pull COX-2 (CHEMBL230) and COX-1 (CHEMBL221) IC50 bioactivity
from ChEMBL.

Recipe mirrors the paper's stated methodology (and the inline COX-1 pull in
cox2_additions.py): standard_type IC50, relation '=', assay_type 'B',
units nM, dedupe by lowest IC50 per SMILES, restrict pIC50 in [3, 12].
Output columns match cox2_screen.py's expectation: SMILES, IC50_nM, pIC50.
"""
import sys, math
import numpy as np
import pandas as pd
from pathlib import Path
from chembl_webresource_client.new_client import new_client

SCRIPTS = Path(__file__).parent
TARGETS = {
    "cox2_chembl.csv": ("CHEMBL230", "COX-2 / Prostaglandin G/H synthase 2"),
    "cox1_chembl.csv": ("CHEMBL221", "COX-1 / Prostaglandin G/H synthase 1"),
}

def pull(cid):
    activity = new_client.activity
    recs = activity.filter(
        target_chembl_id=cid,
        standard_type="IC50",
        relation="=",
        assay_type="B",
    ).only(["canonical_smiles", "standard_value", "standard_units", "pchembl_value"])
    rows = []
    for r in recs:
        smi = r.get("canonical_smiles")
        val = r.get("standard_value")
        unit = r.get("standard_units")
        if not smi or val is None or unit != "nM":
            continue
        try:
            nm = float(val)
        except (ValueError, TypeError):
            continue
        if nm <= 0:
            continue
        rows.append({"SMILES": smi, "IC50_nM": nm})
    return pd.DataFrame(rows)

def main():
    for fname, (cid, label) in TARGETS.items():
        print(f"\n=== {label}  ({cid}) ===", flush=True)
        df = pull(cid)
        print(f"  raw IC50=nM records: {len(df)}", flush=True)
        # dedupe by SMILES keeping lowest IC50 (highest potency)
        df = df.sort_values("IC50_nM").drop_duplicates("SMILES", keep="first")
        df["pIC50"] = 9.0 - np.log10(df["IC50_nM"])
        df = df[df["pIC50"].between(3, 12)]
        df = df[["SMILES", "IC50_nM", "pIC50"]].reset_index(drop=True)
        out = SCRIPTS / fname
        df.to_csv(out, index=False)
        print(f"  final unique compounds (pIC50 3-12): {len(df)}", flush=True)
        print(f"  pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}", flush=True)
        # chemistry sanity: benzamidine/guanidine (protease) vs sulfonamide/CF3 (coxib)
        s = df["SMILES"].astype(str)
        prot = s.str.contains(r"C\(=N\)N|N=C\(N\)|C\(N\)=N", regex=True).sum()
        sulf = s.str.contains(r"S\(N\)\(=O\)=O|S\(=O\)\(=O\)N", regex=True).sum()
        cf3  = s.str.contains(r"C\(F\)\(F\)F", regex=True).sum()
        n = len(df)
        print(f"  benzamidine/guanidine (protease motif): {prot}/{n} ({100*prot/n:.1f}%)", flush=True)
        print(f"  primary sulfonamide: {sulf}/{n} ({100*sulf/n:.1f}%)", flush=True)
        print(f"  trifluoromethyl: {cf3}/{n} ({100*cf3/n:.1f}%)", flush=True)
    print("\nDONE_PULL", flush=True)

if __name__ == "__main__":
    main()
