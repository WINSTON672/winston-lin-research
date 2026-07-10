#!/usr/bin/env python3
"""
cox2_v3_contacts.py — per-residue contact analysis + celecoxib RMSD (v3).

For 9 chain-A pocket residues, compute min heavy-atom distance from each docked
pose; contact declared <= 4.0 A. Compare top-20 Vina candidates vs 20 random ZINC
vs celecoxib. Writes contact_analysis.csv into papers/cox2_journal/ and appends
celecoxib_pose_rmsd to final_stats.json.
"""
import json, subprocess, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SUP = HERE / "v3_support"; POSES = SUP / "poses"
CANDPOSE = HERE / "vina_out"
PAPER = Path("/Users/winstonlin/Desktop/research/papers/cox2_journal")
RECEPTOR_PDB = HERE / "receptor_protein_chainA.pdb"
CONTACT_CUT = 4.0
CELECOXIB_SMILES = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"  # celecoxib (CHEMBL118)
RESIDUES = {  # (resname, resSeq) chain A
    "Arg120": ("ARG", 120), "Tyr355": ("TYR", 355), "Leu384": ("LEU", 384),
    "Tyr385": ("TYR", 385), "Trp387": ("TRP", 387), "Phe518": ("PHE", 518),
    "Met522": ("MET", 522), "Val523": ("VAL", 523), "Ser530": ("SER", 530),
}

def is_heavy(el):
    return el.strip().upper() not in ("H", "HD", "HS")

# receptor residue heavy-atom coords
res_coords = {k: [] for k in RESIDUES}
name2key = {v: k for k, v in RESIDUES.items()}
for ln in RECEPTOR_PDB.read_text().splitlines():
    if ln.startswith(("ATOM", "HETATM")):
        resn = ln[17:20].strip()
        try: resseq = int(ln[22:26])
        except ValueError: continue
        key = name2key.get((resn, resseq))
        if key is None: continue
        el = ln[76:78].strip() or ln[12:16].strip()[0]
        if not is_heavy(el): continue
        res_coords[key].append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
res_coords = {k: np.array(v) for k, v in res_coords.items() if len(v)}
print("receptor residues found:", {k: len(v) for k, v in res_coords.items()})

def pose_heavy_coords(pdbqt_path):
    """First MODEL heavy-atom coords from a Vina output PDBQT."""
    coords = []
    for ln in Path(pdbqt_path).read_text().splitlines():
        if ln.startswith("ENDMDL"): break
        if ln.startswith(("ATOM", "HETATM")):
            adtype = ln[77:79].strip()
            if not is_heavy(adtype): continue
            coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.array(coords)

def contacts(pose_xyz):
    out = {}
    for res, rc in res_coords.items():
        d = np.sqrt(((pose_xyz[:, None, :] - rc[None, :, :])**2).sum(-1)).min()
        out[res] = (d <= CONTACT_CUT, float(d))
    return out

# celecoxib contacts define the "key" residue set
cele_xyz = pose_heavy_coords(POSES / "celecoxib_out.pdbqt")
cele_c = contacts(cele_xyz)
cele_key = [r for r, (hit, _) in cele_c.items() if hit]
print("celecoxib contacts (<=4A):", cele_key)

def analyze_set(pose_files, label):
    rows = []
    for pf in pose_files:
        xyz = pose_heavy_coords(pf)
        if len(xyz) == 0: continue
        c = contacts(xyz)
        val523 = c["Val523"][0]
        recov = np.mean([c[r][0] for r in cele_key]) if cele_key else 0.0
        min_arg120 = c["Arg120"][1]
        rows.append({"set": label, "pose": Path(pf).stem, "val523": int(val523),
                     "recovery": recov, "min_arg120": min_arg120})
    return pd.DataFrame(rows)

# top-20 candidates by Vina score -> map SMILES to pose index in candidates_enriched
enr = pd.read_csv(HERE / "candidates_enriched.csv").reset_index(drop=True)
enr["SMILES"] = enr["SMILES"].str.strip()
dock = pd.read_csv(HERE / "results" / "docking_results.csv")
dock = dock[dock.dock_status == "ok"].copy()
dock["SMILES"] = dock["SMILES"].str.strip()
smi2idx = {s: i for i, s in enumerate(enr["SMILES"])}
dock["pose_idx"] = dock["SMILES"].map(smi2idx)
dock = dock.dropna(subset=["pose_idx"]).sort_values("vina_kcal_mol")
top20 = dock.head(20)
cand_files = [CANDPOSE / f"cand_{int(i):04d}_out.pdbqt" for i in top20["pose_idx"]]
cand_files = [f for f in cand_files if f.exists()]

rand_files = sorted(POSES.glob("rand_0[0-1]*_out.pdbqt"))[:20]

cand_df = analyze_set(cand_files, "ML_top20")
rand_df = analyze_set(rand_files, "random_zinc20")
allc = pd.concat([cand_df, rand_df], ignore_index=True)
allc.to_csv(PAPER / "contact_analysis.csv", index=False)

summary = {}
for label, df in [("ML_top20", cand_df), ("random_zinc20", rand_df)]:
    summary[label] = {"n": int(len(df)), "val523_contacts": int(df["val523"].sum()),
                      "mean_recovery": float(df["recovery"].mean()),
                      "n_far_from_arg120_gt12A": int((df["min_arg120"] > 12).sum())}
print("\nCONTACT SUMMARY:"); print(json.dumps(summary, indent=2))

# ── celecoxib RMSD to crystal S58 ─────────────────────────────────────────────
def crystal_mol():
    het = [ln for ln in (HERE/"1CX2.pdb").read_text().splitlines()
           if ln.startswith("HETATM") and ln[21] == "A" and ln[17:20].strip() == "S58"]
    block = "\n".join(het) + "\nEND\n"
    m = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
    return m
rmsd = None
try:
    tmpl = Chem.MolFromSmiles(CELECOXIB_SMILES)
    cry = crystal_mol()
    cry = AllChem.AssignBondOrdersFromTemplate(tmpl, cry)
    # docked pose pdbqt -> pdb via obabel
    pdb_out = SUP / "celecoxib_docked.pdb"
    subprocess.run(["obabel", str(POSES/"celecoxib_out.pdbqt"), "-O", str(pdb_out),
                    "-f", "1", "-l", "1"], capture_output=True, text=True)
    dk = Chem.MolFromPDBBlock(pdb_out.read_text(), removeHs=True, sanitize=False)
    dk = AllChem.AssignBondOrdersFromTemplate(tmpl, dk)
    rmsd = float(AllChem.GetBestRMS(Chem.RemoveHs(dk), Chem.RemoveHs(cry)))
    print(f"\ncelecoxib pose RMSD to crystal: {rmsd:.3f} A")
except Exception as e:
    print(f"\n[warn] RMSD computation failed: {e}")

# append to final_stats.json
fs = json.loads((PAPER/"final_stats.json").read_text())
fs["celecoxib_pose_rmsd"] = rmsd
fs["contacts"] = summary
fs["celecoxib_key_residues"] = cele_key
(PAPER/"final_stats.json").write_text(json.dumps(fs, indent=2))
print("\nDONE_V3_CONTACTS")
