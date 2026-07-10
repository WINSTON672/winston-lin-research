#!/usr/bin/env python3
"""
cox2_v3_docking.py — supporting docking for the corrected (v3) COX-2 audit.

Docks, into the SAME chain-A 1CX2 celecoxib pocket used for the 42 candidates:
  1. Celecoxib redock (positive control) + heavy-atom RMSD to crystal S58
  2. The 100-compound random ZINC baseline (fresh, consistent Vina run; poses saved)
  3. Vina-on-training validation: ChEMBL COX-2 actives (pIC50>=7) & inactives (pIC50<=5)

Poses are written so per-residue contact analysis can read them.
Uses the same box + settings as dock_pipeline.py (exhaustiveness 8, num_modes 5,
22 A cubic box centered on the chain-A celecoxib heavy-atom centroid).
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SCRIPTS = Path("/Users/winstonlin/Desktop/scripts")
VINA = os.path.expanduser("~/.local/bin/vina")
RECEPTOR = HERE / "receptor.pdbqt"
PDB = HERE / "1CX2.pdb"
BASELINE_CSV = Path("/Users/winstonlin/Desktop/research/papers/cox2_journal/random_zinc_baseline.csv")
COX2_CHEMBL = SCRIPTS / "cox2_chembl.csv"

OUTDIR = HERE / "v3_support"; OUTDIR.mkdir(exist_ok=True)
POSEDIR = OUTDIR / "poses"; POSEDIR.mkdir(exist_ok=True)

CHAIN = "A"; RESN = "S58"; BOX = 22.0; EXH = 8; NMODES = 5
CELECOXIB_SMILES = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"  # celecoxib (CHEMBL118)

# ── box center from chain-A celecoxib crystal centroid ────────────────────────
def crystal_ligand_lines():
    lines = []
    for ln in PDB.read_text().splitlines():
        if ln.startswith("HETATM") and ln[21] == CHAIN and ln[17:20].strip() == RESN:
            lines.append(ln)
    return lines

lig_lines = crystal_ligand_lines()
coords = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lig_lines])
CENTER = coords.mean(axis=0)
print(f"[box] center {CENTER.round(2).tolist()}  (n={len(coords)} crystal S58 heavy atoms)")

# ── ligand prep ───────────────────────────────────────────────────────────────
def smi_to_pdbqt(smi, out_path):
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None: return False
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50) != 0: return False
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception: return False
    try:
        prep = MoleculePreparation(); prep.prepare(mol)
        setups = prep.mol_setups if hasattr(prep, "mol_setups") else [prep.setup]
        s, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not ok: return False
        out_path.write_text(s); return True
    except Exception: return False

def dock_one(name, smi):
    lig = POSEDIR / f"{name}.pdbqt"
    out = POSEDIR / f"{name}_out.pdbqt"
    if not lig.exists():
        if not smi_to_pdbqt(smi, lig): return (None, "prep_failed")
    cx, cy, cz = [f"{v:.3f}" for v in CENTER]; sz = f"{BOX:.1f}"
    cmd = [VINA, "--receptor", str(RECEPTOR), "--ligand", str(lig),
           "--center_x", cx, "--center_y", cy, "--center_z", cz,
           "--size_x", sz, "--size_y", sz, "--size_z", sz,
           "--out", str(out), "--exhaustiveness", str(EXH), "--num_modes", str(NMODES)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0: return (None, f"vina_err:{r.returncode}")
        best = None
        for ln in out.read_text().splitlines():
            if ln.startswith("REMARK VINA RESULT"):
                best = float(ln.split()[3]); break
        return (best, "ok" if best is not None else "no_score")
    except Exception as e:
        return (None, f"err:{e}")

def dock_list(items, tag):
    rows = []
    for name, smi, meta in tqdm(items, desc=tag):
        score, status = dock_one(name, smi)
        rows.append({"name": name, "smiles": smi, "vina_kcal_mol": score,
                     "status": status, **meta})
    return pd.DataFrame(rows)

# ── 1. celecoxib redock ───────────────────────────────────────────────────────
print("\n[1] celecoxib redock")
cele = dock_list([("celecoxib", CELECOXIB_SMILES, {"set": "control"})], "celecoxib")
cele.to_csv(OUTDIR / "celecoxib_redock.csv", index=False)
print(cele[["name", "vina_kcal_mol", "status"]].to_string(index=False))

# ── 2. random ZINC baseline (same 100 compounds, fresh run) ───────────────────
print("\n[2] random ZINC baseline (100, fresh consistent run + poses)")
rb = pd.read_csv(BASELINE_CSV)
smi_col = "smiles" if "smiles" in rb.columns else "SMILES"
items = [(f"rand_{i:03d}", rb.iloc[i][smi_col], {"set": "random_zinc"}) for i in range(len(rb))]
rnd = dock_list(items, "random")
rnd.to_csv(OUTDIR / "random_zinc_v3.csv", index=False)
ok = rnd[rnd.status == "ok"]
print(f"  docked ok: {len(ok)}/{len(rnd)}  best {ok.vina_kcal_mol.min():.2f}  median {ok.vina_kcal_mol.median():.2f}")

# ── 3. Vina-on-training validation ────────────────────────────────────────────
print("\n[3] Vina-on-training validation (COX-2 actives vs inactives)")
df = pd.read_csv(COX2_CHEMBL)
rng = np.random.default_rng(42)
act = df[df.pIC50 >= 7.0]; ina = df[df.pIC50 <= 5.0]
act_s = act.sample(n=min(50, len(act)), random_state=42)
ina_s = ina.sample(n=min(50, len(ina)), random_state=42)
print(f"  sampled actives={len(act_s)} (pool {len(act)}), inactives={len(ina_s)} (pool {len(ina)})")
vitems  = [(f"act_{i:03d}", s, {"set": "chembl_active"})   for i, s in enumerate(act_s.SMILES)]
vitems += [(f"ina_{i:03d}", s, {"set": "chembl_inactive"}) for i, s in enumerate(ina_s.SMILES)]
val = dock_list(vitems, "validation")
val.to_csv(OUTDIR / "vina_validation_v3.csv", index=False)
for grp in ("chembl_active", "chembl_inactive"):
    g = val[(val.set == grp) & (val.status == "ok")]
    print(f"  {grp}: docked ok {len(g)}  median {g.vina_kcal_mol.median():.2f}")

print("\nDONE_V3_DOCKING")
