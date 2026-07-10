#!/usr/bin/env python3
"""
COX-2 docking pipeline — Strix runner.

Pipeline:
  1. Download PDB 1CX2 (human COX-2 + celecoxib crystal)
  2. Strip waters/cofactors, prep receptor PDBQT (via Meeko)
  3. Read candidates_enriched.csv (127 ML-screened SMILES)
  4. For each candidate: 3D conformer (RDKit) -> ligand PDBQT (Meeko)
  5. Dock into celecoxib binding pocket:
       - Uni-Dock (CUDA) if available  -> batch, ~2 min on RTX 5080
       - AutoDock Vina (CPU) fallback   -> sequential, ~1 hour
  6. Write results/docking_results.csv ranked by binding affinity
  7. Git commit + push to research repo

Run:
  python dock_pipeline.py                 (full run)
  python dock_pipeline.py --test 5        (smoke test on 5 candidates)
  python dock_pipeline.py --engine vina   (force CPU Vina)
  python dock_pipeline.py --no-push       (skip git push)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LIGAND_DIR = HERE / "ligands"
RESULTS.mkdir(exist_ok=True)
LIGAND_DIR.mkdir(exist_ok=True)

PDB_ID = "1CX2"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
CELECOXIB_RESNAME = "S58"
CHAIN = "A"          # 1CX2 has 4 chains (homo-tetramer); use chain A only
BOX_SIZE = 22.0      # Å; covers the celecoxib pocket with margin
EXHAUSTIVENESS = 8
N_POSES = 5


# ── deps ──────────────────────────────────────────────────────────────────────

def _pip_install(pkgs):
    print(f"[setup] pip install {' '.join(pkgs)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

def ensure_deps():
    needed = []
    # Note: 'vina' Python package skipped — pip-broken on Windows.
    # Docking uses standalone unidock or vina binary instead.
    for mod, pkg in [("rdkit","rdkit"), ("meeko","meeko"), ("pandas","pandas"),
                     ("requests","requests"), ("tqdm","tqdm"), ("numpy","numpy"),
                     ("scipy","scipy")]:
        try: __import__(mod)
        except ImportError: needed.append(pkg)
    if needed:
        _pip_install(needed)


def detect_engine(forced: str | None):
    """Return ('unidock', path) or ('vina', path) or raise."""
    ud = shutil.which("unidock") or shutil.which("unidock.exe")
    vn = shutil.which("vina") or shutil.which("vina.exe")
    if forced == "unidock":
        if not ud: raise SystemExit("[err] --engine unidock: binary not in PATH. Install: conda install -c conda-forge unidock")
        return ("unidock", ud)
    if forced == "vina":
        if not vn: raise SystemExit("[err] --engine vina: vina.exe not in PATH. Download from https://github.com/ccsb-scripps/AutoDock-Vina/releases")
        return ("vina", vn)
    if ud: return ("unidock", ud)
    if vn: return ("vina", vn)
    raise SystemExit(
        "[err] No docking engine found. Install ONE of:\n"
        "  GPU (fast):  conda install -c conda-forge unidock\n"
        "  CPU:         download vina.exe from https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
        "               put it in C:\\Windows\\System32 or anywhere in PATH"
    )


# ── PDB + receptor prep ───────────────────────────────────────────────────────

def download_pdb() -> Path:
    import requests
    pdb_path = HERE / f"{PDB_ID}.pdb"
    if pdb_path.exists():
        print(f"[pdb] cached: {pdb_path}")
        return pdb_path
    print(f"[pdb] downloading {PDB_URL}")
    r = requests.get(PDB_URL, timeout=60); r.raise_for_status()
    pdb_path.write_text(r.text)
    print(f"[pdb] saved -> {pdb_path}  ({pdb_path.stat().st_size:,} bytes)")
    return pdb_path

def split_pdb(pdb_path: Path):
    """Extract chain-A protein and chain-A celecoxib centroid only.

    1CX2 is a homo-tetramer (chains A/B/C/D, each with celecoxib).
    Earlier versions averaged all 4 celecoxibs and produced a phantom
    docking-box center in the tetramer interior, NOT in any real binding
    pocket. This version restricts to chain A throughout.
    """
    import numpy as np
    lines = pdb_path.read_text().splitlines()
    protein_lines, lig_coords = [], []
    for ln in lines:
        if ln.startswith("ATOM  ") and ln[21] == CHAIN:
            protein_lines.append(ln)
        elif ln.startswith("HETATM") and ln[21] == CHAIN:
            if ln[17:20].strip() == CELECOXIB_RESNAME:
                try:
                    lig_coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError: pass
        elif ln.startswith(("TER","END")):
            protein_lines.append(ln)
    if not lig_coords:
        raise SystemExit(f"[err] no {CELECOXIB_RESNAME} ligand in chain {CHAIN} of {pdb_path}")
    centroid = np.array(lig_coords).mean(axis=0)
    print(f"[box] chain-{CHAIN} celecoxib centroid: {centroid.round(2).tolist()}  (n={len(lig_coords)} atoms)")
    protein_pdb = HERE / f"receptor_protein_chain{CHAIN}.pdb"
    protein_pdb.write_text("\n".join(protein_lines) + "\nEND\n")
    print(f"[receptor] chain-{CHAIN} protein PDB -> {protein_pdb}")
    return protein_pdb, centroid

def prep_receptor(protein_pdb: Path) -> Path:
    """Prep receptor PDBQT. Tries OpenBabel first (most permissive), then Meeko."""
    out_pdbqt = HERE / "receptor.pdbqt"
    if out_pdbqt.exists():
        print(f"[receptor] cached: {out_pdbqt}")
        return out_pdbqt
    tried = []

    # Path A: OpenBabel (robust for crystal PDBs with non-standard residues)
    obabel = shutil.which("obabel")
    if obabel:
        print("[receptor] running obabel (rigid receptor mode)")
        try:
            subprocess.check_call([
                obabel, str(protein_pdb), "-O", str(out_pdbqt),
                "-xr",      # rigid receptor (no rotatable bonds)
                "-p", "7.4" # protonate at physiological pH
            ])
            if out_pdbqt.exists():
                print(f"[receptor] PDBQT -> {out_pdbqt}  (via obabel)")
                return out_pdbqt
        except subprocess.CalledProcessError as e:
            tried.append(f"obabel: {e}")

    # Path B: Meeko fallback
    print("[receptor] obabel unavailable/failed, trying Meeko")
    for cmd in [
        [sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
         "--read_pdb", str(protein_pdb), "-o", str(out_pdbqt.with_suffix("")), "-p"],
        ["mk_prepare_receptor.py",
         "--read_pdb", str(protein_pdb), "-o", str(out_pdbqt.with_suffix("")), "-p"],
    ]:
        try:
            subprocess.check_call(cmd); break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            tried.append(f"{cmd[0]}: {e}")
    if not out_pdbqt.exists():
        raise SystemExit("[err] receptor prep failed (tried obabel + meeko):\n  " + "\n  ".join(tried))
    print(f"[receptor] PDBQT -> {out_pdbqt}  (via meeko)")
    return out_pdbqt


# ── ligand prep ───────────────────────────────────────────────────────────────

def smiles_to_pdbqt_file(smi: str, out_path: Path) -> bool:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return False
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50) != 0: return False
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception: return False
    try:
        prep = MoleculePreparation()
        prep.prepare(mol)
        setups = prep.mol_setups if hasattr(prep, "mol_setups") else [prep.setup]
        pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not ok: return False
        out_path.write_text(pdbqt_str)
        return True
    except Exception:
        return False

def prep_all_ligands(df):
    from tqdm import tqdm
    items = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="ligand prep"):
        smi = str(row["SMILES"]).strip()
        path = LIGAND_DIR / f"cand_{i:04d}.pdbqt"
        ok = smiles_to_pdbqt_file(smi, path) if not path.exists() else True
        items.append({"idx": i, "smiles": smi, "path": str(path), "ok": ok})
    n_ok = sum(1 for x in items if x["ok"])
    print(f"[ligands] {n_ok}/{len(items)} prepped")
    return items


# ── docking engines ───────────────────────────────────────────────────────────

def dock_unidock(unidock_bin, receptor_pdbqt: Path, ligand_items, centroid):
    """Batch dock all ligands with Uni-Dock (CUDA)."""
    cx, cy, cz = [f"{v:.3f}" for v in centroid]
    sz = f"{BOX_SIZE:.1f}"
    ok_items = [x for x in ligand_items if x["ok"]]
    if not ok_items:
        return {}
    ligand_list = HERE / "ligands.txt"
    ligand_list.write_text("\n".join(x["path"] for x in ok_items) + "\n")
    out_dir = HERE / "unidock_out"
    out_dir.mkdir(exist_ok=True)
    cmd = [
        unidock_bin,
        "--receptor", str(receptor_pdbqt),
        "--ligand_index", str(ligand_list),
        "--center_x", cx, "--center_y", cy, "--center_z", cz,
        "--size_x", sz, "--size_y", sz, "--size_z", sz,
        "--dir", str(out_dir),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(N_POSES),
        "--scoring", "vina",
        "--search_mode", "balance",
    ]
    print(f"[unidock] launching CUDA batch dock ({len(ok_items)} ligands)")
    t0 = time.time()
    subprocess.check_call(cmd)
    print(f"[unidock] done in {time.time()-t0:.1f}s")
    # parse outputs: <name>_out.pdbqt; first REMARK VINA RESULT line = best score
    scores = {}
    for item in ok_items:
        name = Path(item["path"]).stem
        out_file = out_dir / f"{name}_out.pdbqt"
        if not out_file.exists():
            scores[item["idx"]] = (None, "no_output")
            continue
        best = None
        for ln in out_file.read_text().splitlines():
            if ln.startswith("REMARK VINA RESULT"):
                parts = ln.split()
                try: best = float(parts[3]); break
                except (IndexError, ValueError): pass
        scores[item["idx"]] = (best, "ok" if best is not None else "no_score")
    return scores

def dock_vina_cpu(vina_bin, receptor_pdbqt: Path, ligand_items, centroid):
    """Sequential CPU docking via standalone vina.exe binary (subprocess)."""
    from tqdm import tqdm
    cx, cy, cz = [f"{v:.3f}" for v in centroid]
    sz = f"{BOX_SIZE:.1f}"
    out_dir = HERE / "vina_out"; out_dir.mkdir(exist_ok=True)
    scores = {}
    for item in tqdm([x for x in ligand_items if x["ok"]], desc="vina"):
        name = Path(item["path"]).stem
        out_file = out_dir / f"{name}_out.pdbqt"
        cmd = [
            vina_bin,
            "--receptor", str(receptor_pdbqt),
            "--ligand", item["path"],
            "--center_x", cx, "--center_y", cy, "--center_z", cz,
            "--size_x", sz, "--size_y", sz, "--size_z", sz,
            "--out", str(out_file),
            "--exhaustiveness", str(EXHAUSTIVENESS),
            "--num_modes", str(N_POSES),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                scores[item["idx"]] = (None, f"vina_err:{res.returncode}")
                continue
            best = None
            if out_file.exists():
                for ln in out_file.read_text().splitlines():
                    if ln.startswith("REMARK VINA RESULT"):
                        try: best = float(ln.split()[3]); break
                        except (IndexError, ValueError): pass
            scores[item["idx"]] = (best, "ok" if best is not None else "no_score")
        except subprocess.TimeoutExpired:
            scores[item["idx"]] = (None, "timeout")
        except Exception as e:
            scores[item["idx"]] = (None, f"err:{e}")
    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--engine", choices=["auto","unidock","vina"], default="auto")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    ensure_deps()

    import pandas as pd

    cand_csv = HERE / "candidates_enriched.csv"
    if not cand_csv.exists():
        raise SystemExit(f"[err] missing {cand_csv}")
    df = pd.read_csv(cand_csv).reset_index(drop=True)
    if args.test: df = df.head(args.test)
    print(f"[input] {len(df)} candidates")

    engine, engine_bin = detect_engine(None if args.engine == "auto" else args.engine)
    print(f"[engine] {engine} ({engine_bin})")

    pdb_path = download_pdb()
    protein_pdb, centroid = split_pdb(pdb_path)
    receptor_pdbqt = prep_receptor(protein_pdb)
    ligand_items = prep_all_ligands(df)

    t0 = time.time()
    if engine == "unidock":
        scores = dock_unidock(engine_bin, receptor_pdbqt, ligand_items, centroid)
    else:
        scores = dock_vina_cpu(engine_bin, receptor_pdbqt, ligand_items, centroid)
    print(f"[done] docking finished in {(time.time()-t0)/60:.1f} min")

    rows = []
    for item in ligand_items:
        s, status = scores.get(item["idx"], (None, "missing"))
        if not item["ok"]: status = "prep_failed"
        row = df.iloc[item["idx"]].to_dict()
        row["vina_kcal_mol"] = s
        row["dock_status"]   = status
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("vina_kcal_mol", ascending=True, na_position="last")
    out_csv = RESULTS / "docking_results.csv"
    out.to_csv(out_csv, index=False)
    print(f"[out] {out_csv}")

    ok = out[out["dock_status"] == "ok"]
    if len(ok):
        print(f"[summary] {len(ok)}/{len(out)} docked successfully")
        print(f"  best:   {ok['vina_kcal_mol'].min():+.2f} kcal/mol")
        print(f"  median: {ok['vina_kcal_mol'].median():+.2f}")
        print(f"  worst:  {ok['vina_kcal_mol'].max():+.2f}")
        print("  top 5 by Vina score:")
        for _, r in ok.head(5).iterrows():
            ml = r.get("pIC50_cox2", "?"); sel = r.get("selectivity", "?")
            tan = r.get("max_tanimoto", "?")
            print(f"    Vina {r['vina_kcal_mol']:+.2f}  ML pIC50 {ml}  sel {sel}  Tan {tan}  | {str(r['SMILES'])[:60]}")

    if not args.no_push:
        repo_root = HERE.parent.parent  # /Desktop/research
        try:
            subprocess.check_call(["git", "add", "code/cox2_docking/results/"], cwd=repo_root)
            msg = (f"cox2_docking: {len(ok)}/{len(out)} docked, best={ok['vina_kcal_mol'].min():+.2f} kcal/mol"
                   if len(ok) else "cox2_docking: run produced no successful docks")
            subprocess.check_call(["git", "commit", "-m", msg], cwd=repo_root)
            subprocess.check_call(["git", "push"], cwd=repo_root)
            print("[git] pushed to origin")
        except subprocess.CalledProcessError as e:
            print(f"[git] push skipped: {e}")


if __name__ == "__main__":
    main()
