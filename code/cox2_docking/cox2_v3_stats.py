#!/usr/bin/env python3
"""
cox2_v3_stats.py — recompute all COX-2 audit statistics on corrected (v3) data.
Writes final_stats.json + random_zinc_baseline.csv (v3) + vina_validation.csv (v3)
into papers/cox2_journal/, and regenerates the Vina figures.

Methodology mirrors main.tex Methods exactly:
 - baseline comparison: Mann-Whitney U, Welch t, Cohen's d (pooled SD), CLES,
   Hodges-Lehmann shift + normal-approx 95% CI
 - power: statsmodels TTestIndPower, MDE at 50/80/95%, power for d in {.2,.3,.5,.8}
 - ML-Vina correlation: Pearson r + Spearman rho, 5000-resample bootstrap CI, rng(42)
 - Vina-on-training validation: 3 Mann-Whitney comparisons
"""
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, spearmanr, norm
from statsmodels.stats.power import TTestIndPower

from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")
import sys, os
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUP = HERE / "v3_support"
PAPER = Path("/Users/winstonlin/Desktop/research/papers/cox2_journal")
CELE = -11.227  # v3 celecoxib (correct CHEMBL118 structure) redock; RMSD filled by contacts script

# ── load ──────────────────────────────────────────────────────────────────────
cand = pd.read_csv(HERE / "results" / "docking_results.csv")
cand = cand[cand.dock_status == "ok"].reset_index(drop=True)
ours = cand["vina_kcal_mol"].to_numpy(float)

rnd = pd.read_csv(SUP / "random_zinc_v3.csv")
rnd = rnd[rnd.status == "ok"].reset_index(drop=True)
rand = rnd["vina_kcal_mol"].to_numpy(float)

ml = cand["pIC50_cox2"].to_numpy(float)

print(f"candidates docked ok: {len(ours)}   random docked ok: {len(rand)}")

# ── candidate distribution ────────────────────────────────────────────────────
S = {}
S["celecoxib_vina"] = CELE
S["our_vina_best"] = float(ours.min()); S["our_vina_mean"] = float(ours.mean())
S["our_vina_median"] = float(np.median(ours)); S["our_vina_worst"] = float(ours.max())
S["our_vina_std"] = float(ours.std(ddof=1))
S["n_strong_le_minus8"] = int((ours <= -8).sum())
S["n_beat_celecoxib"] = int((ours < CELE).sum())
S["random_vina_best"] = float(rand.min()); S["random_vina_mean"] = float(rand.mean())
S["random_vina_median"] = float(np.median(rand))
S["random_n_strong"] = int((rand <= -8).sum())
S["random_n_beat_celecoxib"] = int((rand < CELE).sum())

# ── baseline comparison ───────────────────────────────────────────────────────
U, p_mwu = mannwhitneyu(ours, rand, alternative="two-sided")
S["mwu_U"] = float(U); S["mwu_p"] = float(p_mwu)
t, p_w = ttest_ind(ours, rand, equal_var=False)
S["welch_t"] = float(t); S["welch_p"] = float(p_w)
S["mean_diff_random_minus_ours"] = float(rand.mean() - ours.mean())
# Cohen's d (ours - random)/pooled SD
nx, ny = len(ours), len(rand)
sp = np.sqrt(((nx-1)*ours.var(ddof=1) + (ny-1)*rand.var(ddof=1))/(nx+ny-2))
S["cohens_d"] = float((ours.mean() - rand.mean())/sp)
# CLES: P(ours < random)  (more negative = better)
diffs = ours[:, None] - rand[None, :]
S["cles_ours_better"] = float((diffs < 0).mean())
# Hodges-Lehmann shift + normal-approx 95% CI (Moses interval on pairwise diffs)
d_sorted = np.sort(diffs.ravel())
S["hodges_lehmann"] = float(np.median(d_sorted))
z = norm.ppf(0.975)
K = nx*ny/2 - z*np.sqrt(nx*ny*(nx+ny+1)/12)
k = int(np.floor(K))
lo = d_sorted[max(k-1, 0)]; hi = d_sorted[min(nx*ny-k, nx*ny-1)]
S["hl_ci"] = [float(lo), float(hi)]

# ── power analysis ────────────────────────────────────────────────────────────
pw = TTestIndPower(); ratio = ny/nx
S["mde"] = {q: float(pw.solve_power(effect_size=None, nobs1=nx, alpha=0.05, power=q, ratio=ratio, alternative="two-sided"))
            for q in (0.5, 0.8, 0.95)}
S["power_at_d"] = {d: float(pw.power(effect_size=d, nobs1=nx, alpha=0.05, ratio=ratio, alternative="two-sided"))
                   for d in (0.2, 0.3, 0.5, 0.8)}

# ── ML-Vina correlation + bootstrap CI ────────────────────────────────────────
pr, _ = pearsonr(ml, ours); sr, _ = spearmanr(ml, ours)
S["pearson_r"] = float(pr); S["spearman_rho"] = float(sr)
rng = np.random.default_rng(42); n = len(ml); bp, bs = [], []
for _ in range(5000):
    idx = rng.integers(0, n, n)
    bp.append(pearsonr(ml[idx], ours[idx])[0]); bs.append(spearmanr(ml[idx], ours[idx])[0])
S["pearson_ci"] = [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))]
S["spearman_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# ── SA + BRENK on candidates ──────────────────────────────────────────────────
mols = [Chem.MolFromSmiles(s) for s in cand["SMILES"]]
sa = [sascorer.calculateScore(m) for m in mols if m]
S["sa_mean"] = float(np.mean(sa)); S["sa_median"] = float(np.median(sa))
S["sa_le4"] = int(sum(x <= 4 for x in sa)); S["sa_le5"] = int(sum(x <= 5 for x in sa))
S["sa_gt6"] = int(sum(x > 6 for x in sa))
bp_ = FilterCatalogParams(); bp_.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
brenk = FilterCatalog(bp_)
flagged = sum(1 for m in mols if m and brenk.HasMatch(m))
S["brenk_flagged"] = int(flagged); S["brenk_clean"] = int(len(mols) - flagged)
S["n_candidates_docked"] = int(len(ours)); S["n_random_docked"] = int(len(rand))

# ── Vina-on-training validation ───────────────────────────────────────────────
val = pd.read_csv(SUP / "vina_validation_v3.csv")
act = val[(val.set == "chembl_active") & (val.status == "ok")]["vina_kcal_mol"].to_numpy(float)
ina = val[(val.set == "chembl_inactive") & (val.status == "ok")]["vina_kcal_mol"].to_numpy(float)
def mwu(a, b):
    u, p = mannwhitneyu(a, b, alternative="two-sided"); return float(u), float(p)
S["val_n_active"] = int(len(act)); S["val_n_inactive"] = int(len(ina))
S["val_active_median"] = float(np.median(act)); S["val_inactive_median"] = float(np.median(ina))
S["val_act_vs_ina_U"], S["val_act_vs_ina_p"] = mwu(act, ina)
S["val_ina_vs_rand_U"], S["val_ina_vs_rand_p"] = mwu(ina, rand)
S["val_act_vs_rand_U"], S["val_act_vs_rand_p"] = mwu(act, rand)

# save v3 CSVs into paper dir
rnd.rename(columns={"smiles": "smiles"}).to_csv(PAPER / "random_zinc_baseline.csv", index=False)
val.to_csv(PAPER / "vina_validation.csv", index=False)
(PAPER / "final_stats.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2))

# ── figures ───────────────────────────────────────────────────────────────────
# 1. candidate Vina histogram
plt.figure(figsize=(5, 3.5))
plt.hist(ours, bins=20, color="steelblue", edgecolor="white", lw=0.4)
plt.axvline(CELE, color="red", ls="--", lw=1, label=f"celecoxib {CELE:.2f}")
plt.xlabel("Vina docking affinity (kcal/mol)"); plt.ylabel("Count")
plt.title(f"Candidate docking affinities (n={len(ours)})"); plt.legend()
plt.tight_layout(); plt.savefig(PAPER / "vina_histogram.png", dpi=150); plt.close()

# 2. candidates vs random baseline
plt.figure(figsize=(5, 3.5))
bins = np.linspace(min(ours.min(), rand.min()), max(ours.max(), rand.max()), 22)
plt.hist(rand, bins=bins, alpha=0.55, color="gray", label=f"random ZINC (n={len(rand)})", edgecolor="white", lw=0.3)
plt.hist(ours, bins=bins, alpha=0.55, color="steelblue", label=f"ML candidates (n={len(ours)})", edgecolor="white", lw=0.3)
plt.axvline(CELE, color="red", ls="--", lw=1, label="celecoxib")
plt.xlabel("Vina docking affinity (kcal/mol)"); plt.ylabel("Count")
plt.title(f"Candidates vs. random baseline (MWU p={p_mwu:.2f})"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PAPER / "vina_baseline_comparison.png", dpi=150); plt.close()

# 3. Vina vs ML pIC50 scatter
plt.figure(figsize=(5, 4))
tan = cand["max_tanimoto"].to_numpy(float) if "max_tanimoto" in cand.columns else np.zeros(len(cand))
sc = plt.scatter(ml, ours, c=tan, cmap="viridis", s=40, edgecolor="k", lw=0.3)
plt.colorbar(sc, label="max Tanimoto to training set")
plt.axhline(CELE, color="red", ls="--", lw=1, label="celecoxib")
plt.gca().invert_yaxis()
plt.xlabel("ML predicted COX-2 pIC50"); plt.ylabel("Vina affinity (kcal/mol)")
plt.title(f"Spearman rho = {sr:.3f}  95% CI [{S['spearman_ci'][0]:.2f}, {S['spearman_ci'][1]:.2f}]")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(PAPER / "vina_vs_ml_scatter.png", dpi=150); plt.close()

print("\nfigures written; DONE_V3_STATS")
