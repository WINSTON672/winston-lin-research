#!/usr/bin/env python3
"""Regenerate the v3 graphical abstract (4 panels) from corrected data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PAPER = "/Users/winstonlin/Desktop/research/papers/cox2_journal"
SUP = "/Users/winstonlin/Desktop/research/code/cox2_docking/v3_support"

cand = pd.read_csv(f"{PAPER}/candidates_full.csv")
ours = cand["vina_kcal_mol"].to_numpy(float)
tan = cand["max_tanimoto"].to_numpy(float)
ml = cand["pIC50_cox2"].to_numpy(float)
rnd = pd.read_csv(f"{PAPER}/random_zinc_baseline.csv")
rand = rnd["vina_kcal_mol"].to_numpy(float)
val = pd.read_csv(f"{PAPER}/vina_validation.csv")
act = val[(val.set=="chembl_active")&(val.status=="ok")]["vina_kcal_mol"].to_numpy(float)
ina = val[(val.set=="chembl_inactive")&(val.status=="ok")]["vina_kcal_mol"].to_numpy(float)
CELE = -11.23

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("A standard ML virtual-screening pipeline yields candidates indistinguishable from random ZINC",
             fontsize=13, fontweight="bold", y=0.98)

# A: applicability domain
a = ax[0,0]
a.hist(tan, bins=18, color="#d98a29", edgecolor="white", lw=0.4)
a.axvline(0.40, color="red", ls="--", lw=1.4)
a.set_xlabel("Max Tanimoto to training set"); a.set_ylabel("Candidates")
a.set_title("(A) Applicability domain: 39/42 out-of-domain", fontsize=11, loc="left")
a.text(0.42, a.get_ylim()[1]*0.85, "in-domain\nthreshold T=0.40", color="red", fontsize=8)

# B: candidates vs random
b = ax[0,1]
bins = np.linspace(min(ours.min(), rand.min()), max(ours.max(), rand.max()), 22)
b.hist(rand, bins=bins, alpha=0.55, color="gray", label=f"random ZINC (n={len(rand)})", edgecolor="white", lw=0.3)
b.hist(ours, bins=bins, alpha=0.6, color="#2f6db5", label=f"ML candidates (n={len(ours)})", edgecolor="white", lw=0.3)
b.axvline(CELE, color="red", ls="--", lw=1.2)
b.set_xlabel("Vina affinity (kcal/mol)"); b.set_ylabel("Count")
b.set_title("(B) No docking enrichment (MWU p=0.89)", fontsize=11, loc="left")
b.legend(fontsize=8)

# C: ML vs Vina
c = ax[1,0]
rho = spearmanr(ml, ours)[0]
sc = c.scatter(ml, ours, c=tan, cmap="viridis", s=42, edgecolor="k", lw=0.3)
c.axhline(CELE, color="red", ls="--", lw=1.2)
c.invert_yaxis()
c.set_xlabel("ML predicted COX-2 pIC50"); c.set_ylabel("Vina affinity (kcal/mol)")
c.set_title(f"(C) ML and Vina rankings uncorrelated ($\\rho$={rho:.2f})", fontsize=11, loc="left")
plt.colorbar(sc, ax=c, label="max Tanimoto", fraction=0.046, pad=0.04)

# D: validation — the oracle works
d = ax[1,1]
groups = [("ChEMBL\nactives", act, "#2ca25f"), ("ChEMBL\ninactives", ina, "#99d8c9"),
          ("random\nZINC", rand, "gray"), ("ML\ncandidates", ours, "#2f6db5")]
for i,(lab,arr,col) in enumerate(groups):
    x = np.random.default_rng(i).normal(i, 0.06, len(arr))
    d.scatter(x, arr, s=14, color=col, alpha=0.6, edgecolor="none")
    d.hlines(np.median(arr), i-0.28, i+0.28, color="black", lw=2)
d.set_xticks(range(4)); d.set_xticklabels([g[0] for g in groups], fontsize=8)
d.invert_yaxis()
d.set_ylabel("Vina affinity (kcal/mol)")
d.set_title("(D) Oracle validated: actives$>$random (p=0.006);\ncandidates sit at random level", fontsize=10, loc="left")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{PAPER}/graphical_abstract.png", dpi=150)
print("graphical_abstract.png regenerated (v3, 4 panels)")
