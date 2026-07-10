# COX-2 ML virtual-screening pipeline

Reproduces the analysis in `papers/cox2_journal/`. Run in order:

1. `pull_chembl.py` — retrieve COX-2 (CHEMBL230) and COX-1 (CHEMBL221) IC50
   bioactivity from ChEMBL → `cox2_chembl.csv`, `cox1_chembl.csv`
   (both provided here so the pull is optional).
2. `cox2_screen.py` — train the COX-2 Random Forest, screen ZINC250k, apply
   potency / Lipinski / solubility / PAINS filters.
3. `cox2_additions.py` — COX-1 model, selectivity margins, applicability-domain
   (max-Tanimoto) coverage, Murcko scaffolds, SA / BRENK annotations.
4. `cox2_model_table.py` — held-out model comparison (RF vs XGBoost vs Ridge).

Docking (Vina) lives under `code/cox2_docking/`. All ML random seeds are fixed
at `random_state=42`.

**Targets:** COX-2 = CHEMBL230, COX-1 = CHEMBL221 (PDB 1CX2 for docking).
**Data columns:** `SMILES, IC50_nM, pIC50`.
