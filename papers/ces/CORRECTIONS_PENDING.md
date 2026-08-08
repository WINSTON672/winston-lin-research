# CES paper — corrections pending

Recomputed 2026-08-08 from `~/lin-score/lin/models.py` + `scorer.py`.
ATLAS has already been corrected (commit `01cadfd`); this paper has not, and
the two currently disagree on the metric definition.

## 1. The CES scores are fine. The correlations are not.

`maxCES = 0.718` (variance-optimal $w$ = 0.47/0.53) and `0.770` (baseline
0.60/0.40) **both reproduce exactly** from the released code. The metric
implementation is sound and the leaderboard is real.

The reported Spearman correlations do **not** reproduce:

| Reported | ρ(CES,HumanEval) | ρ(CES,MMLU) |
|---|---|---|
| Paper claims | **+0.923** | **+0.725** |
| log₂(C), w .47/.53 | +0.703 | +0.478 |
| log₂(C), w .60/.40 | +0.819 | +0.604 |
| log₂(1+C), w .47/.53 | +0.780 | +0.549 |
| log₂(1+C), w .60/.40 | +0.802 | +0.593 |
| Pearson instead of Spearman | +0.578 | +0.527 |
| vs transformed b_j not raw accuracy | +0.868 | +0.478 |

Neither denominator, weight set, correlation type, nor pairing produces
0.923/0.725. Those two numbers cannot be regenerated from the released
artifacts and must be replaced with computed values wherever they appear:
lines 396, 712, 814, 821, 893–894, 901, 903, 943.

## 2. Adopt the chance-normalized transform, to match ATLAS

ATLAS now uses `b_j(p) = 1 − H₂(clip₀¹((p − c_j)/(1 − c_j)))`, c_MMLU = 0.25,
c_HE = 0. This paper still uses the binary `1 − H₂(p)`, which assumes a 0.5
chance baseline and so scores a 0.49 on four-option MMLU near zero.

Do **not** use the purely linear rescaling `max(0,(p−c)/(1−c))` — it inverts
the metric (phi-3-mini tops at CES 3.22, both correlations go negative).

### Corrected leaderboard (w = 0.47/0.53, log₂(C))

| Model | old CES | old rk | **new CES** | **new rk** | shift |
|---|---|---|---|---|---|
| gpt-4o | 0.718 | 1 | **0.692** | 1 | – |
| llama-3-70b | 0.656 | 2 | **0.577** | 2 | – |
| claude-3-opus | 0.589 | 3 | **0.548** | 3 | – |
| gpt-4 | 0.517 | 6 | **0.484** | 4 | **+2** |
| claude-3-haiku | 0.565 | 4 | **0.441** | 5 | −1 |
| mistral-7b | 0.531 | 5 | **0.434** | 6 | −1 |
| gemini-ultra | 0.453 | 7 | **0.420** | 7 | – |
| claude-3-sonnet | 0.345 | 9 | **0.272** | 8 | +1 |
| gemini-pro | 0.262 | 11 | **0.191** | 9 | **+2** |
| phi-3-mini | 0.397 | 8 | **0.141** | 10 | **−2** |
| llama-3-8b | 0.297 | 10 | **0.140** | 11 | −1 |
| mixtral-8x7b | 0.169 | 12 | **0.073** | 12 | – |
| gpt-3.5-turbo | 0.097 | 13 | **0.026** | 13 | – |

Rank correlation old vs new: **ρ = 0.956** — the correction is a refinement,
not a reordering. New correlations: **ρ(HE) = 0.813, ρ(MMLU) = 0.632,
max = 0.692 (gpt-4o, no human parity).**

phi-3-mini falling 8 → 10 is the intended effect: the binary transform was
crediting it for sub-chance-corrected MMLU.

## 3. Knock-on edits required

- **Finding 3** ("No model exceeds 0.73 CES", line 716) → 0.692.
- **Finding 1** (line 712): Gemini Ultra is #7 not #6 under the new transform;
  the MMLU-rank-vs-CES-rank claim survives but the numbers change.
- Line 821 attributes CES to "code generation capability (ρ = 0.923 with
  HumanEval)" — becomes 0.813, still the dominant factor.
- Correlation table (893–894) and its 95% CIs need recomputing; the CIs were
  derived from the unreproducible ρ values.
- Figure at line 943 (CES vs MMLU scatter, labelled ρ = 0.725) → 0.632.
- Sensitivity section (987–993) quotes ρ = 0.989 for ±30% FLOPs from a
  two-scenario check. ATLAS now reports a proper 1000-draw estimate:
  mean ρ = 0.987, min 0.962. Use that.

## 4. Separate issue — the TEB numbers in §TEB are from corrupted data

Lines 56–62 and 115–118 cite ρ(quality, TEB) = 0.476, n = 8, "rank divergence
on 5 of 8 models". That run truncated reasoning models at max_tokens = 1024,
which collapsed their quality scores (Qwen3-8B 0.38 → 0.87 and DeepSeek-R1
0.53 → 0.84 once re-run at 8192). ATLAS already moved to token-based TES with
the re-run data and reports ρ = −0.33, n = 8, p ≈ 0.42. This paper's TEB
section is describing an artifact of the truncation bug and needs either the
re-run numbers or removal.

## 5. Weight mismatch, unresolved

`scorer.py` hardcodes 0.60/0.40; both papers state 0.47/0.53. The code's
weights give *better* correlations (0.824/0.753 under chance-ent), which is
exactly why they should not be adopted on that basis. Pick the weights on a
stated principle, fix the code to match the paper, and report whatever falls
out.
