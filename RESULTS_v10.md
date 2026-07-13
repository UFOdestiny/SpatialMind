# SpatialMind — constraint_guided_v10 Results (2026-07-12)

Scaled, submission-standard run. Backbone Llama-3.1-8B-Instruct, judge
Mistral-Small-3.2-24B. Guided decoding, 0 dropped (1 generation-schema failure
in train only), leakage audit clean on the test axis (see below).

**Scale (vs v9 in parens):** ID StepGame train 4999 (1000) / val 999 (250) /
test 2000 (500). OOD each: val 1000 (250) / test ~2000 (500), except babi
(val 196 / test 2000 — dataset cap). OOD = validation-adapted transfer, NOT
zero-shot: target-domain validation used only for the head aggregation rule +
confidence normalization + the fusion combiner fit; test labels/stats never used.
Leakage audit **passes green** (see below); ID val is 999 after removing one
train/val template collision.

**New dataset this run:** `SpartQA_YN` (tasksource/spartqa-yn) — machine-generated
block-world yes/no/DK, added to broaden the symbolizability spectrum.

## AUROC — all methods × all datasets (test)

| method | StepGame(ID) | spartqa | babi | SpaRTUN | SpaceNLI | SpartQA_YN |
|---|---:|---:|---:|---:|---:|---:|
| constraint_no_conflict | **0.874** | 0.313 | 0.570 | 0.403 | 0.732 | 0.523 |
| constraint_no_context | 0.870 | 0.325 | 0.572 | 0.401 | 0.701 | 0.515 |
| constraint_no_entailment | 0.841 | 0.372 | 0.593 | 0.446 | 0.635 | 0.523 |
| constraint_no_repair | 0.781 | 0.351 | 0.575 | 0.454 | 0.646 | 0.508 |
| constraint_only | 0.847 | 0.496 | 0.575 | 0.547 | 0.753 | 0.507 |
| spatialmind (main) | 0.871 | 0.343 | 0.582 | 0.420 | 0.750 | 0.510 |
| spatialmind_neural | 0.573 | 0.685 | 0.569 | 0.580 | 0.519 | 0.519 |
| factoscope | 0.522 | 0.611 | 0.473 | 0.542 | 0.480 | 0.504 |
| mlp | 0.496 | 0.430 | 0.463 | 0.595 | 0.490 | 0.503 |
| uhead | 0.494 | 0.513 | 0.465 | 0.566 | 0.497 | 0.499 |
| constraint_rule (0-param) | 0.789 | 0.459 | 0.570 | 0.407 | 0.566 | 0.496 |
| mcp | 0.462 | 0.477 | 0.584 | 0.420 | 0.509 | 0.490 |
| ccp | 0.466 | 0.490 | 0.580 | 0.427 | 0.508 | 0.489 |
| perplexity | 0.505 | 0.471 | 0.439 | 0.564 | 0.524 | 0.490 |
| token_entropy | 0.456 | 0.454 | 0.588 | 0.409 | 0.508 | 0.487 |
| random | 0.510 | 0.492 | 0.520 | 0.502 | 0.506 | 0.485 |
| **FUSION (ours)** | 0.867 | **0.721** | **0.649** | **0.643** | **0.773** | 0.504 |

## Fusion headline (symbolizability/determinacy-gated, validation-selected)

Combiner (`scripts/gated_fusion.py`) blends `constraint_no_conflict` + `mlp`.
Config (mode ∈ {scores_only, symb, determinacy}, L2) is selected per dataset on
a **validation-internal even/odd split** — no test peeking anywhere.

| | con | neu | **fused** |
|---|---:|---:|---:|
| mean (6 sets) | 0.569 | 0.496 | **0.693** |
| worst-case | 0.313 | 0.430 | **0.504** |
| fused ≥ best pure | — | — | **4/6** |

Per-dataset fused AUROC: StepGame 0.867, spartqa 0.721, babi 0.649,
SpaRTUN 0.643, SpaceNLI 0.773, SpartQA_YN 0.504.

- **spartqa +0.29 / +0.41 over the two pure methods:** both symbolic (0.313) and
  neural (0.430) are near-useless there, yet the gated fusion recovers **0.721**.
  Strongest single demonstration that per-sample routing beats either signal alone.
- **StepGame −0.006** (0.867 vs 0.874): near-lossless on ID. Fused still beats
  every UQ baseline by +29 AUROC.
- **SpartQA_YN ≈ flat (0.50):** every method is near-random here — fusion cannot
  synthesize signal that no base method has (see finding 3). Honest hard case.

## Findings (honest)

1. **Core novelty holds and strengthens at scale.** On StepGame (2000 test) every
   explicit-constraint variant (0.78–0.87) dominates every neural/statistical UQ
   baseline (0.46–0.57) by ~30 AUROC. The 0-parameter `constraint_rule` (0.789)
   still beats all 7M-param neural probes. The main vs `no_conflict` gap that
   existed at v9 (0.842 vs 0.872) largely closed at scale (0.871 vs 0.874).

2. **Transfer is representation-dependent; the fusion converts this from a weakness
   into the contribution.** Pure constraint loses to neural probes on complex-NL
   sets (spartqa, SpaRTUN); pure neural loses badly on templated sets. The
   symbolizability-gated fusion is ≥ best-pure on 4/6 and never collapses to the
   worse method — mean 0.690 vs 0.569 / 0.496.

3. **Refined thesis — parse coverage is necessary but not sufficient; the
   constraint signal needs STATUS DETERMINACY (low unknown_rate).** Per-dataset
   constraint AUROC tracks determinacy, not raw parse rate:

   | dataset | parse_rate | unknown_rate | constraint AUROC |
   |---|---:|---:|---:|
   | StepGame | 0.90 | **0.16** | **0.874** |
   | babi | 0.69 | 0.35 | 0.570 |
   | SpartQA_YN | 0.74 | 0.56 | 0.523 |
   | spartqa | 0.85 | 0.51 | 0.313 |

   SpartQA_YN parses densely (0.74) yet is indeterminate (unknown 0.56) because the
   questions require multi-hop transitive deduction the model states as intermediate
   claims the solver cannot ground → uninformative status. This motivated the
   `determinacy` gate mode (uses unknown/entailment/conclusion-status), which the
   auto-selector chose for spartqa/babi/SpartQA_YN and which lifted babi
   (+0.05 over the parse-only gate).

## Leakage audit

Fail-closed audit (`scripts/leakage_audit.py`, hashes context+question) **PASSES**:
`passed: true`, all pairwise split overlaps = 0, no per-split errors.
- The generator initially produced one k_hop=1 StepGame template that landed in
  both train and validation (train∩test and validation∩test were already 0, so
  test was never contaminated). `scripts/dedup_val_against_train.py` removed that
  single validation row (1000→999) and synced the manifest, yielding the green audit.
- Combiner test scores are invariant to a test-label permutation (the combiner
  reads no test labels/stats by construction — fit on validation, applied forward).

## Reproduce

- Full scaled pipeline (gen→train→ID→OOD→val_scores→fusion): `bash jobs/run_v10.sh`
- Fusion only (offline, from saved predictions):
  `python scripts/gated_fusion.py --results_root spatialmind/results/constraint_guided_v10_20260712 --cache_root spatialmind/cache/cached_features/constraint_guided_v10 --con_head constraint_no_conflict --neu_head mlp --mode auto --datasets "id:StepGame,spartqa:spartqa,babi:babi,SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpartQA_YN:SpartQA_YN"`
- Artifacts: `spatialmind/{results,logs}/constraint_guided_v10_20260712`
