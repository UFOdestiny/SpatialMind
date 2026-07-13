# SpatialMind — constraint_guided_v9 Results (2026-07-12)

Explicit spatial constraint-consistency UQ. Backbone Llama-3.1-8B-Instruct, judge
Mistral-Small-3.2-24B. Cache `constraint_guided_v9` (guided decoding, 0 dropped,
leakage audit clean, counterfactual entailed→contradicted 100%). ID = StepGame
(1000/250/500). OOD = validation-adapted transfer (NOT zero-shot): target-domain
validation used only for aggregation rule + confidence normalization, never test.

## AUROC — all methods × all datasets

| method | StepGame(ID) | spartqa | babi | SpaRTUN | SpaceNLI |
|---|---:|---:|---:|---:|---:|
| **spatialmind** (main, fusion) | 0.8422 | 0.4329 | 0.5851 | 0.5200 | 0.5917 |
| constraint_no_conflict (abl) | **0.8725** | 0.4847 | 0.5109 | 0.4801 | **0.7209** |
| constraint_no_context (abl) | 0.8109 | 0.3673 | 0.5767 | 0.4440 | 0.6383 |
| constraint_no_entailment (abl) | 0.8336 | 0.4619 | 0.5819 | 0.5672 | 0.5336 |
| constraint_no_repair (abl) | 0.7755 | 0.4590 | 0.5964 | 0.5255 | 0.5650 |
| constraint_only (structure-only) | 0.8061 | 0.4767 | 0.5981 | 0.4124 | 0.6042 |
| constraint_rule (deterministic, 0 params) | 0.8025 | 0.4688 | 0.5693 | 0.4109 | 0.5629 |
| spatialmind_neural (neural probe) | 0.5380 | 0.5817 | 0.5058 | 0.6115 | 0.5370 |
| uhead | 0.5312 | 0.5766 | 0.4849 | 0.6263 | 0.5537 |
| factoscope | 0.4862 | 0.5397 | 0.5475 | 0.6295 | 0.5210 |
| mlp | 0.5419 | 0.4907 | 0.4547 | 0.6546 | 0.5422 |
| ccp | 0.5124 | 0.4342 | 0.6153 | 0.4133 | 0.5015 |
| mcp | 0.5053 | 0.4187 | 0.6172 | 0.4067 | 0.5057 |
| perplexity | 0.4855 | 0.5255 | 0.4290 | 0.5767 | 0.5140 |
| token_entropy | 0.4918 | 0.3926 | 0.6194 | 0.4013 | 0.5120 |
| random | 0.4195 | 0.4995 | 0.5373 | 0.5160 | 0.5383 |

## Findings (honest)

1. **Core novelty holds on ID.** On StepGame, every explicit-constraint variant
   (0.78–0.87) dominates every neural/hidden-state UQ baseline (0.49–0.54) by
   ~30 AUROC points. Even the deterministic 0-parameter `constraint_rule` (0.80)
   beats all 7M-param neural probes. Strong, defensible main claim.

2. **Transfer is task-representation dependent, not universal.**
   - Template-style directional tasks (StepGame, babi, SpaceNLI): constraint
     methods lead or are competitive (SpaceNLI no_conflict 0.72; babi
     constraint_only 0.60 top supervised, PR-AUC 0.70–0.75).
   - Complex-NL tasks (spartqa, SpaRTUN): the StepGame-trained constraint parser
     transfers poorly; neural probes (mlp/uhead 0.63–0.65) win. This is a real
     limitation of regex/relation parsing on free-form language and must be
     reported as a coverage variable, not hidden.

3. **The conflict/satisfiability branch is a mixed signal.** `constraint_no_conflict`
   (satisfiability + first-conflict features masked) *beats* the full main method
   on ID (0.8725 vs 0.8422) and SpaceNLI (0.72 vs 0.59), but is *worse* on babi
   (0.51 vs 0.59) and SpaRTUN. Root cause (ID diagnostic): the strongest single
   constraint signal is `conclusion_entailed` (test univariate AUROC 0.71, kept by
   no_conflict); `full_trace_feasible` (0.67) is largely redundant with it and
   `first_conflict_position` (0.56) is weak, so masking conflict removes noise the
   7M-param neural gate would otherwise overfit. The conflict branch earns its
   place only where entailment underdetermines the answer (babi).

## Open decision
Whether to (a) keep `spatialmind` as-is and frame conflict as a per-task ablation,
(b) promote `constraint_no_conflict` to the main method and reframe the novelty as
entailment+repair consistency, or (c) redesign the fusion so the conflict branch is
gated to only contribute when it adds signal. Recommend (c) diagnostic then decide;
see memory `constraint-consistency-plan`.

## Reproduce
- Train: `bash jobs/run_v9_train.sh`
- ID eval: `bash jobs/run_v9_eval_id.sh`
- OOD eval: `bash jobs/run_v9_ood.sh`
- Artifacts: `spatialmind/{results,logs}/constraint_guided_v9_20260712`
