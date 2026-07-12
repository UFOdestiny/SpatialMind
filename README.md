# SpatialMind

**Constraint-grounded uncertainty quantification for spatial reasoning in LLMs.**

SpatialMind detects hallucinations by asking whether a generated reasoning trace
can coexist with the trusted scene in one spatial model. A conservative parser
normalizes the scene and ordered claims into a canonical relation graph. An exact
constraint engine then exposes entailment, contradiction, prefix satisfiability,
first-conflict position, and minimum repair cost. The primary head fuses these
explicit signals with frozen white-box LLM features and outputs per-claim trust;
the same validation-selected claim aggregation is used for every learned method.

Key ideas:

- **Explicit spatial semantics.** Directional relations use two-axis difference
  constraints; containment, distance, topology, location, and path relations use
  inverse/symmetric/transitive closure and incompatibility rules.
- **Conflict localization and repair.** Every claim is checked against the trusted
  context plus its ordered prefix. The cache records its three-way status, the first
  infeasible step, and how many prior generated relations must be removed to repair it.
- **Constraint-energy residual fusion.** The exact solver defines an interpretable
  reliability prior; a gated frozen-representation probe learns only a bounded residual
  for parser ambiguity and latent evidence. `constraint_only`, `constraint_rule`, and
  `spatialmind_neural` isolate the learned-symbolic, fixed-symbolic, and neural sides.
- **Fair sample-level evaluation.** All learned heads use the same claim-to-trace
  aggregation and strictly monotonic validation-only Platt scaling. AUROC cannot be
  changed by calibration. The optional non-monotonic structural calibrator is a
  separately requested secondary analysis, never the headline protocol.
- **Auditable failure surface.** Parser coverage, unknown relations, infeasible source
  contexts, deterministic-rule performance, and paired counterfactual sensitivity are
  reported rather than hidden as preprocessing details.
- **Fail-closed supervision.** The reasoning judge sees trusted premises but never the
  reference answer or final-answer verdict. Malformed/missing claim judgments cause the
  entire sample to be physically removed and counted—labels are never padded, inferred,
  or copied from the trace label. A cache audit verifies split disjointness and exact
  recomputation of every constraint tensor before training.

## Repository layout

```text
SpatialMind/
├── config.py                 # all tunables; every path overridable via env vars
├── data/                     # dataset adapters + cached-feature loading + claim extraction
│   ├── base.py  datasets.py  stepgame.py  spartqa.py  babi.py
│   ├── claims.py             # trace -> ordered spatial claims (+ token alignment)
│   └── cached_features.py    # trace-first dataset over Phase-1 chunks
├── engine/                   # vLLM / HF generation backends (feature extraction)
├── spatial_constraints/      # relation ontology, parser, exact solver, trace analysis
├── models/
│   ├── features/             # frozen token features: hidden states | logprobs | attention
│   ├── heads/                # claim-level UQ heads (see "Head zoo")
│   │   ├── base.py           # HeadOutput contract (claim logits + optional trace logit)
│   │   ├── spatialmind_head.py   # hybrid method + structure/neural ablations
│   │   ├── baselines.py      # supervised baseline probes
│   │   └── ablations.py      # SpatialMind ablation variants
│   ├── unsup_heads.py        # training-free estimators (random/mcp/ppl/entropy/ccp)
│   ├── aggregation.py        # claim -> trace score aggregation + validation selection
│   ├── wrapper.py            # multi-task training/inference wrapper + loss
│   └── inference.py          # plug-and-play LLM + UQ head adapter
├── scripts/                  # generate | judge | train | evaluate | diagnostics
│   ├── constraint_diagnostics.py     # coverage + deterministic structural baseline
│   └── constraint_counterfactual.py  # paired relation-intervention audit
├── utils/                    # collation, results tables, downloads, efficiency, ...
├── jobs/                     # SLURM / shell pipeline (see "Running")
└── spatialmind/              # runtime root (symlink to scratch): models, datasets,
                              #   results/, logs/, cache/   ← all artifacts land here
```

## Pipeline

```
Phase 0  Download        backbone + judge models, datasets
Phase 1  Generate        vLLM JSON-schema guided decoding produces reasoning[] +
                         conclusion; per-token
                         features (hidden state ⊕ top-k logprobs ⊕ attention-lookback)
                         cached; scene and claims parsed into canonical constraints
Phase 1b Claim extract   optional JSON-schema-guided external claim extraction
Phase 1.5 Judge          JSON-schema-guided LLM judge assigns trace/per-claim labels
                         (1 = supported / 0 = hallucinated)
Phase 1.6 Audit          parser coverage, constraint-rule baseline, source feasibility,
                         and paired original→contradictory relation interventions
Phase 2  Train           train each head on cached features; select the best epoch by
                         SAMPLE-LEVEL validation AUROC
Phase 3  Evaluate        sample-level metrics on ID (StepGame) + OOD (SpaRTQA, bAbI)
                         for every head and every baseline; summarize
```

Expensive stages (generation, claim extraction, judging) run once and are cached;
training and evaluation consume the cache directly.

**No test leakage.** Aggregation and calibration are fit on a **validation** split and
only applied to test. Headline calibration has a positive slope by construction, so
it preserves rankings. OOD uses target validation for calibration/thresholding and is
therefore **validation-adapted transfer, not zero-shot transfer**; no target training
examples update the StepGame-trained scorer.

## Datasets & models

| Dataset | Role | Type | Labeling |
| --- | --- | --- | --- |
| **StepGame** | in-distribution | 9-way directional classification | exact match + judge for claims |
| **SpaRTQA** | validation-adapted OOD | 4-way multiple choice | exact match + judge for claims |
| **bAbI** | validation-adapted OOD | free-form QA | LLM-as-judge |
| **SpaRTUN** | validation-adapted OOD | spatial NLI/QA | exact match + judge for claims |
| **SpaceNLI** | validation-adapted OOD | 3-way NLI | exact match + judge for claims |

Backbones (frozen, white-box): **Llama-3.1-8B-Instruct** (primary), **Mistral-7B-Instruct-v0.3**,
**gemma-2-9b-it**. Judge: **Mistral-Small-3.2-24B-Instruct-2506**. A StepGame-trained
head is transferred to SpaRTQA/bAbI *without fine-tuning*.

## Head zoo

- **Ours:** `spatialmind` — exact claim/prefix/layout analysis fused with a frozen-
  representation claim probe through learned claim-local and trace-global gates.
- **Core controls:** `constraint_only` (no LLM activations), `spatialmind_neural`
  (no explicit constraints), and deterministic training-free `constraint_rule`.
- **Constraint leave-one-out:** `constraint_no_context`, `constraint_no_conflict`,
  `constraint_no_entailment`, and `constraint_no_repair`.
- **Supervised baselines:** `saplma`, `factoscope`, `lookback_lens`, `uhead` (LUH),
  `luh_light`, `linear`, `mlp`, `gated_mlp`, `cnn` — all claim-level, calibrated near
  a common parameter budget for a fair efficiency comparison.
- **Unsupervised baselines:** `random`, `mcp`, `perplexity`, `token_entropy`, `ccp`,
  `constraint_rule`.
- **Ablations (cumulative):** `abl_base` → `abl_cross` → `abl_type` → `abl_scope` → full.
  **(leave-one-out):** `abl_no_cross`, `abl_no_type`, `abl_no_scope`, `abl_no_bank`.

**Metrics (sample-level):** AUROC & PR-AUC (ranking reliable vs. unreliable traces),
Acc (final-answer decision at 0.5), ECE (calibration), plus NLL/Brier/AURC/risk@cov.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
# vLLM is needed only for Phase-1 generation; install separately:
pip install vllm
```

### Runtime paths

By default **all** runtime artifacts are written under `<repo>/spatialmind/` (a symlink
to fast scratch storage). Override any path via environment variables — this is the only
thing you edit to move machines.

| Variable | Purpose | Default |
| --- | --- | --- |
| `SPATIALMIND_ROOT` | runtime root for all artifacts | `<repo>/spatialmind` |
| `MODELS_ROOT` | backbone / judge models | `$SPATIALMIND_ROOT/models` |
| `DATASETS_ROOT` | datasets | `$SPATIALMIND_ROOT/datasets` |
| `RESULTS_ROOT` | training / evaluation outputs | `$SPATIALMIND_ROOT/results/<job>` |
| `LOGS_ROOT` | logs | `$SPATIALMIND_ROOT/logs/<job>` |
| `CACHE_ROOT` | cached frozen features | `$SPATIALMIND_ROOT/cache` |
| `HF_CACHE` | Hugging Face cache | `$MODELS_ROOT/.hf_cache` |

`jobs/common.sh` sets these explicitly and exports them so `config.py` resolves identically.

## Running

The `jobs/` scripts encode the full flow and skip already-completed stages.

```bash
# Quick end-to-end smoke test (small samples, subset of heads, all phases)
sbatch jobs/example.sh            # or: bash jobs/example.sh   (no SLURM)

# Full per-backbone pipeline (generate → train zoo → evaluate ID+OOD → summary)
sbatch jobs/pipeline1.sh          # Llama-3.1-8B  (primary)
sbatch jobs/pipeline2.sh          # Mistral-7B-Instruct-v0.3
sbatch jobs/pipeline3.sh          # gemma-2-9b-it
RESUME_JOB_ID=<id> sbatch jobs/pipeline1.sh   # resume a prior run

# Run a single phase
sbatch jobs/p1.sh                 # download + generate + judge
HEAD_TYPES="spatialmind uhead" bash jobs/p2.sh   # train a subset
bash jobs/p3.sh                   # evaluate ID + OOD
```

First run only: `SKIP_DOWNLOAD=0 sbatch jobs/pipeline1.sh` to fetch models/datasets.

### Direct script usage

```bash
# 1) Generate + cache features
python scripts/generate.py --dataset stepgame --split train,validation,test \
    --backend vllm --model_path $MODELS_ROOT/Llama-3.1-8B-Instruct --cache_dir $CACHE

# 2) Judge (trace + claim labels)
python scripts/judge.py --cache_dir $CACHE --split train,validation,test \
    --judge_model $MODELS_ROOT/Mistral-Small-3.2-24B-Instruct-2506

# 3) Train a head (best epoch chosen by sample-level validation AUROC)
python scripts/train.py --head_type spatialmind --cache_dir $CACHE --output_dir $OUT

# 4) Sample-level evaluation (head + all unsupervised baselines)
python scripts/evaluate.py --head_path $OUT/final_model --cache_dir $CACHE --split test
python scripts/evaluate.py --cache_dir $CACHE --split test --eval_baselines

# 5) Summarize (ID / OOD comparison tables + figures)
python utils/results.py --results-root $RESULTS_ROOT --figure-dir $LOGS_ROOT/figure
```

## Output layout

```
$RESULTS_ROOT/<job>/
├── train/<head>/final_model/{head_weights.pth, head_config.json}   # + train_results.json
├── eval/<head>/evaluation_report.json                              # ID head
├── eval/baselines/<name>/evaluation_report.json                    # ID unsupervised
└── eval_ood/<dataset>/<head>/evaluation_report.json                # OOD transfer
```

Each `head_config.json` stores the validation-selected claim→trace aggregation so
evaluation reproduces the exact readout.

## License

MIT
