# SpatialMind

**Constraint-grounded uncertainty quantification for spatial reasoning in LLMs.**

SpatialMind detects unreliable LLM spatial reasoning by asking whether a generated
reasoning trace can coexist with the trusted scene in one spatial model. A conservative
parser normalizes the scene and ordered claims into a canonical relation graph, and an
exact constraint engine exposes entailment, contradiction, prefix satisfiability,
first-conflict position, and minimum repair cost. These auditable symbolic diagnostics
form a reliability signal that is strong when reasoning is symbolizable and degrades when
claims cannot be grounded determinately.

The central finding: the applicability of the symbolic signal is governed by **status
determinacy** (low unknown-rate), not by raw parse coverage. This explains why a purely
symbolic score wins on templated tasks yet fails on free-form language, while a neural
probe shows the opposite pattern. SpatialMind exploits this with a **multi-signal
applicability-aware fusion** combiner that reads per-sample symbolizability/determinacy
and routes among a bank of complementary UQ signals (symbolic heads + neural probes +
training-free baselines) under limited target-validation supervision.

## Key ideas

- **Explicit spatial semantics.** Directional relations use two-axis difference
  constraints; containment, distance, topology, location, and path relations use
  inverse/symmetric/transitive closure and incompatibility rules.
- **Conflict localization and repair.** Every claim is checked against the trusted
  context plus its ordered prefix. The cache records its three-way status, the first
  infeasible step, and the minimum repair cost.
- **Applicability-aware fusion.** A logistic stacker over all base signals, gated by
  symbolizability/determinacy, selected on a validation-internal split. It routes to
  whichever signal is best per sample rather than committing to one method globally.
- **Determinacy governs applicability.** Per-dataset symbolic AUROC tracks the unknown-
  rate, not the parse-rate. The `determinacy` gate encodes this directly.
- **Fair sample-level evaluation.** All learned heads use the same claim-to-trace
  aggregation and strictly monotonic validation-only Platt scaling, so AUROC cannot be
  changed by calibration.
- **Fail-closed supervision.** The reasoning judge sees trusted premises but never the
  reference answer or final-answer verdict. Malformed/missing judgments physically remove
  the sample. A fail-closed cache audit (`scripts/leakage_audit.py`) verifies split
  disjointness and exact recomputation of every constraint tensor before training;
  aggregation, calibration, and the fusion combiner read validation labels only.

## Repository layout

```text
SpatialMind/
├── config.py                 # all tunables; every path overridable via env vars
├── data/                     # dataset adapters + cached-feature loading + claim extraction
│   ├── base.py  datasets.py  stepgame.py  spartqa.py  spartqa_yn.py
│   ├── babi.py  spartun.py   spacenli.py
│   ├── claims.py             # trace -> ordered spatial claims (+ token alignment)
│   └── cached_features.py    # trace-first dataset over data-phase chunks
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
│   └── wrapper.py            # multi-task training/inference wrapper + loss
├── scripts/                  # generate | judge | train | evaluate | fusion | diagnostics
│   ├── fusion.py             # headline multi-signal applicability-aware combiner
│   ├── fusion_pairwise.py    # 2-signal (constraint + mlp) combiner (reference)
│   ├── benchmark_fusion.py   # cross-backbone fusion comparison table
│   ├── constraint_diagnostics.py     # coverage + deterministic structural baseline
│   └── constraint_counterfactual.py  # paired relation-intervention audit
├── utils/                    # collation, results tables, downloads, efficiency, ...
├── tests/                    # pytest suite (constraints, heads, calibration, leakage)
├── jobs/                     # shell / SLURM pipeline (see "Running")
└── spatialmind/              # runtime root (symlink to scratch): models, datasets,
                              #   results/, logs/, cache/   ← all artifacts land here
```

## Pipeline

```
data    generate  vLLM JSON-schema guided decoding -> reasoning[] + conclusion; per-token
                  features (hidden state + top-k logprobs + attention-lookback) cached;
                  scene and claims parsed into canonical constraints
        judge     JSON-schema-guided LLM judge assigns trace/per-claim labels
                  (1 = supported / 0 = hallucinated)
        rebuild   native constraint view recomputed from deployment-visible inputs
        audit     fail-closed leakage audit + coverage + counterfactual diagnostics
train             train each head on cached features; best epoch by validation AUROC
eval              sample-level metrics on ID (StepGame) + 5 OOD sets, heads + baselines
val               validation-split predictions for every fusion signal
fusion            multi-signal applicability-aware combiner (validation-selected)
```

Expensive stages (generation, judging) run once and are cached; training, evaluation,
and fusion consume the cache directly.

**No test leakage.** Aggregation, calibration, and the fusion combiner are fit on a
**validation** split and only applied to test. OOD uses target validation for
calibration/thresholding/combiner-fit and is therefore **validation-adapted transfer,
not zero-shot**; no target training examples update the StepGame-trained scorer.

## Datasets & models

| Dataset | Role | Type |
| --- | --- | --- |
| **StepGame** | in-distribution | 9-way directional classification |
| **SpaRTQA** | validation-adapted OOD | 4-way multiple choice |
| **bAbI** | validation-adapted OOD | free-form QA |
| **SpaRTUN** | validation-adapted OOD | spatial NLI/QA |
| **SpaceNLI** | validation-adapted OOD | 3-way NLI |
| **SpartQA-YN** | validation-adapted OOD | block-world yes/no/DK |

Backbones (frozen, white-box): **Llama-3.1-8B-Instruct** (primary),
**Mistral-7B-Instruct-v0.3**, **gemma-2-9b-it**, **Phi-4-reasoning**.
Judge: **Mistral-Small-3.2-24B-Instruct-2506**. A StepGame-trained head is transferred to
the five OOD sets without fine-tuning.

## Head zoo

- **Ours:** `spatialmind` — exact claim/prefix/layout analysis fused with a frozen-
  representation claim probe through learned claim-local and trace-global gates.
- **Core controls:** `constraint_only` (no LLM activations), `spatialmind_neural`
  (no explicit constraints), deterministic training-free `constraint_rule`.
- **Constraint leave-one-out:** `constraint_no_context`, `constraint_no_conflict`,
  `constraint_no_entailment`, `constraint_no_repair`.
- **Supervised baselines:** `factoscope`, `uhead`, `mlp` (and further probes registered
  in `models/heads`), all claim-level.
- **Unsupervised baselines:** `random`, `mcp`, `perplexity`, `token_entropy`, `ccp`,
  `constraint_rule`.

**Metrics (sample-level):** AUROC & PR-AUC, Acc, ECE, plus NLL/Brier/AURC/risk@cov.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
pip install vllm   # needed only for the data-phase generation
```

### Runtime paths

All runtime artifacts are written under `<repo>/spatialmind/` (a symlink to fast scratch).
Override any path via environment variables; `jobs/common.sh` sets and exports them so
`config.py` resolves identically.

| Variable | Purpose | Default |
| --- | --- | --- |
| `SPATIALMIND_ROOT` | runtime root for all artifacts | `<repo>/spatialmind` |
| `MODELS_ROOT` | backbone / judge models | `$SPATIALMIND_ROOT/models` |
| `DATASETS_ROOT` | datasets | `$SPATIALMIND_ROOT/datasets` |
| `RESULTS_ROOT` | training / evaluation outputs | `$SPATIALMIND_ROOT/results/<job>` |
| `LOGS_ROOT` | logs | `$SPATIALMIND_ROOT/logs/<job>` |
| `CACHE_ROOT` | cached frozen features | `$SPATIALMIND_ROOT/cache` |
| `HF_CACHE` | Hugging Face cache | `$MODELS_ROOT/.hf_cache` |

## Running

The `jobs/` scripts encode the full flow and skip already-completed stages (idempotent,
resumable). Each backbone writes to its own isolated namespace.

```bash
# Quick end-to-end smoke test (small samples, subset of heads, all phases)
bash jobs/smoke.sh                # or: sbatch jobs/smoke.sh

# Full per-backbone pipeline (data -> train zoo -> eval ID+OOD -> val -> fusion)
bash jobs/run_llama.sh            # Llama-3.1-8B  (primary)
bash jobs/run_mistral.sh          # Mistral-7B-Instruct-v0.3
bash jobs/run_gemma.sh            # gemma-2-9b-it
bash jobs/run_phi4.sh             # Phi-4-reasoning

# All backbones sequentially (single GPU; one finishes before the next starts)
bash jobs/run_all.sh

# Run a single phase directly
sbatch jobs/phase_data.sh                        # download + generate + judge + audit
HEAD_TYPES="spatialmind uhead" bash jobs/phase_train.sh   # train a subset
bash jobs/phase_eval.sh                          # evaluate ID + OOD
```

First run only: `SKIP_DOWNLOAD=0 bash jobs/run_llama.sh` to fetch models/datasets.

### Fusion only (offline, from saved predictions)

After a run has produced `eval/`, `eval_ood/`, and `val_scores*/` for a namespace, the
combiner is pure post-processing (no GPU):

```bash
# Single backbone
python scripts/fusion.py \
    --results_root spatialmind/results/constraint_guided_v10_20260712 \
    --cache_subdir constraint_guided_v10 \
    --model Llama-3.1-8B-Instruct \
    --datasets "id:StepGame,spartqa:spartqa,babi:babi,SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpartQA_YN:SpartQA_YN"

# All backbones + SOTA table
python scripts/benchmark_fusion.py
```

### Direct script usage

```bash
# 1) Generate + cache features
python scripts/generate.py --dataset stepgame --split train,validation,test \
    --backend vllm --model_path $MODELS_ROOT/Llama-3.1-8B-Instruct --cache_dir $CACHE
# 2) Judge (trace + claim labels)
python scripts/judge.py --cache_dir $CACHE --split train,validation,test \
    --judge_model $MODELS_ROOT/Mistral-Small-3.2-24B-Instruct-2506
# 3) Train a head (best epoch by sample-level validation AUROC)
python scripts/train.py --head_type spatialmind --cache_dir $CACHE --output_dir $OUT
# 4) Sample-level evaluation (head + all unsupervised baselines)
python scripts/evaluate.py --head_path $OUT/final_model --cache_dir $CACHE --split test
python scripts/evaluate.py --cache_dir $CACHE --split test --eval_baselines
# 5) Summarize (ID / OOD comparison tables + figures)
python utils/results.py --results-root $RESULTS_ROOT --figure-dir $LOGS_ROOT/figure
```

## Output layout

```
$RESULTS_ROOT/<namespace>/
├── train/<head>/final_model/{head_weights.pth, head_config.json}   # + train_results.json
├── eval/<head>/evaluation_report.json                              # ID head
├── eval/baselines/combined_evaluation.json                         # ID unsupervised
├── eval_ood/<dataset>/<head>/evaluation_report.json                # OOD transfer
├── val_scores/<dataset>/<head>/evaluation_report.json              # validation predictions
├── val_scores_baselines/<dataset>/combined_evaluation.json         # validation baselines
└── fusion/<tag>/evaluation_report.json                             # fused UQ scores
```

Each `head_config.json` stores the validation-selected claim->trace aggregation so
evaluation reproduces the exact readout.

## License

MIT
