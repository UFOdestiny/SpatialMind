# SpatialMind

**Claim-Level Uncertainty Quantification for Trustworthy Spatial Reasoning in LLMs.**

SpatialMind detects hallucinations in the spatial reasoning of white-box LLMs. A
frozen backbone generates a `Reasoning`/`Conclusion` trace; SpatialMind decomposes
the trace into ordered spatial **claims**, scores each claim's correctness from the
model's *frozen internal features*, and aggregates the claim scores into a single
**trace-level (sample-level) reliability score**. Because the sample label is what
actually matters in deployment — *should I trust this answer?* — the headline
evaluation is sample-level for every method.

Key ideas:

- **Claim-level modeling, sample-level evaluation.** Per-claim probabilities localize
  where a spatial chain first drifts; a shared, validation-selected aggregation
  collapses them into one trace score that is compared against the trace label.
- **Multi-task head.** The SpatialMind head predicts per-claim correctness *and* a
  learned trace-level logit, so the reported sample-level metric is directly optimized.
- **Fair comparison.** Baselines (supervised probes and training-free estimators) are
  passed through the *identical* claim→trace aggregation, so differences reflect the
  quality of the claim scores — not a readout trick.
- **Post-generation & frozen.** No backbone fine-tuning and no repeated sampling; all
  backbone-dependent computation is cached once.

## Repository layout

```text
SpatialMind/
├── config.py                 # all tunables; every path overridable via env vars
├── data/                     # dataset adapters + cached-feature loading + claim extraction
│   ├── base.py  datasets.py  stepgame.py  spartqa.py  babi.py
│   ├── claims.py             # trace -> ordered spatial claims (+ token alignment)
│   └── cached_features.py    # trace-first dataset over Phase-1 chunks
├── engine/                   # vLLM / HF generation backends (feature extraction)
├── models/
│   ├── features/             # frozen token features: hidden states | logprobs | attention
│   ├── heads/                # claim-level UQ heads (see "Head zoo")
│   │   ├── base.py           # HeadOutput contract (claim logits + optional trace logit)
│   │   ├── spatialmind_head.py   # our multi-task head
│   │   ├── baselines.py      # supervised baseline probes
│   │   └── ablations.py      # SpatialMind ablation variants
│   ├── unsup_heads.py        # training-free estimators (random/mcp/ppl/entropy/ccp)
│   ├── aggregation.py        # claim -> trace score aggregation + validation selection
│   ├── wrapper.py            # multi-task training/inference wrapper + loss
│   └── inference.py          # plug-and-play LLM + UQ head adapter
├── scripts/                  # generate | claim_extract | judge | train | evaluate | metrics
├── utils/                    # collation, results tables, downloads, efficiency, ...
├── jobs/                     # SLURM / shell pipeline (see "Running")
└── spatialmind/              # runtime root (symlink to scratch): models, datasets,
                              #   results/, logs/, cache/   ← all artifacts land here
```

## Pipeline

```
Phase 0  Download        backbone + judge models, datasets
Phase 1  Generate        frozen LLM produces one trace per instance; per-token
                         features (hidden state ⊕ top-k logprobs ⊕ attention-lookback)
                         cached to disk; trace segmented into ordered claims
Phase 1b Claim extract   (optional) LLM-based claim extraction; regex fallback by default
Phase 1.5 Judge          LLM-as-judge assigns the trace label and per-claim labels
                         (1 = supported / 0 = hallucinated)
Phase 2  Train           train each head on cached features; select the best epoch by
                         SAMPLE-LEVEL validation AUROC
Phase 3  Evaluate        sample-level metrics on ID (StepGame) + OOD (SpaRTQA, bAbI)
                         for every head and every baseline; summarize
```

Expensive stages (generation, claim extraction, judging) run once and are cached;
training and evaluation consume the cache directly.

**No test leakage.** Every calibration choice — the claim→trace aggregation rule
and the baselines' confidence normalization — is fit on a **validation** split and
only applied to test. OOD is treated exactly like ID: Phase 3 generates and judges
the target dataset's *validation* split too, so OOD transfer is calibrated on OOD
validation, never on OOD test.

## Datasets & models

| Dataset | Role | Type | Labeling |
| --- | --- | --- | --- |
| **StepGame** | in-distribution | 9-way directional classification | exact match + judge for claims |
| **SpaRTQA** | OOD transfer | 4-way multiple choice | exact match + judge for claims |
| **bAbI** | OOD transfer | free-form QA | LLM-as-judge |

Backbones (frozen, white-box): **Llama-3.1-8B-Instruct** (primary), **Mistral-7B-Instruct-v0.3**,
**gemma-2-9b-it**. Judge: **Mistral-Small-3.2-24B-Instruct-2506**. A StepGame-trained
head is transferred to SpaRTQA/bAbI *without fine-tuning*.

## Head zoo

- **Ours:** `spatialmind` — claim marking → local Transformer → cross-claim BiLSTM →
  scope-aware span statistics → global-consistency discrepancy → reliability-pattern
  bank → per-claim logit **and** a learned trace logit (multi-task).
- **Supervised baselines:** `saplma`, `factoscope`, `lookback_lens`, `uhead` (LUH),
  `luh_light`, `linear`, `mlp`, `gated_mlp`, `cnn` — all claim-level, calibrated near
  a common parameter budget for a fair efficiency comparison.
- **Unsupervised baselines:** `random`, `mcp`, `perplexity`, `token_entropy`, `ccp`.
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
