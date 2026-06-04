# SpatialMind

SpatialMind is a research codebase for claim-level uncertainty estimation in spatial reasoning pipelines.

The repository includes:

- dataset adapters and cached-feature loading
- generation backends and feature extraction
- the SpatialMind claim-aware UQ head
- unsupervised reference baselines used by evaluation
- generation, judging, training, evaluation, and result-summary scripts

## Repository layout

```text
SpatialMind/
├── config.py
├── data/
├── engine/
├── models/
│   ├── features/
│   └── heads/
├── scripts/
└── utils/
```

## Setup

Use Python 3.10+ and install the packages needed by the scripts you plan to run.

```bash
pip install -r requirements.txt
```

If you need vLLM generation or Weights & Biases logging, install them separately:

```bash
pip install vllm wandb
```

## Runtime paths

By default, runtime artifacts are written under `./artifacts/` inside the repository. You can override everything through environment variables.

| Variable | Purpose | Default |
| --- | --- | --- |
| `SPATIALMIND_ROOT` | Base directory for runtime artifacts | `<repo>/artifacts` |
| `MODELS_ROOT` | Local model directory | `$SPATIALMIND_ROOT/models` |
| `DATASETS_ROOT` | Local dataset directory | `$SPATIALMIND_ROOT/datasets` |
| `RESULTS_ROOT` | Training and evaluation outputs | `$SPATIALMIND_ROOT/results` |
| `LOGS_ROOT` | Log directory | `$SPATIALMIND_ROOT/logs` |
| `HF_CACHE` | Hugging Face cache directory | `$MODELS_ROOT/.hf_cache` |

## Typical workflow

### 1. Generate cached features

```bash
python scripts/generate.py \
  --dataset stepgame \
  --split train,validation,test \
  --backend vllm
```

### 2. Optionally extract claims later

```bash
python scripts/claim_extract.py \
  --cache_dir /path/to/cache \
  --split train,validation,test \
  --claim_extractor_model /path/to/model
```

### 3. Judge free-form outputs when needed

```bash
python scripts/judge.py \
  --cache_dir /path/to/cache \
  --split test \
  --judge_model /path/to/judge-model
```

### 4. Train a supervised head

```bash
python scripts/train.py \
  --head_type uq \
  --cache_dir /path/to/cache \
  --output_dir /path/to/output
```

### 5. Evaluate a trained head

```bash
python scripts/evaluate.py \
  --head_path /path/to/output/final_model \
  --cache_dir /path/to/cache \
  --split test
```

### 6. Evaluate unsupervised reference baselines

```bash
python scripts/evaluate.py \
  --cache_dir /path/to/cache \
  --split test \
  --eval_baselines
```

### 7. Summarize results

```bash
python utils/results.py --results-root /path/to/results
```

## Task adapters

The repository includes the dataset adapters currently used by the pipeline:

- `stepgame`
- `spartqa`
- `babi`

See `data/datasets.py` for the registry and constructor names.

## Notes

- The supervised model in this repository is the SpatialMind claim-aware UQ head, exposed as `uq`.
- The unsupervised reference baselines used by `scripts/evaluate.py` are also available.
- Download helper scripts are included. Configure model and dataset entries in `config.py` or pass them through CLI arguments.

## License

MIT
