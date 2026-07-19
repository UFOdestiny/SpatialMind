# SpatialMind

SpatialMind estimates whether an LLM's spatial-reasoning trace is correct.

## Idea

1. **Layout Auditor** parses the scene and ordered trace into spatial constraints.
   Each parsed claim is entailed, contradicted, unknown, or not evaluable after
   an earlier conflict. It also records prefix feasibility and bounded repair cost.
2. **Determinacy Profile** measures whether the auditor can usefully evaluate a
   trace: symbolizability \(\pi\), determinacy \(d\), and applicability
   \(\delta=\pi d\).
3. **DARC composer** screens and orients validation-informative constraint,
   representation, and decoding scores. It learns a regularized logistic
   composer over scores and their interactions with \(\pi\) and \(\delta\),
   then selects a monotone, Brier-aware calibrator using target validation only.

The generator and source-trained scorers remain frozen during target adaptation.
No test labels are used for score selection, standardization, composition, or
calibration.

## Structure

```text
data/                 dataset adapters and cached feature loader
spatial_constraints/  parser, relation algebra, solver, trace auditor
models/               frozen-feature scorers and baselines
scripts/              generation, claim labeling, training, evaluation, DARC
jobs/                 resumable end-to-end and Slurm entrypoints
tests/                auditor, cache, scorer, calibration, leakage tests
```

## Run

Install Python 3.10+ dependencies:

```bash
pip install -r requirements.txt
pip install vllm  # required for generation
```

Set site-local paths or a Slurm account in `.vscode/.env` (start from
`.env.example`). The file is ignored by Git and loaded by the job scripts.

Run one backbone end to end:

```bash
MODEL_NAME=Llama-3.1-8B-Instruct RUN_TAG=llama bash jobs/run_backbone.sh
```

The pipeline generates and audits traces, trains source scorers on StepGame,
evaluates the four transfer targets, fits DARC on each target validation split,
and writes artifacts under `spatialmind/{cache,results,logs}/constraint_guided_<tag>`.

For a quick integration check:

```bash
bash jobs/smoke.sh
PYTHONPATH=. pytest -q
```
