#!/usr/bin/env python3
"""
Build spatialmind/results/res.txt: a comprehensive, LaTeX-formatted results
dump (same table style as latex/table/result.tex) covering all 5 datasets,
all baselines, and all 12 metrics for each base LLM — plus a per-(model,
metric,dataset) breakdown of SpatialMind's percentage improvement over the
best competing baseline (SOTA-excluding-ours).

Usage:
    python scripts/build_res_txt.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "spatialmind" / "results"
OUT_PATH = RESULTS_ROOT / "res.txt"

# ── Registry ──────────────────────────────────────────────────────────────
MODEL_JOBS = {"Llama": "36443940", "Mistral": "36443941", "Gemma": "36443942"}

# (display name, scope, ood-dir-name-or-None-for-ID)
DATASETS = [
    ("StepGame", "ID", None),
    ("bAbI", "OOD", "babi"),
    ("SpaRTQA", "OOD", "spartqa"),
    ("SpaRTUN", "OOD", "SpaRTUN"),
    ("SpaceNLI", "OOD", "SpaceNLI"),
]

# Row order: unsupervised baselines, then supervised baselines, then ours.
UNSUPERVISED = ["random", "ccp", "mcp", "perplexity", "token_entropy"]
SUPERVISED = ["saplma", "factoscope", "lookback_lens", "uhead"]
OURS = "spatialmind"
METHOD_ORDER = UNSUPERVISED + SUPERVISED + [OURS]

METHOD_DISPLAY = {
    "random": "Random", "ccp": "CCP", "mcp": "MCP", "perplexity": "Perplexity",
    "token_entropy": "Token-Entropy", "saplma": "SAPLMA", "factoscope": "Factoscope",
    "lookback_lens": "Lookback", "uhead": "UHead", "spatialmind": r"\m",
}

# Metric key (as in overall_metrics) -> (display label, lower_is_better)
METRICS: List[tuple] = [
    ("roc_auc", "AUROC", False),
    ("aurc", "AURC", True),
    ("ece", "ECE", True),
    ("pr_auc", "PR-AUC", False),
    ("accuracy", "Acc", False),
    ("f1", "F1", False),
    ("precision", "Prec", False),
    ("recall", "Recall", False),
    ("nll", "NLL", True),
    ("brier", "Brier", True),
    ("risk_at_80_cov", "Risk@80", True),
    ("risk_at_90_cov", "Risk@90", True),
]
METRIC_KEYS = [m[0] for m in METRICS]


def eval_dir_for(job: str, scope: str, ood_name: Optional[str]) -> Path:
    if scope == "ID":
        return RESULTS_ROOT / job / "eval"
    return RESULTS_ROOT / job / "eval_ood" / ood_name


def report_path(job: str, scope: str, ood_name: Optional[str], method: str) -> Path:
    base = eval_dir_for(job, scope, ood_name)
    if method in UNSUPERVISED:
        return base / "baselines" / method / "evaluation_report.json"
    return base / method / "evaluation_report.json"


_cache: Dict[Path, Dict[str, Any]] = {}


def load_metrics(job: str, scope: str, ood_name: Optional[str], method: str) -> Dict[str, float]:
    p = report_path(job, scope, ood_name, method)
    if p not in _cache:
        with p.open("r", encoding="utf-8") as f:
            _cache[p] = json.load(f)
    return _cache[p].get("overall_metrics", {})


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.3f}"


# ── Part 1: comprehensive LaTeX result tables (one per dataset) ───────────

def build_dataset_table(dataset_name: str, scope: str, ood_name: Optional[str]) -> str:
    # For each model, gather method -> {metric_key: value}, then compute
    # best/second-best per metric column within that model's method block.
    lines: List[str] = []
    n_metrics = len(METRICS)
    col_spec = "ll " + " ".join(["c"] * n_metrics)

    lines.append(r"\begin{table*}[htbp]\small")
    lines.append(
        r"\caption{Comprehensive results on " + dataset_name + " ("
        + scope + r") across all backbone models, baselines, and metrics. "
        r"Best results per model block are in \textbf{bold} and second-best "
        r"results are \underline{underlined}. $\uparrow$ indicates higher is "
        r"better, and $\downarrow$ indicates lower is better.}"
    )
    lines.append(r"\label{tab:result_" + dataset_name.lower() + "}")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_cells = [r"\textbf{Model}", r"\textbf{Method}"]
    for _, label, lower_better in METRICS:
        arrow = r"$\downarrow$" if lower_better else r"$\uparrow$"
        header_cells.append(f"{label}{arrow}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for model_idx, (model, job) in enumerate(MODEL_JOBS.items()):
        lines.append(f"% {'=' * 20} {model} {'=' * 20}")
        lines.append(r"\multirow{" + str(len(METHOD_ORDER)) + "}{*}{" + model + "}")

        # method -> metric_key -> value
        values: Dict[str, Dict[str, Optional[float]]] = {}
        for method in METHOD_ORDER:
            om = load_metrics(job, scope, ood_name, method)
            values[method] = {k: om.get(k) for k in METRIC_KEYS}

        # per-metric best/second-best among all methods in this model block
        best_second: Dict[str, tuple] = {}
        for key, _, lower_better in METRICS:
            vals = [(m, values[m][key]) for m in METHOD_ORDER if values[m][key] is not None]
            if not vals:
                best_second[key] = (None, None)
                continue
            ordered = sorted(vals, key=lambda kv: kv[1], reverse=not lower_better)
            best = ordered[0][0]
            second = ordered[1][0] if len(ordered) > 1 else None
            best_second[key] = (best, second)

        for method in METHOD_ORDER:
            is_ours = method == OURS
            cells = ["", r"\textbf{" + METHOD_DISPLAY[method] + "}" if is_ours else METHOD_DISPLAY[method]]
            for key, _, _lower in METRICS:
                v = values[method][key]
                s = fmt(v)
                best_m, second_m = best_second[key]
                if method == best_m:
                    s = r"\textbf{" + s + "}"
                elif method == second_m:
                    s = r"\underline{" + s + "}"
                cells.append(s)
            prefix = ""
            if is_ours:
                lines.append(r"\rowcolor{gray!15}")
                cells[0] = r"\cellcolor{white}"
            lines.append(" & ".join(cells) + r" \\")

        if model_idx < len(MODEL_JOBS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ── Part 2: SpatialMind vs. SOTA (best non-ours baseline) improvement % ───

def compute_improvements() -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("SpatialMind vs. SOTA (best competing baseline) — percentage improvement")
    lines.append("Per (Model, Metric, Dataset). Positive % always means SpatialMind is better.")
    lines.append("=" * 100)

    for model, job in MODEL_JOBS.items():
        lines.append("")
        lines.append(f"### {model}")
        wins, losses, ties = 0, 0, 0
        for key, label, lower_better in METRICS:
            lines.append(f"  [{label}]")
            for dataset_name, scope, ood_name in DATASETS:
                ours_val = load_metrics(job, scope, ood_name, OURS).get(key)
                sota_method, sota_val = None, None
                for method in UNSUPERVISED + SUPERVISED:
                    v = load_metrics(job, scope, ood_name, method).get(key)
                    if v is None:
                        continue
                    if sota_val is None or (v < sota_val if lower_better else v > sota_val):
                        sota_val, sota_method = v, method

                if ours_val is None or sota_val is None:
                    lines.append(f"    {dataset_name:10s}: -- (missing data)")
                    continue

                sota_label = METHOD_DISPLAY[sota_method].replace(r"\m", "SpatialMind")
                if sota_val == 0:
                    if ours_val == 0:
                        pct_str = "+0.00% (tie, both 0)"
                        ties += 1
                    else:
                        pct_str = "N/A (SOTA=0)"
                else:
                    diff = (sota_val - ours_val) if lower_better else (ours_val - sota_val)
                    pct = diff / sota_val * 100.0
                    pct_str = f"{pct:+.2f}%"
                    if pct > 0:
                        wins += 1
                    elif pct < 0:
                        losses += 1
                    else:
                        ties += 1
                lines.append(
                    f"    {dataset_name:10s}: SpatialMind={ours_val:.4f}  "
                    f"SOTA={sota_val:.4f} ({sota_label:14s})  "
                    f"improvement={pct_str}"
                )
        total = wins + losses + ties
        lines.append(
            f"  -> {model} summary: SpatialMind beats SOTA in {wins}/{total} "
            f"(metric, dataset) cells, ties in {ties}, loses in {losses}."
        )
    return "\n".join(lines)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    parts.append("=" * 100)
    parts.append("Part 1: Comprehensive results tables (format of latex/table/result.tex)")
    parts.append("All 5 datasets x all baselines x all 12 metrics, per base LLM.")
    parts.append("=" * 100)
    parts.append("")

    for dataset_name, scope, ood_name in DATASETS:
        parts.append(build_dataset_table(dataset_name, scope, ood_name))
        parts.append("")

    parts.append(compute_improvements())
    parts.append("")

    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
