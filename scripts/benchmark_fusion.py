#!/usr/bin/env python3
"""Benchmark the SpatialMind fusion combiner across all backbones.

For every (backbone, dataset) this:
  1. runs the multi-signal combiner (scripts/fusion.py), writing fusion/*;
  2. prints a table of the best single base signal (test-oracle upper reference),
     the pairwise 2-signal baseline (fusion_pairwise/*, if present), the
     multi-signal fusion, and whether fusion reaches SOTA (>= best base).

The pairwise column is a reference only; it is populated by running
scripts/fusion_pairwise.py first (or by the pipeline). Missing pairwise
reports simply show 0.000.
"""
import json, os, sys, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.metrics import compute_all_metrics  # noqa

BB = [("Llama", "constraint_guided_v10_20260712", "constraint_guided_v10", "Llama-3.1-8B-Instruct"),
      ("Mistral7B", "constraint_guided_v10_mistral7b", "constraint_guided_v10_mistral7b", "Mistral-7B-Instruct-v0.3"),
      ("Gemma2", "constraint_guided_v10_gemma2", "constraint_guided_v10_gemma2", "gemma-2-9b-it"),
      ("Phi4", "constraint_guided_v10_phi4reason", "constraint_guided_v10_phi4reason", "Phi-4-reasoning")]
DS = [("id", "StepGame"), ("spartqa", "SpaRTQA"), ("babi", "bAbI"),
      ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI"), ("SpartQA_YN", "SpartQA-YN")]
ALLH = ["constraint_no_conflict", "constraint_only", "spatialmind_neural", "spatialmind",
        "mlp", "uhead", "factoscope", "constraint_rule", "ccp", "mcp", "perplexity", "token_entropy"]


def au(p):
    try:
        return json.load(open(p))["overall_metrics"]["roc_auc"]
    except Exception:
        return None


def base(R, tag, h):
    b = f"{R}/eval" if tag == "id" else f"{R}/eval_ood/{tag}"
    p = f"{b}/{h}/evaluation_report.json"
    if not os.path.exists(p):
        p = f"{b}/baselines/{h}/evaluation_report.json"
    return au(p)


def main():
    grand = {"fusion_sota": 0, "pairwise_sota": 0, "cells": 0}
    for name, ns, sub, model in BB:
        R = f"spatialmind/results/{ns}"
        # run the multi-signal combiner (writes fusion/*)
        subprocess.run([sys.executable, "scripts/fusion.py",
                        "--results_root", R, "--cache_subdir", sub, "--model", model],
                       capture_output=True)
        print(f"\n===== {name} =====")
        print(f"{'dataset':12s}{'best_base':>10s}{'pairwise':>10s}{'fusion':>9s}  SOTA")
        fus, pair = [], []
        for tag, d in DS:
            bb = max([(base(R, tag, h) or -1, h) for h in ALLH])
            p = au(f"{R}/fusion_pairwise/{tag}/evaluation_report.json")
            f = au(f"{R}/fusion/{tag}/evaluation_report.json")
            fus.append(f if f else 0); pair.append(p if p else 0)
            sota = "YES" if (f and f >= bb[0] - 0.003) else ""
            grand["cells"] += 1
            if f and f >= bb[0] - 0.003:
                grand["fusion_sota"] += 1
            if p and p >= bb[0] - 0.003:
                grand["pairwise_sota"] += 1
            print(f"{d:12s}{bb[0]:>10.3f}{(p or 0):>10.3f}{(f or 0):>9.3f}  {sota}  ({bb[1]})")
        print(f"{'MEAN':12s}{'':>10s}{np.mean(pair):>10.3f}{np.mean(fus):>9.3f}")
    print(f"\n### fusion SOTA cells {grand['fusion_sota']}/{grand['cells']}  "
          f"|  pairwise SOTA {grand['pairwise_sota']}/{grand['cells']}")


if __name__ == "__main__":
    main()
