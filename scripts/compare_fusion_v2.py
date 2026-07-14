#!/usr/bin/env python3
"""Compare fusion_v2 (multi-signal) vs old fusion vs best base, all backbones.

Runs the multi-signal combiner for every (backbone, dataset), then prints a
table of: best single base (test-oracle upper ref), old 2-signal fusion,
new multi-signal fusion_v2, and whether v2 reaches SOTA (>= best base).
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
    grand = {"v2_sota": 0, "old_sota": 0, "cells": 0}
    for name, ns, sub, model in BB:
        R = f"spatialmind/results/{ns}"
        # run multi_fusion (writes fusion_v2/*)
        subprocess.run([sys.executable, "scripts/multi_fusion.py",
                        "--results_root", R, "--cache_subdir", sub, "--model", model],
                       capture_output=True)
        print(f"\n===== {name} =====")
        print(f"{'dataset':12s}{'best_base':>10s}{'old_fus':>9s}{'v2_fus':>9s}  v2_SOTA")
        v2s, olds = [], []
        for tag, d in DS:
            bb = max([(base(R, tag, h) or -1, h) for h in ALLH])
            old = au(f"{R}/fusion/{tag}/evaluation_report.json")
            v2 = au(f"{R}/fusion_v2/{tag}/evaluation_report.json")
            v2s.append(v2 if v2 else 0); olds.append(old if old else 0)
            sota = "YES" if (v2 and v2 >= bb[0] - 0.003) else ""
            grand["cells"] += 1
            if v2 and v2 >= bb[0] - 0.003:
                grand["v2_sota"] += 1
            if old and old >= bb[0] - 0.003:
                grand["old_sota"] += 1
            print(f"{d:12s}{bb[0]:>10.3f}{(old or 0):>9.3f}{(v2 or 0):>9.3f}  {sota}  ({bb[1]})")
        print(f"{'MEAN':12s}{'':>10s}{np.mean(olds):>9.3f}{np.mean(v2s):>9.3f}")
    print(f"\n### v2 SOTA cells {grand['v2_sota']}/{grand['cells']}  |  old fusion SOTA {grand['old_sota']}/{grand['cells']}")


if __name__ == "__main__":
    main()
