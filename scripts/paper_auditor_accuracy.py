#!/usr/bin/env python3
"""Direct validation of the Layout Auditor's local outputs (P5 in review).

The reviewer asks whether the auditor's per-claim verdicts are themselves
reliable, not just whether they help downstream UQ. We score the auditor's
three-valued status against the per-claim reference label already stored in the
cache (`verified`: 1 = consistent/entailed by the reference, 0 = inconsistent),
which for reasoning claims is the GT-supervised / judge label and for the
conclusion claim is the deterministic gold-answer correctness (see judge.py).

For every parsed claim with a DETERMINATE auditor verdict we compare:
  * entailed   -> expect verified == 1   (entailment precision)
  * contradicted -> expect verified == 0 (contradiction precision)
Unknown verdicts are reported as a coverage/abstention rate, not scored for
precision (the auditor makes no claim there, by design).

We also report the determinate-subset status accuracy and a confusion matrix,
per dataset and per relation type. This is an automatic reference-based
evaluation; a separate human spot-check is reported in the appendix table.

Pure post-processing over the cache. No GPU. Writes
spatialmind/results/<ns>/fusion/auditor_accuracy.json.
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np
import torch

BACKBONE = os.environ.get("AUDIT_BACKBONE", "llama")
_MODELS = {"llama": "Llama-3.1-8B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3",
           "gemma": "gemma-2-9b-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
MODEL = _MODELS[BACKBONE]
SUB = f"constraint_guided_{BACKBONE}"
R = f"spatialmind/results/constraint_guided_{BACKBONE}"
CACHE_ROOT = f"spatialmind/cache/cached_features/{SUB}"
# (cache_folder, display)
DATASETS = [("StepGame", "StepGame"), ("spartqa", "SpaRTQA"), ("SpaRTUN", "SpaRTUN"),
            ("SpaceNLI", "SpaceNLI"), ("SpaRP_PS3", "SpaRP")]


def iter_rows(cache_folder, split="test"):
    d = f"{CACHE_ROOT}/{cache_folder}/{MODEL}/{split}"
    for c in sorted(glob.glob(f"{d}/chunk_*.pt")):
        for row in torch.load(c, map_location="cpu", weights_only=False):
            yield row


def score_dataset(cache_folder):
    n_claim = 0
    n_parsed = 0
    det = {"entailed": 0, "contradicted": 0}         # determinate verdict counts
    correct = {"entailed": 0, "contradicted": 0}     # verdict matches reference
    n_unknown = 0
    by_rel = {}                                       # relation -> [n_det, n_correct]
    conf = {"ent_v1": 0, "ent_v0": 0, "con_v1": 0, "con_v0": 0}
    for row in iter_rows(cache_folder):
        ca = row.get("constraint_analysis") or {}
        claims = ca.get("claims") or []
        verified = row.get("verified") or []
        # constraint_analysis.claims align 1:1 with row['claims']/verified
        for i, c in enumerate(claims):
            if i >= len(verified):
                break
            v = verified[i]
            try:
                v = int(v)
            except Exception:
                v = -1
            if v not in (0, 1):
                continue                              # no reference label
            n_claim += 1
            parsed = c.get("parsed") or []
            status = str(c.get("status", "unknown")).lower()
            if not parsed:
                continue
            n_parsed += 1
            if status == "unknown":
                n_unknown += 1
                continue
            if status not in ("entailed", "contradicted"):
                continue
            det[status] += 1
            rel = parsed[0].get("relation", "?")
            by_rel.setdefault(rel, [0, 0])
            by_rel[rel][0] += 1
            ok = (status == "entailed" and v == 1) or (status == "contradicted" and v == 0)
            if ok:
                correct[status] += 1
                by_rel[rel][1] += 1
            if status == "entailed":
                conf["ent_v1" if v == 1 else "ent_v0"] += 1
            else:
                conf["con_v1" if v == 1 else "con_v0"] += 1
    ndet = det["entailed"] + det["contradicted"]
    ncorr = correct["entailed"] + correct["contradicted"]
    out = {
        "n_labeled_claims": n_claim,
        "n_parsed": n_parsed,
        "parse_rate": round(n_parsed / n_claim, 4) if n_claim else None,
        "determinate_coverage": round(ndet / n_parsed, 4) if n_parsed else None,
        "unknown_rate_parsed": round(n_unknown / n_parsed, 4) if n_parsed else None,
        "status_accuracy_determinate": round(ncorr / ndet, 4) if ndet else None,
        "entailment_precision": round(correct["entailed"] / det["entailed"], 4) if det["entailed"] else None,
        "contradiction_precision": round(correct["contradicted"] / det["contradicted"], 4) if det["contradicted"] else None,
        "n_entailed": det["entailed"], "n_contradicted": det["contradicted"],
        "confusion": conf,
        "by_relation": {k: {"n": v[0], "acc": round(v[1] / v[0], 4)}
                        for k, v in sorted(by_rel.items(), key=lambda x: -x[1][0]) if v[0] >= 20},
    }
    return out


def main():
    out = {"backbone": BACKBONE, "datasets": {}}
    for cache_folder, disp in DATASETS:
        try:
            out["datasets"][disp] = score_dataset(cache_folder)
        except Exception as e:
            out["datasets"][disp] = {"error": str(e)}
    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/auditor_accuracy.json", "w"), indent=2)
    # summary
    print(f"{'dataset':10s}{'parse':>8s}{'detCov':>8s}{'stAcc':>8s}{'entP':>8s}{'conP':>8s}   nDet")
    for disp, d in out["datasets"].items():
        if "error" in d:
            print(f"{disp:10s}  {d['error']}"); continue
        def g(k):
            v = d.get(k); return f"{v:.3f}" if isinstance(v, float) else "--"
        nd = (d.get("n_entailed") or 0) + (d.get("n_contradicted") or 0)
        print(f"{disp:10s}{g('parse_rate'):>8s}{g('determinate_coverage'):>8s}"
              f"{g('status_accuracy_determinate'):>8s}{g('entailment_precision'):>8s}"
              f"{g('contradiction_precision'):>8s}   {nd}")


if __name__ == "__main__":
    main()
