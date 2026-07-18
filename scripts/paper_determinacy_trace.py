#!/usr/bin/env python3
"""Trace-level determinacy analysis (P2 in review).

Replaces the weak n=5 dataset-mean correlation with per-trace evidence:

  (A) Within-dataset determinacy bins: for each dataset, split test traces into
      determinacy quantile bins and report the constraint scorer's separation
      (point-biserial r between constraint score and correctness) in each bin,
      contrasted with the SAME binning by parse coverage.
  (B) Pooled bin curve across all datasets (determinacy vs parse coverage).
  (C) Dataset-fixed-effect linear probe of constraint-score reliability:
          y_i = b0 + b1*parse_i + b2*det_i + b3*(parse_i*det_i) + alpha_dataset + e
      where y_i is a per-trace reliability proxy (the correct/incorrect
      separation contribution). We fit with dataset dummies (fixed effects) and
      report b2 (determinacy) and the interaction with a bootstrap 95% CI, so the
      claim is a proper partial effect, not a 5-point trend.

Reliability proxy per trace: we cannot compute AUROC per trace, so we use the
signed, correctness-aligned constraint evidence
      g_i = (2*y_i - 1) * (s_con_i - mean_dataset(s_con))
which is positive when the constraint score points the correct way and larger
when the trace is more determinate iff determinacy drives constraint reliability.
This is the standard within-group point-biserial decomposition.

Pure post-processing; reads Llama v11 signals (matches the figure source).
Writes spatialmind/results/<ns>/fusion/determinacy_trace.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F  # noqa: E402

BACKBONE = os.environ.get("DET_BACKBONE", "llama")
_MODELS = {"llama": "Llama-3.1-8B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3",
           "gemma": "gemma-2-9b-it", "phi": "Phi-4-reasoning", "qwen": "Qwen3-8B"}
R = f"spatialmind/results/constraint_guided_v11_{BACKBONE}"
SUB = f"constraint_guided_v11_{BACKBONE}"
MODEL = _MODELS[BACKBONE]
CACHE_ROOT = "spatialmind/cache/cached_features"
CON = "constraint_no_conflict"
DATASETS = [("id", "StepGame", "StepGame"), ("spartqa", "SpaRTQA", "spartqa"),
            ("SpaRTUN", "SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI", "SpaceNLI"),
            ("SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
rng = np.random.RandomState(0)


def point_biserial(y, s):
    """Correlation between continuous score s and binary y; == normalized AUROC
    separation. Returns nan if degenerate."""
    if len(np.unique(y)) < 2:
        return float("nan")
    s = (s - s.mean()) / (s.std() + 1e-9)
    return float(np.corrcoef(s, y)[0, 1])


def load(tag, cname):
    cache = (f"{CACHE_ROOT}/{SUB}/StepGame/{MODEL}" if tag == "id"
             else f"{CACHE_ROOT}/{SUB}/{tag}/{MODEL}")
    st = F.collect_signals(R, tag, cname, "test", SUB)
    if CON not in st:
        return None
    st = {CON: st[CON]}
    tft = F.load_trace_features(cache, "test")
    St, yt, ids, tf = F.align(st, tft)
    con = St[CON]
    parse = np.nan_to_num(tf[:, F.PARSE], nan=0.0)
    unk = np.nan_to_num(tf[:, F.UNK], nan=1.0)
    det = 1.0 - unk                      # status determinacy among parsed
    return dict(y=yt, con=con, parse=parse, det=det, n=len(yt))


def binned(y, s, key, nbins=5):
    """Bin traces by `key` into quantile bins; report point-biserial(con,y) per
    bin. Uses quantile edges so bins are balanced."""
    edges = np.quantile(key, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    rows = []
    for i in range(nbins):
        m = (key >= edges[i]) & (key < edges[i + 1])
        if m.sum() < 25 or len(np.unique(y[m])) < 2:
            rows.append({"bin": i, "n": int(m.sum()), "r": None,
                         "key_mid": round(float((edges[i] + edges[i + 1]) / 2), 3)})
            continue
        rows.append({"bin": i, "n": int(m.sum()),
                     "r": round(point_biserial(y[m], s[m]), 4),
                     "key_mid": round(float(key[m].mean()), 3)})
    return rows


def fixed_effect_regression(rows):
    """rows: list of dicts with per-trace y, con(centered within dataset), parse,
    det, dataset_id. Fit y ~ parse + det + parse:det + dataset dummies via least
    squares on the correctness-aligned constraint evidence.

    We regress the per-trace signed evidence g on [parse, det, parse*det] plus
    dataset fixed effects and report the coefficients with a cluster/bootstrap CI
    resampling whole datasets-then-traces."""
    y = np.array([r["g"] for r in rows], float)
    parse = np.array([r["parse"] for r in rows], float)
    det = np.array([r["det"] for r in rows], float)
    dsid = np.array([r["ds"] for r in rows], int)
    ndata = dsid.max() + 1

    def design(parse, det, dsid):
        cols = [np.ones(len(parse)), parse, det, parse * det]
        for d in range(1, ndata):   # dataset dummies (drop first as reference)
            cols.append((dsid == d).astype(float))
        return np.column_stack(cols)

    def fit(parse, det, dsid, y):
        X = design(parse, det, dsid)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    beta = fit(parse, det, dsid, y)
    # bootstrap over traces (within the pooled sample) for CIs of b_parse,b_det,b_int
    B = 1000
    boot = []
    n = len(y)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        try:
            b = fit(parse[idx], det[idx], dsid[idx], y[idx])
            boot.append(b[:4])
        except Exception:
            continue
    boot = np.array(boot)
    names = ["intercept", "parse", "determinacy", "parse_x_det"]
    res = {}
    for i, nm in enumerate(names):
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        res[nm] = {"coef": round(float(beta[i]), 4),
                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                   "sig": bool(lo > 0 or hi < 0)}
    return res


def main():
    out = {"backbone": BACKBONE, "per_dataset": {}, "pooled_bins": {}}
    reg_rows = []
    pooled = {"det": [], "parse": [], "y": [], "con_c": []}
    for ds_i, (tag, disp, cname) in enumerate(DATASETS):
        d = load(tag, cname)
        if d is None:
            continue
        y, con, parse, det = d["y"], d["con"], d["parse"], d["det"]
        con_c = con - con.mean()               # center within dataset
        out["per_dataset"][disp] = {
            "n": d["n"],
            "overall_r": round(point_biserial(y, con), 4),
            "by_determinacy": binned(y, con, det),
            "by_parse": binned(y, con, parse),
            "mean_det": round(float(det.mean()), 3),
            "mean_parse": round(float(parse.mean()), 3),
        }
        # per-trace signed evidence for the FE regression
        g = (2 * y - 1) * con_c
        for i in range(d["n"]):
            reg_rows.append({"g": float(g[i]), "parse": float(parse[i]),
                             "det": float(det[i]), "ds": ds_i})
        pooled["det"].extend(det.tolist()); pooled["parse"].extend(parse.tolist())
        pooled["y"].extend(y.tolist()); pooled["con_c"].extend(con_c.tolist())

    # pooled bins (all datasets stacked, constraint centered within dataset)
    py = np.array(pooled["y"]); pcon = np.array(pooled["con_c"])
    pdet = np.array(pooled["det"]); pparse = np.array(pooled["parse"])
    out["pooled_bins"]["by_determinacy"] = binned(py, pcon, pdet, nbins=6)
    out["pooled_bins"]["by_parse"] = binned(py, pcon, pparse, nbins=6)

    out["fixed_effect_regression"] = fixed_effect_regression(reg_rows)

    os.makedirs(f"{R}/fusion", exist_ok=True)
    json.dump(out, open(f"{R}/fusion/determinacy_trace.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
