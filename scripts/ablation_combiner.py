#!/usr/bin/env python3
"""Combiner ablation for the DARC/fusion table (pure post-processing, no GPU).

Reproduces the paper's ablation rows per (backbone, dataset):
  * fixed 0.5 average of oriented signals
  * oracle per-dataset pure pick        (uses TEST AUROC -> not deployable, ref only)
  * stacking (scores only)              (gate="scores")
  * + symbolizability gate              (gate="symb")
  * + determinacy gate                  (gate="determinacy")
  * full \\m (validation-selected gate + L2 + Brier calibration)  == scripts/fusion.py

Every row is fit on VALIDATION only (except the oracle, explicitly labeled), then
applied to test. Reports AUROC and class-balanced (macro) Brier.

Reuses fusion.py internals so numbers match the headline combiner exactly.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fusion as F  # noqa: E402


def macro_brier(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[pos] - 1) ** 2) + 0.5 * np.mean(s[neg] ** 2))


def _prep(R, cache_root, model, tag, cname, sub):
    """Load + orient signals exactly as fusion.fuse_one does, up to the point of
    combiner selection. Returns oriented Sv/St, yv/yt, tfv/tft, idt or None."""
    if tag == "id":
        cache = f"{cache_root}/{sub}/StepGame/{model}"
    else:
        cache = f"{cache_root}/{sub}/{tag}/{model}"
    sv = F.collect_signals(R, tag, cname, "validation", sub)
    st = F.collect_signals(R, tag, cname, "test", sub)
    common = sorted(set(sv) & set(st))
    if len(common) < 2:
        return None
    sv = {k: sv[k] for k in common}; st = {k: st[k] for k in common}
    tfv_full = F.load_trace_features(cache, "validation")
    tft_full = F.load_trace_features(cache, "test")
    Sv, yv, idv, tfv = F.align(sv, tfv_full)
    St, yt, idt, tft = F.align(st, tft_full)
    if len(np.unique(yv)) < 2:
        return None
    val_au = {k: F.auroc(yv, v) for k, v in Sv.items()}
    oriented_v, oriented_t = {}, {}
    for k in sorted(Sv.keys()):
        a = val_au[k]
        if np.isnan(a) or abs(a - 0.5) < 0.03:
            continue
        sv_, st_ = Sv[k], St[k]
        if a < 0.5:
            sv_, st_ = 1 - sv_, 1 - st_
        oriented_v[k] = sv_; oriented_t[k] = st_
    if not oriented_v:
        return None
    return oriented_v, oriented_t, yv, yt, tfv, tft, idt, St, val_au


def _stack(Sv, St, yv, tfv, tft, gate, l2):
    Xv, _ = F.build_matrix(Sv, tfv, gate)
    Xt, _ = F.build_matrix(St, tft, gate)
    fit = F.fit_logreg(Xv, yv, l2=l2)
    return F.apply_logreg(fit, Xt)


def ablate_cell(R, cache_root, model, tag, cname, sub):
    p = _prep(R, cache_root, model, tag, cname, sub)
    if p is None:
        return None
    Sv, St, yv, yt, tfv, tft, idt, St_all, val_au = p
    rows = {}
    # fixed 0.5 average of oriented signals
    avg = np.mean(np.column_stack(list(St.values())), axis=1)
    rows["fixed_avg"] = avg
    # oracle per-dataset pure pick (TEST AUROC — reference only)
    best_k = max(St_all, key=lambda k: F.auroc(yt, St_all[k]))
    ob = St_all[best_k]
    if F.auroc(yt, ob) < 0.5:
        ob = 1 - ob
    rows["oracle_pick"] = ob
    # stacking variants (fixed reference L2=5)
    rows["scores_only"] = _stack(Sv, St, yv, tfv, tft, "scores", 5.0)
    rows["symb_gate"] = _stack(Sv, St, yv, tfv, tft, "symb", 5.0)
    rows["determinacy_gate"] = _stack(Sv, St, yv, tfv, tft, "determinacy", 5.0)
    # full validation-selected combiner == fusion.fuse_one
    full = F.fuse_one(R, cache_root, model, tag, cname, sub)
    rows["full"] = full["fused"] if full else None
    out = {}
    for name, s in rows.items():
        if s is None:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (F.auroc(yt, s), macro_brier(yt, s))
    return out, yt


DATASETS = [("id", "StepGame"), ("spartqa", "spartqa"), ("babi", "babi"),
            ("SpaRTUN", "SpaRTUN"), ("SpaceNLI", "SpaceNLI")]
ROW_ORDER = ["fixed_avg", "oracle_pick", "scores_only", "symb_gate",
             "determinacy_gate", "full"]
ROW_LABEL = {"fixed_avg": "Fixed 0.5 average",
             "oracle_pick": "Oracle pure pick (test)",
             "scores_only": "Stacking (scores only)",
             "symb_gate": "  + symbolizability gate",
             "determinacy_gate": "  + determinacy gate",
             "full": "SpatialMind (full)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--cache_root", default="spatialmind/cache/cached_features")
    ap.add_argument("--cache_subdir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cells = {}
    for tag, cname in DATASETS:
        try:
            r = ablate_cell(args.results_root, args.cache_root, args.model,
                            tag, cname, args.cache_subdir)
        except Exception as e:
            print(f"[warn] {cname}: {e}"); r = None
        cells[cname] = r[0] if r else None
    # print AUROC/Brier table
    hdr = f"{'variant':26s}" + "".join(f"{d[1]:>16s}" for d in DATASETS)
    print(hdr)
    for row in ROW_ORDER:
        line = f"{ROW_LABEL[row]:26s}"
        for _, cname in DATASETS:
            c = cells.get(cname)
            if c and row in c:
                au, br = c[row]
                line += f"{au:>7.3f}/{br:<8.3f}"
            else:
                line += f"{'-':>16s}"
        print(line)


if __name__ == "__main__":
    main()
