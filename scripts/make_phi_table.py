#!/usr/bin/env python3
"""Emit the LaTeX body rows for the Phi-4 backbone table (tab:backbone-phi4),
with best (\\textbf) / second-best (\\underline) marking per column, excluding
Random. Fair greedy-label protocol, AUROC / class-balanced macro-Brier.
Run once the phi v11 fusion + eval + sampling are on disk.
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

R = "spatialmind/results/constraint_guided_v11_phi"
DS = [("id", "eval", "StepGame"), ("spartqa", "eval_ood/spartqa", "SpaRTQA"),
      ("SpaRTUN", "eval_ood/SpaRTUN", "SpaRTUN"), ("SpaceNLI", "eval_ood/SpaceNLI", "SpaceNLI"),
      ("SpaRP_PS3", "eval_ood/SpaRP_PS3", "SpaRP")]
ROWS = [("Unsup.", "Random", ("base", "random")), ("Unsup.", "Perplexity", ("base", "perplexity")),
        ("Unsup.", "Token Entropy", ("base", "token_entropy")), ("Unsup.", "MCP", ("base", "mcp")),
        ("Unsup.", "CCP", ("base", "ccp")),
        ("Sampl.", "Semantic Entropy", ("samp", "semantic_entropy")),
        ("Sampl.", "SelfCheckGPT", ("samp", "selfcheckgpt")), ("Sampl.", "P(True)", ("samp", "p_true")),
        ("Neural", "Factoscope", ("head", "factoscope")), ("Neural", "UHead", ("head", "uhead")),
        ("Neural", "Neural-Seq", ("head", "spatialmind_neural")), ("Neural", "MLP", ("head", "mlp")),
        ("Symb.", "Constraint-Rule", ("base", "constraint_rule")),
        ("Symb.", "Constraint-Only", ("head", "constraint_only")),
        ("Symb.", "Constraint", ("head", "spatialmind")), ("Fusion", "\\m", ("fusion", None))]
GROUPS = [("Unsup.", 5), ("Sampl.", 3), ("Neural", 4), ("Symb.", 3)]


def sm(pr): return {x["sample_id"]: x["trace_score"] for x in pr}
def lm(pr): return {x["sample_id"]: x["trace_label"] for x in pr}


def loadhead(edir, h):
    p = f"{edir}/{h}/evaluation_report.json"
    return sm(json.load(open(p))["predictions"]) if os.path.exists(p) else None


def loadcomb(edir, sub, k):
    p = f"{edir}/{sub}/combined_evaluation.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return sm(d[k]["predictions"]) if k in d and "predictions" in d[k] else None


def loadfus(tag):
    p = f"{R}/fusion/{tag}/evaluation_report.json"
    return sm(json.load(open(p))["predictions"]) if os.path.exists(p) else None


def reflab(tag):
    p = f"{R}/fusion/{tag}/evaluation_report.json"
    return lm(json.load(open(p))["predictions"]) if os.path.exists(p) else None


def macro(y, s):
    s = np.clip(s, 1e-6, 1 - 1e-6); p = y == 1; n = y == 0
    if p.sum() == 0 or n.sum() == 0:
        return float(np.mean((s - y) ** 2))
    return float(0.5 * np.mean((s[p] - 1) ** 2) + 0.5 * np.mean(s[n] ** 2))


def get(edir, spec, tag):
    k, key = spec
    if k == "head": return loadhead(edir, key)
    if k == "base": return loadcomb(edir, "baselines", key)
    if k == "samp": return loadcomb(edir, "baselines_sampling", key)
    if k == "fusion": return loadfus(tag)


MIN_POS = 20  # minority-class floor below which AUROC is not reported (degenerate)


def met(scm, ref):
    if scm is None or ref is None:
        return None
    ids = sorted(set(scm) & set(ref))
    if not ids:
        return None
    y = np.array([ref[i] for i in ids], float); s = np.array([scm[i] for i in ids], float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    # AUROC undefined or statistically meaningless when a class is tiny
    au = roc_auc_score(y, s) if (npos >= MIN_POS and nneg >= MIN_POS) else float("nan")
    br = macro(y, s) if (npos > 0 and nneg > 0) else float("nan")
    return au, br


def main():
    if not os.path.isdir(f"{R}/fusion"):
        print("Phi fusion not ready yet."); return
    refs = {d: reflab(ft) for ft, er, d in DS}
    D = {}
    for _, lab, spec in ROWS:
        D[lab] = {d: met(get(f"{R}/{er}", spec, ft), refs[d]) for ft, er, d in DS}
    dcols = [d for _, _, d in DS]
    rk = {}
    for d in dcols:
        aus = sorted([(D[l][d][0], l) for _, l, _ in ROWS if l != "Random" and D[l][d] and not np.isnan(D[l][d][0])], reverse=True)
        brs = sorted([(D[l][d][1], l) for _, l, _ in ROWS if l != "Random" and D[l][d] and not np.isnan(D[l][d][1])])
        rk[d] = {"a1": aus[0][1] if aus else None, "a2": aus[1][1] if len(aus) > 1 else None,
                 "b1": brs[0][1] if brs else None, "b2": brs[1][1] if len(brs) > 1 else None}

    def fmt(l, d):
        m = D[l][d]
        if m is None:
            return "--"
        a, b = m
        if np.isnan(a):
            # AUROC undefined (single-class pool): report only balanced Brier
            bw = "\\textbf{%.3f}" % b if rk[d]["b1"] == l else ("\\underline{%.3f}" % b if rk[d]["b2"] == l else "%.3f" % b)
            return f"n/a\\ /\\ {bw}" if not np.isnan(b) else "n/a"
        aw = "\\textbf{%.3f}" % a if rk[d]["a1"] == l else ("\\underline{%.3f}" % a if rk[d]["a2"] == l else "%.3f" % a)
        bw = "\\textbf{%.3f}" % b if rk[d]["b1"] == l else ("\\underline{%.3f}" % b if rk[d]["b2"] == l else "%.3f" % b)
        return f"{aw}\\ /\\ {bw}"

    # print full table body grouped
    idx = 0
    for gname, cnt in GROUPS:
        block = [l for g, l, _ in ROWS if g == gname]
        print(f"\\multirow{{{cnt}}}{{*}}{{\\rotatebox{{90}}{{\\textit{{{gname}}}}}}}")
        for l in block:
            disp = l
            print(f"& {disp} & " + " & ".join(fmt(l, d) for d in dcols) + " \\\\")
        print("\\midrule")
    # fusion row
    print("\\rowcolor{gray!15}")
    print("\\cellcolor{white} & \\textbf{\\m} & " + " & ".join(fmt("\\m", d) for d in dcols) + " \\\\")
    # means
    aus = [D["\\m"][d][0] for d in dcols if D["\\m"][d]]
    print(f"% Phi \\m mean AUROC = {np.mean(aus):.3f}")


if __name__ == "__main__":
    main()
