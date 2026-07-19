#!/usr/bin/env python3
"""Emit the main result table body (Table 1, one backbone) under Protocol A.
Every method (baselines/heads via stored Platt-calibrated reports, SpatialMind
via the Platt-recalibrated fusion report) is read from disk. AUROC + class-
balanced Brier, best bold / second underline per column (Random excluded from
ranking). Usage: python scripts/emit_result_table.py mistral"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F

BB = sys.argv[1] if len(sys.argv) > 1 else "mistral"
R = f"spatialmind/results/constraint_guided_{BB}"
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
        ("Symb.", "Constraint", ("head", "spatialmind")),
        ("Fusion", "SpatialMind", ("fusion", None))]


def macro_brier(y, s):
    return F._macro_brier(np.asarray(y, float), np.asarray(s, float))


def load(spec, tag, erel):
    kind, name = spec
    if kind == "fusion":
        p = f"{R}/fusion/{tag}/evaluation_report.json"
        if not os.path.exists(p):
            return None
        pr = json.load(open(p))["predictions"]
    elif kind == "head":
        p = f"{R}/{erel}/{name}/evaluation_report.json"
        if not os.path.exists(p):
            return None
        pr = json.load(open(p)).get("predictions")
    else:  # base / samp -> baselines combined
        sub = "baselines_sampling" if kind == "samp" else "baselines"
        p = f"{R}/{erel}/{sub}/combined_evaluation.json"
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        if name not in d or "predictions" not in d[name]:
            return None
        pr = d[name]["predictions"]
    if not pr:
        return None
    y = np.array([x["trace_label"] for x in pr]); s = np.array([x["trace_score"] for x in pr])
    if len(np.unique(y)) < 2:
        return None
    return F.auroc(y, s), macro_brier(y, s)


# gather
cells = {}
for grp, lab, spec in ROWS:
    cells[lab] = {}
    for tag, erel, disp in DS:
        cells[lab][disp] = load(spec, tag, erel)

# rank per column (exclude Random)
DISPS = [d for _, _, d in DS]
rankable = [lab for _, lab, _ in ROWS if lab != "Random"]
best = {}; second = {}
for disp in DISPS:
    aus = sorted([(cells[l][disp][0], l) for l in rankable if cells[l][disp]], reverse=True)
    brs = sorted([(cells[l][disp][1], l) for l in rankable if cells[l][disp]])
    best[disp] = {"au": aus[0][0] if aus else None, "br": brs[0][0] if brs else None}
    second[disp] = {"au": aus[1][0] if len(aus) > 1 else None, "br": brs[1][0] if len(brs) > 1 else None}


def fmt(v, b, s):
    t = f"{v:.3f}"
    if b is not None and abs(v - b) < 1e-9:
        return f"\\textbf{{{t}}}"
    if s is not None and abs(v - s) < 1e-9:
        return f"\\underline{{{t}}}"
    return t


print(f"% Table 1 body, backbone={BB}, Protocol A (all Platt)")
for grp, lab, spec in ROWS:
    parts = []
    for disp in DISPS:
        c = cells[lab][disp]
        if c is None:
            parts += ["--", "--"]; continue
        au, br = c
        if lab == "Random":
            parts += [f"{au:.3f}", f"{br:.3f}"]
        else:
            parts.append(fmt(au, best[disp]["au"], second[disp]["au"]))
            parts.append(fmt(br, best[disp]["br"], second[disp]["br"]))
    prefix = "\\rowcolor{gray!15}\n\\cellcolor{white} & \\textbf{\\m}" if lab == "SpatialMind" else f"& {lab}"
    print(f"{prefix} & " + " & ".join(parts) + " \\\\")
