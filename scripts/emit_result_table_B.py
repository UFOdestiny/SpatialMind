#!/usr/bin/env python3
"""Emit the main result table body (Table 1, one backbone) under Protocol B:
EVERY DARC-eligible method uses the identical validation-fit calibrator
select_calibrator (macro-Brier-aware, CV-chosen from {affine,temperature,beta,
isotonic}). AUROC is calibration-invariant.

Fairness details (match benchmark_fair.py):
  * Sampling baselines re-decode at temperature and their own labels differ from
    the delivered greedy trace (~19% flip). We therefore score every method,
    including sampling baselines, against the SAME greedy-trace labels taken from
    the fusion report's sample_ids. Sampling baselines have only test scores on
    disk, so their Brier keeps the stored (greedy-aligned Platt) value; they are
    monotone-calibrated and near-chance in AUROC, so this cannot let them beat a
    high-AUROC method on macro-Brier.
  * Degenerate cells (single-class test pool, or a near-constant base-rate
    predictor whose macro-Brier collapses) are rendered gray via \deg{...}.

Usage: python scripts/emit_result_table_B.py mistral
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.fusion as F

BB = sys.argv[1] if len(sys.argv) > 1 else "mistral"
R = f"spatialmind/results/constraint_guided_{BB}"
SUB = f"constraint_guided_{BB}"
DS = [("id", "eval", "StepGame", "StepGame"), ("spartqa", "eval_ood/spartqa", "SpaRTQA", "spartqa"),
      ("SpaRTUN", "eval_ood/SpaRTUN", "SpaRTUN", "SpaRTUN"),
      ("SpaceNLI", "eval_ood/SpaceNLI", "SpaceNLI", "SpaceNLI"),
      ("SpaRP_PS3", "eval_ood/SpaRP_PS3", "SpaRP", "SpaRP_PS3")]
SIGKEY = {"Perplexity": "perplexity", "Token Entropy": "token_entropy", "MCP": "mcp",
          "CCP": "ccp", "Constraint-Rule": "constraint_rule", "Factoscope": "factoscope",
          "UHead": "uhead", "Neural-Seq": "spatialmind_neural", "MLP": "mlp",
          "Constraint-Only": "constraint_only", "Constraint": "spatialmind"}
SAMP = {"Semantic Entropy": "semantic_entropy", "SelfCheckGPT": "selfcheckgpt", "P(True)": "p_true"}
ROWS = [("Unsup.", "Random"), ("Unsup.", "Perplexity"), ("Unsup.", "Token Entropy"),
        ("Unsup.", "MCP"), ("Unsup.", "CCP"),
        ("Sampl.", "Semantic Entropy"), ("Sampl.", "SelfCheckGPT"), ("Sampl.", "P(True)"),
        ("Neural", "Factoscope"), ("Neural", "UHead"), ("Neural", "Neural-Seq"), ("Neural", "MLP"),
        ("Symb.", "Constraint-Rule"), ("Symb.", "Constraint-Only"), ("Symb.", "Constraint"),
        ("Fusion", "SpatialMind")]
CN = {"id": "StepGame", "spartqa": "spartqa", "SpaRTUN": "SpaRTUN",
      "SpaceNLI": "SpaceNLI", "SpaRP_PS3": "SpaRP_PS3"}


def clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def darc_brier(sv, yv, st, yt):
    cal, _ = F.select_calibrator(clip(sv), yv)
    return F._macro_brier(yt, F._apply_calib(cal, st))


def greedy_labels(tag):
    """Reference greedy labels from the fusion report (sample_id -> label)."""
    fp = f"{R}/fusion/{tag}/evaluation_report.json"
    if not os.path.exists(fp):
        return None
    pr = json.load(open(fp))["predictions"]
    return {x["sample_id"]: x["trace_label"] for x in pr}


def is_degenerate_brier(au, br, y):
    """A cell is 'degenerate' (gray) if the test pool is single-class or the
    predictor is near-chance (|AUROC-0.5|<0.03) so its Brier reflects a constant
    base-rate collapse rather than discrimination."""
    if y is not None and len(np.unique(y)) < 2:
        return True
    return au is not None and abs(au - 0.5) < 0.03


def main():
    cells = {lab: {} for _, lab in ROWS}
    deg = {lab: {} for _, lab in ROWS}
    for tag, erel, disp, _c in DS:
        ref = greedy_labels(tag)
        sv_sig = F.collect_signals(R, tag, CN[tag], "validation", SUB)
        st_sig = F.collect_signals(R, tag, CN[tag], "test", SUB)
        # DARC-eligible
        for lab, key in SIGKEY.items():
            if key in sv_sig and key in st_sig:
                idv, yv, sv = sv_sig[key]; idt, yt, st = st_sig[key]
                if len(np.unique(yv)) >= 2 and len(np.unique(yt)) >= 2:
                    au = F.auroc(yt, st); br = darc_brier(sv, yv, st, yt)
                    cells[lab][disp] = (au, br)
                    deg[lab][disp] = is_degenerate_brier(au, br, yt)
        # Random + sampling: greedy-aligned from stored reports
        for lab, key in list(SAMP.items()) + [("Random", "random")]:
            subdir = "baselines_sampling" if lab in SAMP else "baselines"
            p = f"{R}/{erel}/{subdir}/combined_evaluation.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if key not in d or "predictions" not in d[key]:
                continue
            pr = d[key]["predictions"]
            smap = {x["sample_id"]: x["trace_score"] for x in pr}
            if ref is not None:
                ids = sorted(set(ref) & set(smap))
                y = np.array([ref[i] for i in ids], float)
                s = np.array([smap[i] for i in ids], float)
            else:
                y = np.array([x["trace_label"] for x in pr], float)
                s = np.array([x["trace_score"] for x in pr], float)
            if len(np.unique(y)) < 2:
                continue
            au = F.auroc(y, s); br = F._macro_brier(y, s)
            cells[lab][disp] = (au, br)
            deg[lab][disp] = is_degenerate_brier(au, br, y)
        # SpatialMind
        fp = f"{R}/fusion/{tag}/evaluation_report.json"
        if os.path.exists(fp):
            pr = json.load(open(fp))["predictions"]
            y = np.array([x["trace_label"] for x in pr]); s = np.array([x["trace_score"] for x in pr])
            if len(np.unique(y)) >= 2:
                au = F.auroc(y, s); br = F._macro_brier(y, s)
                cells["SpatialMind"][disp] = (au, br)
                deg["SpatialMind"][disp] = is_degenerate_brier(au, br, y)

    DISPS = [d for _, _, d, _ in DS]
    rankable = [lab for _, lab in ROWS if lab != "Random"]

    # ranking excludes degenerate Brier cells and Random
    best, second = {}, {}
    for disp in DISPS:
        aus = sorted([(cells[l][disp][0], l) for l in rankable if disp in cells[l]], reverse=True)
        brs = sorted([(cells[l][disp][1], l) for l in rankable
                      if disp in cells[l] and not deg[l].get(disp)])
        best[disp] = {"au": aus[0][0] if aus else None, "br": brs[0][0] if brs else None}
        second[disp] = {"au": aus[1][0] if len(aus) > 1 else None,
                        "br": brs[1][0] if len(brs) > 1 else None}

    def fmt(v, b, s, gray=False):
        t = f"{v:.3f}"
        if gray:
            return f"\\deg{{{t}}}"
        if b is not None and abs(v - b) < 1e-9:
            return f"\\textbf{{{t}}}"
        if s is not None and abs(v - s) < 1e-9:
            return f"\\underline{{{t}}}"
        return t

    print(f"% Table 1 body, backbone={BB}, Protocol B; degenerate Brier in gray via \\deg")
    print("% add to preamble: \\newcommand{\\deg}[1]{\\textcolor{gray}{#1}}  (or \\textcolor{black!45})")
    for grp, lab in ROWS:
        parts = []
        for disp in DISPS:
            c = cells[lab].get(disp)
            if c is None:
                parts += ["--", "--"]; continue
            au, br = c
            g = deg[lab].get(disp, False)
            if lab == "Random":
                parts += [f"{au:.3f}", f"\\deg{{{br:.3f}}}"]
            else:
                parts.append(fmt(au, best[disp]["au"], second[disp]["au"]))
                parts.append(fmt(br, best[disp]["br"], second[disp]["br"], gray=g))
        prefix = "\\rowcolor{gray!15}\n\\cellcolor{white} & \\textbf{\\m}" if lab == "SpatialMind" else f"& {lab}"
        print(f"{prefix} & " + " & ".join(parts) + " \\\\")


if __name__ == "__main__":
    main()
