#!/usr/bin/env python3
"""Emit the fair-ablation LaTeX table body (Mistral primary) from fair_ablation.json.
All rows share the SAME calibrator; the last two columns give the paired-bootstrap
delta-AUROC (SpatialMind - scores-only) with significance marker."""
import json, sys

BB = sys.argv[1] if len(sys.argv) > 1 else "mistral"
d = json.load(open(f"spatialmind/results/constraint_guided_{BB}/fusion/fair_ablation.json"))
DS = ["StepGame", "SpaRTQA", "SpaRTUN", "SpaceNLI", "SpaRP"]
ROWS = [("scores", "Stacking (scores only)"),
        ("symb", "\\quad + symbolizability gate"),
        ("determinacy", "\\quad + determinacy gate"),
        ("spatialmind", "\\m")]


def mark(vals, v, lower=False):
    xs = sorted(set(vals), reverse=not lower)
    best = xs[0]; second = xs[1] if len(xs) > 1 else None
    t = f"{v:.3f}"
    if v == best:
        return f"\\textbf{{{t}}}"
    if second is not None and v == second:
        return f"\\underline{{{t}}}"
    return t


# collect per-column values for ranking
au = {ds: [] for ds in DS}
br = {ds: [] for ds in DS}
for key, _ in ROWS:
    for ds in DS:
        c = d["datasets"].get(ds, {}).get(key)
        if c:
            au[ds].append(c["auroc"]); br[ds].append(c["macro_brier"])

print(f"% fair ablation, backbone={BB} (all rows share the selected calibrator)")
for key, lab in ROWS:
    cells = []
    for ds in DS:
        c = d["datasets"].get(ds, {}).get(key)
        if not c:
            cells += ["--", "--"]; continue
        cells.append(mark(au[ds], c["auroc"]))
        cells.append(mark(br[ds], c["macro_brier"], lower=True))
    prefix = "\\rowcolor{gray!15}\n\\textbf{\\m}" if key == "spatialmind" else lab
    print(f"{prefix} & " + " & ".join(cells) + " \\\\")

# significance line: SpatialMind vs scores-only
print("\n% significance (SpatialMind vs scores-only stacking), paired bootstrap")
sig_cells = []
for ds in DS:
    sv = d["datasets"].get(ds, {}).get("sm_vs_scores")
    if not sv:
        sig_cells.append(f"{ds}: n/a"); continue
    star = "$^{*}$" if sv["p_two_sided"] < 0.05 else ""
    sig_cells.append(f"{ds}: $\\Delta{sv['delta_auroc']:+.3f}${star} (p={sv['p_two_sided']:.3f})")
print("% " + "; ".join(sig_cells))
