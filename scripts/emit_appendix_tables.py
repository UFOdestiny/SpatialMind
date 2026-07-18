#!/usr/bin/env python3
"""Emit appendix LaTeX tables from the analysis JSONs:
  * significance (SM vs scores-only) across backbones
  * auditor faithfulness (per dataset, mean over backbones + Llama detail)
  * adaptation ladder (Mistral) + validation-size curve
  * comprehensive metrics (Mistral)
Writes each to latex/table/<name>.tex."""
import json
import numpy as np

BBS = [("mistral", "Mistral-7B"), ("llama", "Llama-3.1-8B"), ("gemma", "Gemma-2-9B"),
       ("phi", "Phi-4"), ("qwen", "Qwen3-8B")]
DS = ["StepGame", "SpaRTQA", "SpaRTUN", "SpaceNLI", "SpaRP"]

RB_OPEN = r"\resizebox{\columnwidth}{!}{%"
RB_CLOSE = r"}"


def wrap_resizebox(lines):
    """Insert \\resizebox around the tabular env in a list of latex lines."""
    out = []
    for ln in lines:
        if ln.startswith(r"\begin{tabular}"):
            out.append(RB_OPEN); out.append(ln)
        elif ln.startswith(r"\end{tabular}"):
            out.append(ln); out.append(RB_CLOSE)
        else:
            out.append(ln)
    return out


def load(bb, name):
    p = f"spatialmind/results/constraint_guided_v11_{bb}/fusion/{name}.json"
    return json.load(open(p))


# ---------- 1. significance table ----------
def significance():
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l ccccc}", r"\toprule",
             r"\textbf{Backbone} & StepGame & SpaRTQA & SpaRTUN & SpaceNLI & SpaRP \\", r"\midrule"]
    nsig = ntot = 0
    for bb, disp in BBS:
        d = load(bb, "fair_ablation")
        cells = []
        for ds in DS:
            v = d["datasets"].get(ds, {})
            sv = v.get("sm_vs_scores")
            if sv:
                ntot += 1
                sig = sv["p_two_sided"] < 0.05 and sv["delta_auroc"] > 0
                nsig += sig
                star = "$^{*}$" if sig else ""
                cells.append(f"${sv['delta_auroc']:+.3f}${star}")
            elif "error" in v:
                cells.append("--")
            else:
                cells.append("n/a")
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Paired-bootstrap AUROC gain of \m over scores-only stacking, "
              r"per backbone and dataset ($2{,}000$ resamples, two-sided; $^{*}$: $p<0.05$). "
              f"The gain is positive and significant on {nsig} of {ntot} non-degenerate cells. "
              r"``--'' marks degenerate single-class cells (Phi-4 on SpaRP; Qwen on SpaRTQA has no "
              r"informative base signal to compose).}",
              r"\label{tab:significance}", r"\end{table}"]
    open("latex/table/significance.tex", "w").write("\n".join(wrap_resizebox(lines)) + "\n")
    print(f"significance: {nsig}/{ntot} significant")


# ---------- 2. auditor faithfulness ----------
def auditor():
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l ccccc}", r"\toprule",
             r"\textbf{Dataset} & Parse & Det.\ cov. & Status acc. & Ent.\ prec. & Con.\ prec. \\",
             r"\midrule"]
    # average over backbones per dataset
    agg = {ds: {"parse_rate": [], "determinate_coverage": [], "status_accuracy_determinate": [],
                "entailment_precision": [], "contradiction_precision": []} for ds in DS}
    for bb, _ in BBS:
        try:
            d = load(bb, "auditor_accuracy")
        except FileNotFoundError:
            continue
        for ds in DS:
            r = d["datasets"].get(ds, {})
            for k in agg[ds]:
                v = r.get(k)
                if isinstance(v, float):
                    agg[ds][k].append(v)
    for ds in DS:
        c = agg[ds]
        row = [f"{np.mean(c[k]):.2f}" if c[k] else "--" for k in
               ("parse_rate", "determinate_coverage", "status_accuracy_determinate",
                "entailment_precision", "contradiction_precision")]
        lines.append(f"{ds} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Layout Auditor faithfulness, averaged over the four backbones. "
              r"Auditor verdicts are scored against the reference claim labels. "
              r"``Parse'' is the fraction of labeled claims parsed; ``Det.\ cov.'' the fraction of "
              r"parsed claims that receive a determinate (entailed/contradicted) verdict; "
              r"``Status acc.'' the accuracy of those determinate verdicts; ``Ent.\ prec.'' and "
              r"``Con.\ prec.'' the precision of the entailed and contradicted verdicts. "
              r"Entailed verdicts are highly precise ($0.74$--$0.91$); contradicted verdicts are "
              r"flagged against the (possibly wrong) generated prefix and so are less precise by "
              r"design, which is exactly what the first-conflict feature is meant to localize.}",
              r"\label{tab:auditor}", r"\end{table}"]
    open("latex/table/auditor.tex", "w").write("\n".join(wrap_resizebox(lines)) + "\n")
    print("auditor table written")


# ---------- 3. adaptation ladder + val-size ----------
def ladder():
    d = load("mistral", "adaptation_ladder")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l ccccc}", r"\toprule",
             r"\textbf{Rung (target-val use)} & StepGame & SpaRTQA & SpaRTUN & SpaceNLI & SpaRP \\",
             r"\midrule"]
    rungs = [("L1_zeroshot", "L1 Source-only (zero-shot)"),
             ("L3_target_stacking", "L3 Target stacking"),
             ("L4_determinacy", "L4 \\quad + determinacy"),
             ("L5_spatialmind", "L5 \\m (selected)")]
    for key, lab in rungs:
        cells = []
        for ds in DS:
            v = d["datasets"].get(ds, {}).get(key, {})
            a = v.get("auroc") if isinstance(v, dict) else None
            cells.append(f"{a:.3f}" if a is not None else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Adaptation ladder on Mistral-7B (test AUROC). L2 (target calibration only) "
              r"leaves AUROC unchanged from L1 by construction and is omitted. Zero-shot transfer of the "
              r"source-trained constraint scorer collapses on the hard targets (SpaRTQA, SpaRTUN); "
              r"target stacking recovers a reliable ranking and the determinacy interactions add a further "
              r"increment on all but the in-distribution benchmark.}",
              r"\label{tab:ladder}", r"\end{table}"]
    open("latex/table/ladder.tex", "w").write("\n".join(wrap_resizebox(lines)) + "\n")

    # val-size curve
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l cccccc}", r"\toprule",
             r"\textbf{Dataset} & 16 & 32 & 64 & 128 & 256 & full \\", r"\midrule"]
    for ds in DS:
        v = d["datasets"].get(ds, {})
        curve = v.get("val_size_curve", []) if isinstance(v, dict) else []
        by_m = {c["m"]: c for c in curve}
        cells = []
        ms = sorted(by_m)
        full_m = ms[-1] if ms else None
        for m in [16, 32, 64, 128, 256]:
            c = by_m.get(m)
            cells.append(f"{c['mean']:.3f}" if c else "--")
        cf = by_m.get(full_m) if full_m and full_m not in (16, 32, 64, 128, 256) else None
        cells.append(f"{cf['mean']:.3f}" if cf else "--")
        lines.append(f"{ds} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Validation-size sensitivity on Mistral-7B (mean test AUROC of the determinacy "
              r"design over $5$ random target-validation subsamples of each size; ``full'' is the complete "
              r"validation split). Performance is stable from roughly $128$ target labels, so \m does not "
              r"require a large labeled target set to select and compose scorers.}",
              r"\label{tab:valsize}", r"\end{table}"]
    open("latex/table/valsize.tex", "w").write("\n".join(wrap_resizebox(lines)) + "\n")
    print("ladder + valsize written")


# ---------- 4. comprehensive metrics ----------
def metrics():
    d = load("mistral", "extra_metrics")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ll cccccc}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Method} & AUROC$\uparrow$ & bBS$\downarrow$ & Brier$\downarrow$ & ECE$\downarrow$ & NLL$\downarrow$ & AURC$\downarrow$ \\",
             r"\midrule"]
    for ds in DS:
        rec = d["datasets"].get(ds, {})
        for key, lab in [("spatialmind", r"\m"),
                         ("best_baseline", rec.get("best_baseline", {}).get("name", "base"))]:
            r = rec.get(key)
            if not r:
                continue
            name = lab if key == "spatialmind" else lab.replace("_", r"\_")
            lines.append(f"{ds if key=='spatialmind' else ''} & {name} & "
                         f"{r['auroc']:.3f} & {r['macro_brier']:.3f} & {r['brier']:.3f} & "
                         f"{r['ece']:.3f} & {r['nll']:.3f} & {r['aurc']:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              r"\caption{Comprehensive metrics on Mistral-7B: \m versus the strongest external baseline "
              r"(highest test AUROC) per dataset. bBS is the class-balanced Brier Score. \m wins on AUROC "
              r"and bBS everywhere. On plain Brier, ECE, NLL, and AURC a degenerate base-rate predictor can "
              r"score better while carrying no discriminative power (AUROC near $0.5$), which is precisely why "
              r"the class-balanced Brier Score is the headline calibration metric (Appendix~\ref{app_metrics}).}",
              r"\label{tab:extrametrics}", r"\end{table}"]
    open("latex/table/extrametrics.tex", "w").write("\n".join(wrap_resizebox(lines)) + "\n")
    print("metrics table written")


if __name__ == "__main__":
    significance()
    auditor()
    ladder()
    metrics()
