#!/usr/bin/env python3
"""Rebuild latex/table/backbone.tex from Protocol-B emitter output for the three
appendix backbones (llama, gemma, qwen), preserving the grouped multirow layout
and per-backbone caption/label."""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKBONES = [("llama", "Llama-3.1-8B-Instruct", "tab:backbone-llama"),
             ("gemma", "Gemma-2-9B-it", "tab:backbone-gemma"),
             ("qwen", "Qwen3-8B", "tab:backbone-qwen")]
# row label -> group; group order and sizes
GROUPS = [("Unsup.", ["Random", "Perplexity", "Token Entropy", "MCP", "CCP"]),
          ("Sampl.", ["Semantic Entropy", "SelfCheckGPT", "P(True)"]),
          ("Neural", ["Factoscope", "UHead", "Neural-Seq", "MLP"]),
          ("Symb.", ["Constraint-Rule", "Constraint-Only", "Constraint"])]

HEADER = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{0.95}
\begin{tabular}{ll cc cc cc cc cc}
\toprule
& \multirow{2}{*}{\textbf{Method}}
& \multicolumn{2}{c}{\textbf{StepGame} (ID)}
& \multicolumn{2}{c}{\textbf{SpaRTQA} (OOD)}
& \multicolumn{2}{c}{\textbf{SpaRTUN} (OOD)}
& \multicolumn{2}{c}{\textbf{SpaceNLI} (OOD)}
& \multicolumn{2}{c}{\textbf{SpaRP} (OOD)} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}
& & AUC $\uparrow$ & BS $\downarrow$ & AUC $\uparrow$ & BS $\downarrow$ & AUC $\uparrow$ & BS $\downarrow$ & AUC $\uparrow$ & BS $\downarrow$ & AUC $\uparrow$ & BS $\downarrow$ \\
\midrule
"""


def emit_body(bb):
    out = subprocess.run([sys.executable, "scripts/emit_result_table_B.py", bb],
                         capture_output=True, text=True).stdout
    rows = {}
    sm_line = None
    for ln in out.splitlines():
        ln = ln.replace("\\deg{", "\\graycell{")
        if "\\textbf{\\m}" in ln:
            sm_line = ln
            continue
        if ln.startswith("& "):
            name = ln.split("&")[1].strip()
            rows[name] = ln
    return rows, sm_line


def build_table(bb, model, label):
    rows, sm_line = emit_body(bb)
    s = HEADER
    for gi, (gname, members) in enumerate(GROUPS):
        s += f"\\multirow{{{len(members)}}}{{*}}{{\\rotatebox{{90}}{{\\textit{{{gname}}}}}}}\n"
        for m in members:
            line = rows.get(m, f"& {m} " + "& -- " * 10 + "\\\\")
            s += line + "\n"
        s += "\\midrule\n"
    s += "\\rowcolor{gray!15}\n"
    s += (sm_line or "\\cellcolor{white} & \\textbf{\\m} " + "& -- " * 10 + "\\\\") + "\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    s += (f"\\caption{{Overall results with the frozen \\textbf{{{model}}} backbone, "
          f"under the same unified-calibrator, greedy-aligned protocol as "
          f"Table~\\ref{{tab:result}}. \\graycell{{Gray}} Brier cells are "
          f"non-discriminative (excluded from ranking).}}\n")
    s += f"\\label{{{label}}}\n\\end{{table*}}\n"
    return s


def main():
    blocks = [build_table(bb, model, label) for bb, model, label in BACKBONES]
    open("latex/table/backbone.tex", "w").write("\n\n".join(blocks))
    print("wrote latex/table/backbone.tex with", len(blocks), "tables")


if __name__ == "__main__":
    main()
