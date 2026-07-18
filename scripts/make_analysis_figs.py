#!/usr/bin/env python3
"""Regenerate fig/parse.pdf and fig/determinacy.pdf: verifier applicability
scatter plots over the benchmarks. x = parse coverage / status determinacy,
y = constraint-scorer AUROC. Reads the Llama v11 paper_analysis.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "spatialmind/results/constraint_guided_v11_llama/fusion/paper_analysis.json"
OUT = "latex/fig"
DISP = {"StepGame": "StepGame", "spartqa": "SpaRTQA", "babi": "bAbI",
        "SpaRTUN": "SpaRTUN", "SpaceNLI": "SpaceNLI",
        "SpaRP_PS1": "SpaRP-PS1", "SpaRP_PS3": "SpaRP"}

INK = "#1b2a4a"
POINT = "#2f6db3"
TREND = "#c0483a"


HEADLINE = ["StepGame", "spartqa", "SpaRTUN", "SpaceNLI", "SpaRP_PS3"]


def load():
    d = json.load(open(SRC))["datasets"]
    names = [k for k in HEADLINE if k in d]
    parse = np.array([d[k]["parse_rate"] for k in names])
    det = np.array([1 - d[k]["unknown_rate"] for k in names])
    con = np.array([d[k]["con"] for k in names])
    labels = [DISP.get(k, k) for k in names]
    return parse, det, con, labels


def panel(x, y, labels, xlabel, fname, show_trend):
    r = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(x, y, s=70, c=POINT, edgecolors="white", linewidths=1.0, zorder=3)
    if show_trend:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.03, x.max() + 0.03, 50)
        ax.plot(xs, a + b * xs, "--", color=TREND, lw=1.6, zorder=2)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.0, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel("Constraint AUROC", fontsize=10, color=INK)
    ax.text(0.04, 0.93, f"$r={r:.2f}$", transform=ax.transAxes,
            fontsize=11, color=INK, fontweight="bold", va="top")
    ax.set_ylim(0.25, 0.95)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa4b2")
    ax.grid(True, alpha=0.18, lw=0.6)
    fig.tight_layout(pad=0.4)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{fname}  r={r:.3f}")


def main():
    parse, det, con, labels = load()
    panel(parse, con, labels, "Parse coverage", "parse.pdf", show_trend=False)
    panel(det, con, labels, "Status determinacy", "determinacy.pdf", show_trend=True)


if __name__ == "__main__":
    main()
