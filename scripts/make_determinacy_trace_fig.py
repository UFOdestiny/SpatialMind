#!/usr/bin/env python3
"""Regenerate fig/determinacy_trace.pdf: the trace-level determinacy analysis
(P2). Left: pooled within-dataset constraint separation (point-biserial r
between constraint score and correctness) rises with status determinacy but is
flat across parse coverage. This is the robust replacement for the n=5 scatter.

Reads the determinacy_trace.json produced by paper_determinacy_trace.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "spatialmind/results/constraint_guided_v11_llama/fusion/determinacy_trace.json"
OUT = "latex/fig"
INK = "#1b2a4a"
DET = "#2f6db3"
PARSE = "#c0483a"


def main():
    d = json.load(open(SRC))
    det = [b for b in d["pooled_bins"]["by_determinacy"] if b["r"] is not None]
    par = [b for b in d["pooled_bins"]["by_parse"] if b["r"] is not None]
    dx = [b["key_mid"] for b in det]; dy = [b["r"] for b in det]
    px = [b["key_mid"] for b in par]; py = [b["r"] for b in par]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(dx, dy, "-o", color=DET, lw=1.8, ms=5, label="Status determinacy")
    ax.plot(px, py, "--s", color=PARSE, lw=1.6, ms=4, label="Parse coverage")
    ax.set_xlabel("Applicability bin (trace-level)", fontsize=9.5, color=INK)
    ax.set_ylabel("Constraint separation ($r_{pb}$)", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa4b2")
    ax.grid(True, alpha=0.18, lw=0.6)
    fig.tight_layout(pad=0.4)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/determinacy_trace.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/determinacy_trace.pdf")
    fe = d["fixed_effect_regression"]
    print("FE determinacy coef:", fe["determinacy"])
    print("FE parse coef:", fe["parse"])


if __name__ == "__main__":
    main()
