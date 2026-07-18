---
name: review-response-v11
description: 2026-07-18 revision responding to latex/review.md (5/10 Weak Reject); new experiments + edits
metadata:
  type: project
---

Responded to the AAAI reviewer in `latex/review.md` (5/10, 8 major concerns). All new
experiments are pure post-processing over cached v11 predictions (no GPU needed despite the
free B200). New analysis scripts (each writes JSON under `results/<ns>/fusion/`):

- `scripts/paper_fair_ablation.py` — P1: every stacked variant shares the SAME calibrator;
  paired-bootstrap CIs + p. Result: SpatialMind > scores-only stacking significant on 16/23
  cells; determinacy gate alone 13/23. Killed the calibration-confounded Brier gap (old
  0.348→0.164 was a calib artifact; fair is 0.180→0.164).
- `scripts/paper_determinacy_trace.py` (+`make_determinacy_trace_fig.py`) — P2: replaces
  n=5 scatter. Dataset-fixed-effect regression: β_det=+0.018 CI[0.006,0.029] (sig),
  β_parse=−0.018 (not positive). Pooled bins: constraint separation rises with determinacy,
  flat/non-monotone with parse. New `fig/determinacy_trace.pdf` replaces parse.pdf+determinacy.pdf.
- `scripts/paper_auditor_accuracy.py` — P5/RQ5: auditor verdicts vs reference labels.
  Entailment precision 0.74–0.91 across backbones; contradiction precision lower BY DESIGN
  (flagged against possibly-wrong generated prefix = first-conflict localization).
- `scripts/paper_adaptation_ladder.py` — P3/P6/RQ6: L1 zero-shot collapses on hard OOD
  (SpaRTQA 0.31, SpaRP-gemma 0.199); target stacking recovers; determinacy adds increment.
  Val-size curve: stable from ~128 target labels.
- `scripts/paper_extra_metrics.py` — P7: adds Brier/ECE/NLL/AURC. Shows degenerate base-rate
  predictors win those but have AUROC≈0.5 → justifies class-balanced Brier as headline.
- `scripts/emit_appendix_tables.py`, `emit_fair_ablation_table.py` — regenerate LaTeX tables.

KEY FACTS discovered:
- Trace correctness labels for ALL 5 headline datasets are DETERMINISTIC gold-match
  (`data/*.py check_correctness`), NOT the Qwen judge. Judge only labels intermediate
  reasoning claims for training. This resolves reviewer concern #4 (was a writing ambiguity).
- Random-baseline Brier=0.43 (not 0.25) because collapsed baselines predict the base rate p,
  so class-balanced Brier = 0.5(1-p)²+0.5p². Verified empirically; fixed Table 1 caption.
- d=1 boundary bug (concern #6): zero-parse traces got d=1 (21% of SpaceNLI). Fixed def to
  d=0 when π=0 + added 1[π=0] indicator to the profile. δ=πd was already invariant.

Model-name gotcha: gemma cache dir is `gemma-2-9b-it` (lowercase), not `Gemma-2-9B-it`.
Phi-4 has degenerate single-class cells (SpaRP) — bootstrap is slow there; cap NBOOT.

Paper: main text still 8 pages (refs page 9, same budget as pre-revision). Compiles clean,
no undefined refs, 0 overfull boxes. Softened "Determinacy Explains"→"Characterizes".
Added RQ5/RQ6/RQ7 (appendix) + Evidence-Combination related-work subsection. See
[[paper-writing-v11-session]].

## 2026-07-18 calibration-protocol overhaul (concern #7, decisive)
Reviewer #7 (calibration fairness) turned out to be the real threat. Findings:
- Both StandardCalibrator (Platt) and fusion's select_calibrator (macro-Brier affine,
  the "DARC calibrator") are MONOTONE → AUROC is protocol-invariant; only Brier moves.
- The paper's OLD Table-1 Brier gap (0.348→0.164) was a CALIBRATION ARTIFACT: SpatialMind
  used select_calibrator, baselines used Platt. Unfair.
- Decision: **Protocol B = the identical select_calibrator applied to EVERY method.**
  fusion.py now takes FUSION_CALIB env (default platt; set darc for Protocol B); reran all
  5 backbones. Baselines/heads recalibrated via select_calibrator in the emitters.
- CRITICAL fairness detail: sampling baselines (SE/SelfCheck/P(True)) MUST be scored against
  the SAME greedy-trace labels (benchmark_fair.py), not their own re-decoded labels (they
  flip ~19%). Their own-label Brier looked deceptively low (that's what made SM "lose 3 cols"
  in an early buggy check). Greedy-aligned, they drop to AUROC~0.5.
- RESULT under Protocol B + greedy-aligned: SpatialMind wins AUROC AND macro-Brier on ALL 5
  cells for Mistral(5/5) & Gemma(6/6); Llama 5/7, Phi 3/5, Qwen 1/5 (degenerate/weak cells).
- HEADLINE (final, conservative): Protocol A/B, DEF-B (per-dataset best of ALL baselines),
  4 backbones = **AUROC +8%, macro-Brier +7%**. This REPLACES the old +17%/+25% (which used
  the calibration-asymmetric, single-fixed-baseline definition). Abstract/intro now say ~8%/~7%
  and reduced numeric density throughout per user request.
- Tables: result.tex (Mistral) + backbone.tex (llama/gemma/qwen) rebuilt via
  scripts/emit_result_table_B.py + build_backbone_tex.py. Degenerate Brier cells (AUROC≈0.5 or
  single-class) rendered gray via \graycell (defined in main.tex; NOT \deg — clashes with LaTeX).
- Figures: analysis/v3.ipynb (fresh, reuses v2 config/style: serif+usetex, fig (5,4)) exports
  fig/determinacy_trace.pdf + fig/risk_coverage.pdf; placed side-by-side at 0.47\linewidth in
  RQ3. Risk-coverage MUST use ranking-based selective risk (rank by score, risk=1-precision of
  accepted), NOT metrics.py compute_aurc (0.5-threshold, calibration-dependent, made SM look bad).
- PAGE BUDGET: now exactly 7 main-text pages, refs start page 8, page 7 full (~4940 chars).
  Achieved by compressing methodology (bank paragraphs→1), RQ2/RQ3/RQ4 prose, preliminary,
  intro P5, conclusion. Compiles clean, 0 overfull, no undefined refs.
- Helper scripts: paper_calib_protocol.py (SM raw-score reconstruction, cached to
  /tmp/sm_raw_<bb>.json), paper_metric_protocol.py (A vs B × Brier/AURC × DEF-A/DEF-B).
- Killed a K=5 sampling regen experiment (user decided DARC-for-all is the answer, not K).
