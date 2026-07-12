# Data-leakage audit

This repository uses a fail-closed protocol. A run is reportable only after
`scripts/leakage_audit.py` exits successfully for every evaluated cache.

## Information boundaries

| Stage | May read | Must not affect model inputs |
| --- | --- | --- |
| Generation | trusted context, question, answer options | reference answer, correctness label |
| Claim extraction | question, generated response | reference answer, trace/claim labels |
| Constraint analysis | context, question, generated claims | ground truth, `label`, `verified`, judge verdict |
| Reasoning judge | trusted context, question, ordered reasoning claims | reference answer, final-answer verdict |
| Training | frozen token features, masks/types, constraint features; train labels only as loss targets | validation/test labels |
| Calibration | validation scores and validation labels | test labels/statistics |
| Evaluation | frozen model and test inputs; test labels only after scoring | any fitting or model selection |

The conclusion label is the final-answer correctness target and is computed by exact
match for classification datasets or a separate answer judge for free-form datasets.
It is supervision, never an input to constraint construction.

## Enforced controls

- `analyze_trace` accepts only `context`, `question`, and generated `claims`.
- vLLM backbone generation is constrained to a JSON schema containing 1--6 bounded
  reasoning strings and one conclusion; the 768-token budget was GPU-smoke-tested
  to close the contract without format exclusions.
- Native cache loading requires schema v2, complete 0/1 trace labels, complete 0/1
  labels for every claim, and exact claim/constraint alignment. There is no fallback
  from a missing claim label to the sample label.
- The Stage-2 judge receives trusted premises and ordered reasoning claims. Its API has
  no ground-truth or final-answer-verdict argument.
- Both judge stages use vLLM JSON-schema guided decoding. Stage-2 output must be
  direct JSON with an exact-length integer 0/1 array. Missing,
  malformed, coerced, padded, truncated, or inferred labels are forbidden.
- Any judge/extraction contract failure physically removes the whole sample and records
  the count, reason, and trace-label distribution in the manifest. Each source sample
  is judged at most once; force-judging a previously filtered split is not reportable.
  Parser `unknown` and source infeasibility are kept and reported because deleting
  method failures would bias evaluation.
- Baseline estimators receive a sanitized dictionary with `label`, `verified`,
  `ground_truth`, and `answer_correct` removed.
- A disjoint validation cache is mandatory. Baseline normalization, aggregation, and
  calibration cannot fall back to test.
- Headline Platt scaling parameterizes its slope as `exp(alpha)`, so it cannot reverse
  rankings or change AUROC.
- The leakage audit hashes `(context, question)` to check train/validation/test overlap
  and recomputes every constraint tensor solely from deployment-visible inputs. It also
  validates physical/manifest counts, exact claim structure, exclusion accounting, and
  one-pass full-split judge counts.

Because physical deletion can be label- or difficulty-dependent, every table using a
filtered cache must disclose retention and deletion rates by trace label. Retained-set
metrics alone are not sufficient evidence; coverage-aware sensitivity results must be
reported when exclusions are materially imbalanced.

## Automated regression tests

`tests/test_leakage_controls.py` verifies the judge API, strict JSON parsing, and
baseline-input sanitization. `tests/test_constraint_cache.py` verifies that incomplete
claim labels fail closed. The complete test suite must pass before a run is summarized.

## Interpretation of a strong constraint score

The solver can obtain a strong score because it recomputes whether the model's predicted
relation is compatible with the supplied scene and question. This is task-specific
verification, not access to the reference answer. We therefore report a deterministic
`constraint_rule` baseline and explicit parser/source-feasibility coverage so the gain
cannot be attributed vaguely to hidden-state uncertainty.
