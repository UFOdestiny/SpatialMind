"""
calibration.py - Post-hoc trace-score calibration/recalibration.

Two families, both FIT ON A REFERENCE (validation) SPLIT and APPLIED TO TEST, so
test labels never influence the transform (no leakage). This mirrors how the eval
pipeline already fits the aggregation rule and baseline normalization on validation.

    1. StandardCalibrator (baselines): a monotonic map of the scalar trace score
       (temperature / Platt in logit space). Monotonic => AUROC / PR-AUC are
       provably unchanged; only calibration (ECE/Brier/NLL) and threshold metrics
       move. This is the conventional post-hoc calibration baselines get.

    2. StructuralCalibrator (optional secondary analysis): a regularized logistic model over
       STRUCTURAL FEATURES of the per-claim probability sequence, ANCHORED on the
       raw trace score (the raw logit is an input feature). Because it re-reads the
       claim vector, the resulting score is NON-monotonic in the old score, so it
       can improve RANKING (AUROC/PR-AUC) as well as calibration — the whole point
       being that claim-level structure carries reorder-able signal. Strong L2
       regularization makes it degrade gracefully to the raw score when structure
       is uninformative.

Design principle (fairness): the headline comparison gives every learned method
the same rank-preserving StandardCalibrator.  StructuralCalibrator is available
only as an explicitly requested secondary analysis because it can change ranking.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.special import expit, logit

EPS = 1e-6


def _logit(p):
    return logit(np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS))


# --------------------------------------------------------------------------- #
# Structural features of one trace's claim-probability sequence
# --------------------------------------------------------------------------- #
def _trace_features(claim_probs: Sequence[float], conclusion_prob: Optional[float] = None) -> np.ndarray:
    """Structural descriptors of a claim-probability sequence.

    conclusion_prob: if given, the conclusion claim's probability; otherwise the
        last claim is treated as the conclusion (matches aggregation.py fallback).
    """
    p = np.clip(np.asarray(list(claim_probs), dtype=np.float64), EPS, 1 - EPS)
    if p.size == 0:
        p = np.array([0.5])
    conc = float(conclusion_prob) if conclusion_prob is not None else float(p[-1])
    conc = float(np.clip(conc, EPS, 1 - EPS))
    reas = p[:-1] if p.size > 1 else p
    zr = _logit(reas)
    return np.array([
        conc,                                             # trust-the-answer
        reas.mean(),                                      # trust-the-chain
        reas.min(),                                       # weakest link
        reas.max(),
        reas.std() if reas.size > 1 else 0.0,             # chain disagreement
        float(np.exp(np.log(reas).mean())),              # geometric mean (product consistency)
        float(np.quantile(reas, 0.10)),                  # soft weakest link
        float(np.quantile(reas, 0.25)),
        float(np.average(p, weights=np.arange(1, p.size + 1))),  # position-weighted (late errors)
        float(-np.log(np.mean(np.exp(-zr)))) if zr.size else float(_logit([conc])[0]),  # logit soft-min
        float(zr.mean()) if zr.size else 0.0,
        float(zr.min()) if zr.size else 0.0,
        float(p.size),                                    # chain length
        conc - reas.min(),                                # answer vs weakest-premise gap
    ], dtype=np.float64)


N_STRUCT_FEATURES = _trace_features([0.5, 0.5]).shape[0]


# --------------------------------------------------------------------------- #
# Standard monotonic calibrator (baselines)
# --------------------------------------------------------------------------- #
class StandardCalibrator:
    """Platt scaling in logit space: p' = sigmoid(a*logit(s) + b).

    Strictly monotonic (a>0 by construction) => rank metrics unchanged; fixes prior shift (b) and
    sharpness (a). Falls back to identity if the fit is degenerate.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "StandardCalibrator":
        from scipy.optimize import minimize
        z = _logit(scores)
        y = np.asarray(labels, dtype=np.float64)
        if np.unique(y).size < 2:
            return self  # cannot fit; identity

        # Optimise log(a), rather than a itself.  An unconstrained Platt fit can
        # choose a negative slope on a noisy validation split and silently
        # reverse every test ranking.  Besides being unfair, that contradicts
        # the advertised rank-preserving evaluation protocol.
        def nll(w):
            log_a, b = w
            a = np.exp(log_a)
            logits = a * z + b
            return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

        prior = float(np.clip(y.mean(), EPS, 1 - EPS))
        x0 = np.array([0.0, float(logit(prior))], dtype=np.float64)
        r = minimize(
            nll, x0, method="L-BFGS-B",
            bounds=[(np.log(1e-6), np.log(1e3)), (-30.0, 30.0)],
        )
        if r.success or np.isfinite(r.fun):
            self.a, self.b = float(np.exp(r.x[0])), float(r.x[1])
            self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(scores, dtype=np.float64)
        return expit(self.a * _logit(scores) + self.b)


# --------------------------------------------------------------------------- #
# Structure-aware calibrator (SpatialMind)
# --------------------------------------------------------------------------- #
class StructuralCalibrator:
    """Regularized logistic stacker over structural claim features + raw trace logit.

    Anchored on the raw trace score (its logit is feature 0), so with strong
    regularization it reduces to a monotonic map of the raw score; the extra
    structural features let it reorder traces when claim structure is informative.

    Fit on validation, apply to test — no leakage.
    """

    def __init__(self, C: float = 0.01):
        self.C = C
        self.model = None
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    @staticmethod
    def _design(trace_scores, claim_probs_per_trace, conclusion_probs=None):
        z = _logit(trace_scores).reshape(-1, 1)
        feats = []
        for i, cp in enumerate(claim_probs_per_trace):
            cprob = None if conclusion_probs is None else conclusion_probs[i]
            feats.append(_trace_features(cp, cprob))
        F = np.asarray(feats, dtype=np.float64)
        return z, F

    def fit(self, trace_scores, claim_probs_per_trace, labels, conclusion_probs=None) -> "StructuralCalibrator":
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(labels, dtype=int)
        z, F = self._design(trace_scores, claim_probs_per_trace, conclusion_probs)
        if np.unique(y).size < 2 or len(y) < 8:
            return self  # not enough signal; identity fallback at transform time
        # Standardize structural features (raw logit anchor kept on its natural scale).
        self.mean_ = F.mean(axis=0)
        self.std_ = F.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Fs = (F - self.mean_) / self.std_
        X = np.hstack([z, Fs])
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        self.fitted = True
        return self

    def transform(self, trace_scores, claim_probs_per_trace, conclusion_probs=None) -> np.ndarray:
        trace_scores = np.asarray(trace_scores, dtype=np.float64)
        if not self.fitted:
            return trace_scores
        z, F = self._design(trace_scores, claim_probs_per_trace, conclusion_probs)
        Fs = (F - self.mean_) / self.std_
        X = np.hstack([z, Fs])
        return self.model.predict_proba(X)[:, 1]
