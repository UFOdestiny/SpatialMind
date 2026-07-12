import numpy as np
from sklearn.metrics import roc_auc_score

from models.calibration import StandardCalibrator


def test_standard_calibrator_cannot_reverse_ranking():
    # Deliberately anti-correlated validation labels previously caused the
    # unconstrained Platt slope to become negative and flip OOD AUROC.
    scores = np.linspace(0.05, 0.95, 40)
    labels = (scores < 0.5).astype(int)
    cal = StandardCalibrator().fit(scores, labels)
    transformed = cal.transform(scores)
    assert cal.a > 0.0
    assert np.all(np.diff(transformed) >= 0.0)
    assert roc_auc_score(labels, transformed) == roc_auc_score(labels, scores)


def test_standard_calibrator_improves_simple_prior_shift_nll():
    scores = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    cal = StandardCalibrator().fit(scores, labels)
    out = np.clip(cal.transform(scores), 1e-6, 1 - 1e-6)
    raw = np.clip(scores, 1e-6, 1 - 1e-6)
    nll = lambda p: -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    assert cal.a > 0.0
    assert nll(out) <= nll(raw) + 1e-8
