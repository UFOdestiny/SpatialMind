import torch

from models.heads.spatialmind_head import (
    ConstraintNoConflictHead,
    ConstraintNoContextHead,
    ConstraintNoEntailmentHead,
    ConstraintNoRepairHead,
    ConstraintOnlyHead,
    ConstraintSpatialMindHead,
)
from models.wrapper import ClaimUQModel
from spatial_constraints import CLAIM_CONSTRAINT_DIM, TRACE_CONSTRAINT_DIM


def _batch():
    features = torch.randn(2, 8, 16)
    attention = torch.ones(2, 8)
    claim_masks = [
        torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0]]).float(),
        torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]]).float(),
    ]
    claim_types = [torch.tensor([1, 2]), torch.tensor([2])]
    claim_cf = [
        torch.randn(2, CLAIM_CONSTRAINT_DIM),
        torch.randn(1, CLAIM_CONSTRAINT_DIM),
    ]
    trace_cf = torch.randn(2, TRACE_CONSTRAINT_DIM)
    labels = [torch.tensor([1.0, 0.0]), torch.tensor([1.0])]
    trace_labels = torch.tensor([0, 1])
    return features, attention, claim_masks, claim_types, claim_cf, trace_cf, labels, trace_labels


def _run(head):
    f, a, cm, ct, ccf, tcf, labels, trace_labels = _batch()
    model = ClaimUQModel(head, trace_loss_weight=0.5)
    out = model(
        f, a, cm, ct,
        claim_constraint_features=ccf,
        trace_constraint_features=tcf,
        claim_labels=labels,
        trace_labels=trace_labels,
    )
    assert out.claim_logits.shape == (2, 2, 1)
    assert out.trace_logit is None
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()


def test_constraint_only_head_contract():
    _run(ConstraintOnlyHead(feature_dim=16, head_dim=16))


def test_hybrid_constraint_spatialmind_contract():
    _run(ConstraintSpatialMindHead(
        feature_dim=16, head_dim=16, n_layers=1, n_heads=4,
        max_seq_len=16, dropout=0.0,
    ))


def test_constraint_ablation_heads_contract():
    for head_cls in (
        ConstraintNoContextHead,
        ConstraintNoConflictHead,
        ConstraintNoEntailmentHead,
        ConstraintNoRepairHead,
    ):
        _run(head_cls(
            feature_dim=16, head_dim=16, n_layers=1, n_heads=4,
            max_seq_len=16, dropout=0.0,
        ))
