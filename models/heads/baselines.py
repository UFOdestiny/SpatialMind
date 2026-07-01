"""
baselines.py - Supervised claim-level baseline heads.

Faithful re-implementations of the supervised UQ probes we compare against, all
adapted to the SpatialMind head contract (return a list of per-claim logit
tensors from `forward_claims`; the base class packs and, for baselines, leaves
`trace_logit=None` so the shared claim->trace aggregation handles the readout).

Included (paper set + full zoo):
    saplma        : SAPLMA per-token MLP, claim-pooled            (Azaria & Mitchell 2023)
    factoscope    : scope-contrast local-vs-global CLS head       (He et al. 2024)
    lookback_lens : prefix-lookback dynamics + CLS transformer    (Chuang et al. 2024)
    uhead         : LUH uncertainty head (CLS+entity marker)      (Shelmanov et al. 2025)
    luh_light     : lighter single-projection LUH variant
    linear        : linear probe on claim-mean features
    mlp           : 2-layer MLP on claim-mean features
    gated_mlp     : gated 2-branch MLP on claim-mean features
    cnn           : 1D-CNN over the claim token span

Parameter budgets are calibrated near the LUH-Light head so efficiency
comparisons are fair (see the paper's efficiency table).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import UncertaintyHeadBase


# --------------------------------------------------------------------------- #
# Shared claim CLS-transformer encoder used by several transformer baselines.
# --------------------------------------------------------------------------- #
class _ClaimCLSEncoder(nn.Module):
    """Project tokens, mark the claim span, prepend CLS, run a Transformer, read CLS.

    This is the canonical LUH-style claim encoder shared by uhead / luh_light /
    lookback_lens / factoscope so their capacities stay comparable.
    """

    def __init__(self, feature_dim, head_dim, n_layers, n_heads, dropout, max_seq_len, use_entity=True):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.use_entity = use_entity
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(), nn.Dropout(dropout),
        )
        if use_entity:
            self.entity_embedding = nn.Embedding(2, head_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, head_dim))
        self.position_embeddings = nn.Embedding(self.max_seq_len, head_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=head_dim, nhead=n_heads, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def encode_trace(self, features_i, valid_mask_i, claim_mask):
        """Return (n_claims, head_dim) CLS states + projected tokens for one trace."""
        device = features_i.device
        x = self.proj(features_i)                        # (L, d_h)
        L = x.shape[0]
        seq_len = min(L, self.max_seq_len)
        x = x[:seq_len]
        cm = claim_mask
        if cm.shape[1] > seq_len:
            cm = cm[:, :seq_len]
        elif cm.shape[1] < seq_len:
            cm = F.pad(cm, (0, seq_len - cm.shape[1]), value=0.0)
        n_claims = cm.shape[0]

        base = x.unsqueeze(0).expand(n_claims, -1, -1)   # (C, L, d_h)
        if self.use_entity:
            base = base + self.entity_embedding(cm.long())
        cls = self.cls_token.expand(n_claims, -1, -1)
        seq_in = torch.cat([cls, base], dim=1)
        in_len = min(seq_in.shape[1], self.max_seq_len)
        seq_in = seq_in[:, :in_len, :]
        pos_ids = torch.arange(in_len, device=device).unsqueeze(0).expand(n_claims, -1)
        seq_in = seq_in + self.position_embeddings(pos_ids)

        src_pad = (valid_mask_i[:seq_len] == 0)
        cls_pad = torch.zeros(n_claims, 1, dtype=torch.bool, device=device)
        pad_mask = torch.cat([cls_pad, src_pad.unsqueeze(0).expand(n_claims, -1)], dim=1)[:, :in_len]
        out = self.transformer(seq_in, src_key_padding_mask=pad_mask)
        return out[:, 0, :], x  # CLS states, projected tokens


class _ClaimCLSBaseline(UncertaintyHeadBase):
    """Base for transformer-CLS claim baselines (uhead / luh_light)."""

    supports_claim_inputs = True
    emits_trace_logit = False
    _use_entity = True

    def __init__(self, feature_dim, num_classes=1, head_dim=256, n_layers=2,
                 n_heads=8, dropout=0.1, max_seq_len=512, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.enc = _ClaimCLSEncoder(feature_dim, head_dim, n_layers, n_heads,
                                    dropout, max_seq_len, use_entity=self._use_entity)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_dim, num_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.enc.cls_token, std=0.02)

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        results = []
        for i in range(features.shape[0]):
            cm = claim_masks[i].to(features.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=features.device))
                continue
            cls_states, _ = self.enc.encode_trace(features[i], attention_mask[i].float(), cm)
            results.append(self.classifier(cls_states))
        return results


class UHead(_ClaimCLSBaseline):
    """LUH uncertainty head (Shelmanov et al. 2025): 2-layer transformer + entity marker."""
    _use_entity = True


class LUHLightHead(_ClaimCLSBaseline):
    """Lighter LUH variant: single-projection transformer CLS head."""
    _use_entity = True

    def __init__(self, feature_dim, num_classes=1, head_dim=256, n_layers=1,
                 n_heads=8, dropout=0.1, max_seq_len=512, **kwargs):
        super().__init__(feature_dim, num_classes, head_dim, n_layers, n_heads,
                         dropout, max_seq_len, **kwargs)


class LookbackLensHead(UncertaintyHeadBase):
    """Lookback Lens (Chuang et al. 2024): prefix-mean lookback dynamics + CLS transformer."""

    supports_claim_inputs = True

    def __init__(self, feature_dim, num_classes=1, head_dim=256, n_layers=2,
                 n_heads=8, dropout=0.1, max_seq_len=512, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.max_seq_len = int(max_seq_len)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.lookback_mlp = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim), nn.LayerNorm(head_dim), nn.GELU(), nn.Dropout(dropout),
        )
        # Entity embedding marks which tokens belong to the claim under evaluation,
        # so different claims of the same trace get distinct representations.
        self.entity_embedding = nn.Embedding(2, head_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, head_dim))
        self.position_embeddings = nn.Embedding(self.max_seq_len, head_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=head_dim, nhead=n_heads, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_dim, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        bsz, L, _ = features.shape
        seq_len = min(L, self.max_seq_len)
        feats = features[:, :seq_len, :]
        am = attention_mask[:, :seq_len]
        x = self.proj(feats)
        valid = am.unsqueeze(-1).to(x.dtype)
        prefix_sum = (x * valid).cumsum(dim=1)
        prefix_cnt = valid.cumsum(dim=1).clamp(min=1.0)
        lookback_delta = x - prefix_sum / prefix_cnt
        x = self.lookback_mlp(torch.cat([x, lookback_delta], dim=-1))
        src_pad = am == 0

        results = []
        for i in range(bsz):
            cm = claim_masks[i].to(x.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=x.device))
                continue
            if cm.shape[1] != seq_len:
                cm = cm[:, :seq_len] if cm.shape[1] > seq_len else F.pad(cm, (0, seq_len - cm.shape[1]))
            n = cm.shape[0]
            base = x[i].unsqueeze(0).expand(n, -1, -1) + self.entity_embedding(cm.long())
            cls = self.cls_token.expand(n, -1, -1)
            seq_in = torch.cat([cls, base], dim=1)
            in_len = min(seq_in.shape[1], self.max_seq_len)
            seq_in = seq_in[:, :in_len, :]
            pos = torch.arange(in_len, device=x.device).unsqueeze(0).expand(n, -1)
            seq_in = seq_in + self.position_embeddings(pos)
            cls_pad = torch.zeros(n, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_pad, src_pad[i].unsqueeze(0).expand(n, -1)], dim=1)[:, :in_len]
            out = self.transformer(seq_in, src_key_padding_mask=pad_mask)
            results.append(self.classifier(out[:, 0, :]))
        return results


class FactoscopeHead(UncertaintyHeadBase):
    """Factoscope (He et al. 2024): local-vs-global scope contrast on claim CLS."""

    supports_claim_inputs = True

    def __init__(self, feature_dim, num_classes=1, head_dim=256, n_layers=2,
                 n_heads=8, dropout=0.1, max_seq_len=512, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.enc = _ClaimCLSEncoder(feature_dim, head_dim, n_layers, n_heads,
                                    dropout, max_seq_len, use_entity=True)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim * 3, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_dim, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.enc.cls_token, std=0.02)

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        results = []
        for i in range(features.shape[0]):
            cm = claim_masks[i].to(features.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=features.device))
                continue
            cls_states, x = self.enc.encode_trace(features[i], attention_mask[i].float(), cm)
            n = cls_states.shape[0]
            global_vec = self.masked_mean(x.unsqueeze(0), attention_mask[i][:x.shape[0]].unsqueeze(0))
            global_vec = global_vec.expand(n, -1)
            gap = torch.abs(cls_states - global_vec)
            results.append(self.classifier(torch.cat([cls_states, global_vec, gap], dim=-1)))
        return results


class SaplmaHead(UncertaintyHeadBase):
    """SAPLMA (Azaria & Mitchell 2023): 3-hidden-layer MLP on per-token logits, claim-pooled."""

    supports_claim_inputs = True

    def __init__(self, feature_dim, num_classes=1, internal_dim1=256,
                 internal_dim2=128, internal_dim3=64, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, internal_dim1), nn.ReLU(),
            nn.Linear(internal_dim1, internal_dim2), nn.ReLU(),
            nn.Linear(internal_dim2, internal_dim3), nn.ReLU(),
            nn.Linear(internal_dim3, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        token_logits = self.layers(features)  # (B, L, C)
        results = []
        for i in range(features.shape[0]):
            cm = claim_masks[i].to(features.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=features.device))
                continue
            cm = cm * attention_mask[i].float().unsqueeze(0)
            denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
            results.append((cm @ token_logits[i]) / denom)
        return results


class _ClaimPoolMLP(UncertaintyHeadBase):
    """Base for heads that operate on claim-mean-pooled raw features."""

    supports_claim_inputs = True

    def build_scorer(self):
        raise NotImplementedError

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        results = []
        for i in range(features.shape[0]):
            cm = claim_masks[i].to(features.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=features.device))
                continue
            proto = self.claim_prototype(features[i], cm, attention_mask[i].float())  # (C, D)
            results.append(self.scorer(proto))
        return results


class LinearHead(_ClaimPoolMLP):
    """Linear probe on claim-mean features."""

    def __init__(self, feature_dim, num_classes=1, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.scorer = nn.Linear(feature_dim, num_classes)
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)


class MLPHead(_ClaimPoolMLP):
    """2-hidden-layer MLP on claim-mean features."""

    def __init__(self, feature_dim, num_classes=1, head_dim=256, dropout=0.1, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(),
            nn.Linear(head_dim // 2, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GatedMLPHead(_ClaimPoolMLP):
    """Gated two-branch MLP on claim-mean features."""

    def __init__(self, feature_dim, num_classes=1, head_dim=256, dropout=0.1, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.proj = nn.Linear(feature_dim, head_dim)
        self.value = nn.Linear(head_dim, head_dim)
        self.gate = nn.Linear(head_dim, head_dim)
        self.norm = nn.LayerNorm(head_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(head_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def scorer(self, proto):
        h = F.gelu(self.proj(proto))
        h = self.norm(self.value(h) * torch.sigmoid(self.gate(h)))
        return self.out(self.drop(h))


class CNNHead(UncertaintyHeadBase):
    """1D-CNN over the claim token span (temporal convolution baseline)."""

    supports_claim_inputs = True

    def __init__(self, feature_dim, num_classes=1, head_dim=256, dropout=0.1,
                 kernel_size=3, max_seq_len=512, **kwargs):
        super().__init__(feature_dim, num_classes)
        self.max_seq_len = int(max_seq_len)
        self.proj = nn.Linear(feature_dim, head_dim)
        self.conv1 = nn.Conv1d(head_dim, head_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(head_dim, head_dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(head_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_claims(self, features, attention_mask, claim_masks, claim_types=None):
        bsz, L, _ = features.shape
        seq_len = min(L, self.max_seq_len)
        x = self.proj(features[:, :seq_len, :])                     # (B, L, d_h)
        x = x.transpose(1, 2)                                       # (B, d_h, L)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x)).transpose(1, 2)                   # (B, L, d_h)
        x = self.norm(self.drop(x))
        results = []
        for i in range(bsz):
            cm = claim_masks[i].to(x.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                results.append(torch.zeros(0, self.num_classes, device=x.device))
                continue
            if cm.shape[1] != seq_len:
                cm = cm[:, :seq_len] if cm.shape[1] > seq_len else F.pad(cm, (0, seq_len - cm.shape[1]))
            cm = cm * attention_mask[i][:seq_len].float().unsqueeze(0)
            denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
            proto = (cm @ x[i]) / denom
            results.append(self.classifier(proto))
        return results
