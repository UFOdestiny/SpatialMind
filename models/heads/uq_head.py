"""
uq_head.py - SpatialMind claim-aware uncertainty head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import UncertaintyHeadBase


class UQHead(UncertaintyHeadBase):
    supports_claim_inputs = True

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.15,
        max_seq_len: int = 512,
        **kwargs,
    ):
        super().__init__(feature_dim, num_classes)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = head_dim

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.entity_embedding = nn.Embedding(2, head_dim)
        self.claim_type_embedding = nn.Embedding(3, head_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, head_dim))
        self.position_embeddings = nn.Embedding(self.max_seq_len, head_dim)
        
        self.claim_pos_embedding = nn.Embedding(64, head_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=n_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.claim_lstm = nn.LSTM(
            input_size=head_dim,
            hidden_size=head_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.claim_lstm_norm = nn.LayerNorm(head_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(head_dim * 3, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Linear(head_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for name, param in self.claim_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    @staticmethod
    def _pack_claim_logits(
        logits_per_sample: list[torch.Tensor],
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not logits_per_sample:
            return torch.zeros(0, num_classes, device=device)
        max_claims = max(x.shape[0] for x in logits_per_sample)
        padded = [
            F.pad(x, (0, 0, 0, max_claims - x.shape[0]), value=-100.0)
            for x in logits_per_sample
        ]
        return torch.stack(padded, dim=0)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        claim_masks=None,
        claim_types=None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = features.shape
        seq_len = min(seq_len, self.max_seq_len)
        features = features[:, :seq_len, :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]

        x = self.proj(features)

        if claim_masks is None:
            cls = self.cls_token.expand(bsz, -1, -1)
            seq_in = torch.cat([cls, x], dim=1)
            input_len = min(seq_in.shape[1], self.max_seq_len)
            seq_in = seq_in[:, :input_len, :]

            pos_ids = torch.arange(input_len, device=x.device).unsqueeze(0)
            seq_in = seq_in + self.position_embeddings(pos_ids)

            if attention_mask is not None:
                cls_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=x.device)
                pad_mask = torch.cat([cls_pad, attention_mask == 0], dim=1)[:, :input_len]
            else:
                pad_mask = None

            out = self.transformer(seq_in, src_key_padding_mask=pad_mask)
            local_cls = out[:, 0, :]
            global_pool = self.pool_features(x, attention_mask)

            gate = self.fusion_gate(torch.cat([local_cls, global_pool], dim=-1))
            fused = gate * local_cls + (1 - gate) * global_pool
            scope_gap = torch.abs(local_cls - global_pool)
            type_vec = torch.zeros_like(local_cls)

            return self.classifier(torch.cat([fused, scope_gap, type_vec], dim=-1))

        # Claim-level
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, dtype=torch.long, device=x.device)
        src_pad = attention_mask == 0

        global_vecs = self.pool_features(x, attention_mask)

        results = []
        for i in range(bsz):
            cm = claim_masks[i].to(x.device).float()
            if cm.numel() == 0 or cm.shape[0] == 0:
                continue
            if cm.shape[1] > seq_len:
                cm = cm[:, :seq_len]
            elif cm.shape[1] < seq_len:
                cm = F.pad(cm, (0, seq_len - cm.shape[1]), value=0.0)

            n_claims = cm.shape[0]

            ent_emb = self.entity_embedding(cm.long())
            base = x[i].unsqueeze(0).expand(n_claims, -1, -1)
            claim_tokens = base + ent_emb
            
            claim_pos_ids = torch.arange(min(n_claims, 64), device=x.device)
            if n_claims > 64:
                claim_pos_ids = F.pad(claim_pos_ids, (0, n_claims - 64), value=63)
            claim_pos_emb = self.claim_pos_embedding(claim_pos_ids)

            cls = self.cls_token.expand(n_claims, -1, -1)
            claim_input = torch.cat([cls, claim_tokens], dim=1)
            input_len = min(claim_input.shape[1], self.max_seq_len)
            claim_input = claim_input[:, :input_len, :]

            pos_ids = torch.arange(input_len, device=x.device).unsqueeze(0).expand(n_claims, -1)
            claim_input = claim_input + self.position_embeddings(pos_ids)

            item_pad = src_pad[i].unsqueeze(0).expand(n_claims, -1)
            cls_pad = torch.zeros(n_claims, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_pad, item_pad], dim=1)[:, :input_len]

            out = self.transformer(claim_input, src_key_padding_mask=pad_mask)
            local_cls = out[:, 0, :]
            
            local_cls = local_cls + claim_pos_emb

            claim_seq = local_cls.unsqueeze(0)  # (1, n_claims, head_dim)
            lstm_out, _ = self.claim_lstm(claim_seq)
            lstm_out = self.claim_lstm_norm(lstm_out.squeeze(0))

            valid = attention_mask[i].float().unsqueeze(0)
            cm_valid = cm * valid
            denom = cm_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_vec = (cm_valid @ x[i]) / denom

            gate = self.fusion_gate(torch.cat([local_cls, lstm_out], dim=-1))
            fused = gate * local_cls + (1 - gate) * lstm_out

            global_vec = global_vecs[i].unsqueeze(0).expand(n_claims, -1)
            scope_gap = torch.abs(fused - global_vec)

            if claim_types is not None:
                t = claim_types[i].to(x.device).long()
                if t.shape[0] > n_claims:
                    t = t[:n_claims]
                elif t.shape[0] < n_claims:
                    t = F.pad(t, (0, n_claims - t.shape[0]), value=2)
            else:
                t = torch.full((n_claims,), 2, dtype=torch.long, device=x.device)
            type_vec = self.claim_type_embedding(t)

            results.append(self.classifier(torch.cat([fused, scope_gap, type_vec], dim=-1)))

        return self._pack_claim_logits(results, self.num_classes, x.device)
