"""
spatialmind_head.py - The SpatialMind claim-level UQ head (our method).

A structure-aware, multi-task head that scores the correctness of every spatial
claim in a reasoning trace and jointly predicts a trace-level reliability logit.
It operationalizes the paper's design while being trained end-to-end so that the
reported SAMPLE-LEVEL metric is directly optimized.

Pipeline for one trace with token features H = [x_1..x_T] and claims (c_k, t_k, m_k):

  1. Project tokens        u_t = MLP(LN(x_t))                        (D -> d_h)
  2. Claim marking         mark span + claim-type, one Transformer pass per claim,
                           read a CLS state z_k (local contextual evidence).
  3. Sequential integration BiLSTM over ordered [z_1..z_K] with a claim-position
                           embedding, then a learned gate fuses local z_k and the
                           ordered state ~z_k  ->  hat_z_k.
  4. Global anchor         h^g = masked-mean(H); the discrepancy |hat_z_k - h^g|
                           exposes claims that look locally fluent but contradict
                           the global spatial layout.
  5. Span statistics       normalized position/length descriptor phi(m_k) tells the
                           scorer whether a claim is early/mid/late in the chain.
  6. Reliability bank      a small learned bank of reusable reliability profiles
                           (restate / one-hop / multi-hop / conclusion) selected
                           per claim; fused with the claim seed by a gate.
  7. Per-claim logit       p_k from [prototype ; reliability state ; global gap ; type].
  8. Trace logit (multi-task): a learned read-out over the claim states + global
                           anchor, giving a differentiable trace-level score whose
                           BCE against the trace label is added to the claim BCE.

Everything trains on cached frozen features; the backbone is never updated.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import UncertaintyHeadBase

REASONING_TYPE_ID = 1
CONCLUSION_TYPE_ID = 2
_NUM_CLAIM_TYPES = 3  # {0 unused, 1 reasoning, 2 conclusion}
_MAX_CLAIM_POS = 64
_NUM_RELIABILITY_PROTOTYPES = 4


class SpatialMindHead(UncertaintyHeadBase):
    """Structure-aware, multi-task claim-level UQ head."""

    supports_claim_inputs = True
    emits_trace_logit = True

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        head_dim: int = 256,
        n_layers: int = 1,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_reliability_bank: bool = True,
        use_cross_claim: bool = True,
        use_scope: bool = True,
        use_type: bool = True,
        **kwargs,
    ):
        super().__init__(feature_dim, num_classes)
        self.max_seq_len = int(max_seq_len)
        self.head_dim = head_dim
        self.use_reliability_bank = use_reliability_bank
        self.use_cross_claim = use_cross_claim
        self.use_scope = use_scope
        self.use_type = use_type

        # 1. Token projection.
        self.input_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Project raw frozen features (for prototypes / anchor) into scorer space.
        self.raw_proj = nn.Linear(feature_dim, head_dim)

        # 2. Claim marking + local encoder.
        self.span_embedding = nn.Embedding(2, head_dim)            # in-span indicator
        self.type_embedding = nn.Embedding(_NUM_CLAIM_TYPES, head_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, head_dim))
        self.token_pos_embedding = nn.Embedding(self.max_seq_len, head_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=head_dim, nhead=n_heads, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 3. Sequential claim integration.
        self.claim_pos_embedding = nn.Embedding(_MAX_CLAIM_POS, head_dim)
        if use_cross_claim:
            # Bidirectional output width is 2*hidden; keep it consistent with the
            # norm even when head_dim is odd (config-settable).
            lstm_hidden = head_dim // 2
            lstm_out_dim = 2 * lstm_hidden
            self.claim_lstm = nn.LSTM(
                input_size=head_dim, hidden_size=lstm_hidden,
                num_layers=1, batch_first=True, bidirectional=True,
            )
            self.claim_lstm_norm = nn.LayerNorm(lstm_out_dim)
            # Project BiLSTM output back to head_dim so the fusion gate and residuals
            # stay in a single width regardless of parity.
            self.claim_lstm_proj = nn.Linear(lstm_out_dim, head_dim) if lstm_out_dim != head_dim else nn.Identity()
            self.seq_gate = nn.Linear(head_dim * 2, head_dim)

        # 5. Span-position statistics MLP.
        if use_scope:
            self.scope_mlp = nn.Sequential(
                nn.Linear(4, head_dim), nn.GELU(), nn.Linear(head_dim, head_dim),
            )

        # Reliability-aware claim seed projection.
        # seed input = [prototype(d_h) ; fused(d_h) ; anchor(d_h) ; extras(d_h)]
        self.seed_proj = nn.Linear(head_dim * 4, head_dim)

        # 6. Reliability pattern bank.
        if use_reliability_bank:
            self.reliability_bank = nn.Parameter(
                torch.randn(_NUM_RELIABILITY_PROTOTYPES, head_dim) * 0.02
            )
            self.bank_selector = nn.Linear(head_dim * 3, _NUM_RELIABILITY_PROTOTYPES)
            self.reliability_gate = nn.Linear(head_dim * 4, head_dim)
            self.reliability_temp = nn.Parameter(torch.ones(1))
            self.reliability_residual = nn.Sequential(
                nn.Linear(head_dim, head_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(head_dim, head_dim),
            )

        # 7. Per-claim scoring head.
        # input = [prototype ; reliability state ; global gap ; type]
        self.claim_scorer = nn.Sequential(
            nn.Linear(head_dim * 4, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_dim, num_classes),
        )

        # 8. Multi-task trace read-out.
        # Aggregates claim states (attention-pooled) + global anchor + conclusion state.
        self.trace_query = nn.Parameter(torch.randn(1, head_dim) * 0.02)
        self.trace_attn = nn.Linear(head_dim, 1)
        self.trace_scorer = nn.Sequential(
            nn.Linear(head_dim * 3, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_dim, num_classes),
        )

        self._init_weights()
        # Cache of per-claim reliability states from the last forward_claims call,
        # so forward_trace can reuse them without recomputation.
        self._cached_states: List[dict] = []

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.trace_query, std=0.02)

    # ------------------------------------------------------------------ #
    def _scope_descriptor(self, claim_mask: torch.Tensor, valid_len: float) -> torch.Tensor:
        """Normalized (|m|/|v|, log-len ratio, centroid, span-width) per claim -> (C, 4)."""
        C, L = claim_mask.shape
        device = claim_mask.device
        positions = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)
        size = claim_mask.sum(dim=1).clamp(min=1.0)                                    # (C,)
        v = max(valid_len, 1.0)
        frac = (size / v).clamp(0.0, 1.0)
        log_ratio = torch.log1p(size) / float(torch.log1p(torch.tensor(v)))
        centroid = (claim_mask * positions).sum(dim=1) / (size * v)
        has_tok = claim_mask > 0
        first = torch.where(has_tok.any(dim=1),
                            torch.argmax(has_tok.float(), dim=1).float(),
                            torch.zeros(C, device=device))
        rev = torch.flip(has_tok.float(), dims=[1])
        last = torch.where(has_tok.any(dim=1),
                           (L - 1 - torch.argmax(rev, dim=1)).float(),
                           torch.zeros(C, device=device))
        width = ((last - first + 1.0) / v).clamp(0.0, 1.0)
        return torch.stack([frac, log_ratio, centroid.clamp(0.0, 1.0), width], dim=-1)

    def forward_claims(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_masks: List[torch.Tensor],
        claim_types: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Fully-batched claim scoring.

        All claims across the batch are flattened into a single Transformer pass
        (instead of one pass per trace), and the cross-claim BiLSTM is run with
        pack_padded_sequence over the per-trace claim sequences. This is
        numerically equivalent to the per-trace loop (self-attention and LSTM are
        independent across rows/sequences) but launches O(1) big kernels instead
        of O(batch) small ones. See `_forward_claims_loop` for the reference.
        """
        bsz, seq_len, _ = features.shape
        seq_len = min(seq_len, self.max_seq_len)
        features = features[:, :seq_len, :]
        attention_mask = attention_mask[:, :seq_len]
        device = features.device

        u = self.proj(self.input_norm(features))                 # (B, L, d_h)
        raw = self.raw_proj(features)                            # (B, L, d_h)
        global_anchor = self.masked_mean(raw, attention_mask)     # (B, d_h)
        src_pad = attention_mask == 0                            # (B, L)

        # ---- Flatten claims across the batch, preserving per-trace order. ----
        counts: List[int] = []
        cm_list: List[torch.Tensor] = []
        type_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        trace_ids: List[int] = []
        for i in range(bsz):
            cm = claim_masks[i].to(device).float()
            n = 0 if (cm.numel() == 0) else cm.shape[0]
            if n > 0:
                if cm.shape[1] > seq_len:
                    cm = cm[:, :seq_len]
                elif cm.shape[1] < seq_len:
                    cm = F.pad(cm, (0, seq_len - cm.shape[1]), value=0.0)
                cm_list.append(cm)
                if self.use_type and claim_types is not None:
                    t = claim_types[i].to(device).long()
                    t = t[:n] if t.shape[0] > n else F.pad(t, (0, n - t.shape[0]), value=CONCLUSION_TYPE_ID)
                else:
                    t = torch.full((n,), CONCLUSION_TYPE_ID, dtype=torch.long, device=device)
                type_list.append(t)
                cp = torch.arange(n, device=device).clamp(max=_MAX_CLAIM_POS - 1)
                pos_list.append(cp)
                trace_ids.extend([i] * n)
            counts.append(n)

        total = sum(counts)
        if total == 0:
            self._cached_states = [{} for _ in range(bsz)]
            return [torch.zeros(0, self.num_classes, device=device) for _ in range(bsz)]

        cm_all = torch.cat(cm_list, dim=0)                        # (T, L)
        t_all = torch.cat(type_list, dim=0)                       # (T,)
        pos_all = torch.cat(pos_list, dim=0)                      # (T,)
        trace_idx = torch.tensor(trace_ids, device=device, dtype=torch.long)  # (T,)
        type_vec = self.type_embedding(t_all)                     # (T, d_h)

        # ---- 2. Claim marking + local encoding (ONE Transformer pass) ----
        base = u[trace_idx]                                       # (T, L, d_h)
        marked = base + self.span_embedding(cm_all.long())
        if self.use_type:
            marked = marked + type_vec.unsqueeze(1)
        cls = self.cls_token.expand(total, -1, -1)                # (T, 1, d_h)
        seq_in = torch.cat([cls, marked], dim=1)                  # (T, 1+L, d_h)
        in_len = min(seq_in.shape[1], self.max_seq_len)
        seq_in = seq_in[:, :in_len, :]
        pos_ids = torch.arange(in_len, device=device).unsqueeze(0).expand(total, -1)
        seq_in = seq_in + self.token_pos_embedding(pos_ids)
        cls_pad = torch.zeros(total, 1, dtype=torch.bool, device=device)
        pad_mask = torch.cat([cls_pad, src_pad[trace_idx]], dim=1)[:, :in_len]
        enc = self.encoder(seq_in, src_key_padding_mask=pad_mask)
        z = enc[:, 0, :]                                          # (T, d_h)

        # ---- 3. Sequential integration (cross-claim BiLSTM via packing) ----
        z_pos = z + self.claim_pos_embedding(pos_all)
        if self.use_cross_claim:
            nz = [c for c in counts if c > 0]
            seqs = torch.split(z_pos, nz, dim=0)                  # tuple of (C_i, d_h)
            padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (n_seq, maxC, d_h)
            lengths = torch.tensor(nz, dtype=torch.long)
            packed = nn.utils.rnn.pack_padded_sequence(
                padded, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.claim_lstm(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            z_seq_flat = torch.cat([unpacked[j, :nz[j]] for j in range(len(nz))], dim=0)  # (T, 2*hidden)
            z_seq = self.claim_lstm_proj(self.claim_lstm_norm(z_seq_flat))
            gate = torch.sigmoid(self.seq_gate(torch.cat([z, z_seq], dim=-1)))
            fused = gate * z + (1.0 - gate) * z_seq
        else:
            fused = z_pos

        # ---- 4. Prototype + global anchor/gap (batched) ----
        anchor = global_anchor[trace_idx]                         # (T, d_h)
        prototype = self._claim_prototype_batched(raw, attention_mask, cm_all, trace_idx)
        global_gap = torch.abs(fused - anchor)

        # ---- 5. Span statistics (batched) ----
        if self.use_scope:
            valid_len_all = attention_mask.float().sum(dim=1)[trace_idx]   # (T,)
            scope = self.scope_mlp(self._scope_descriptor_batched(cm_all, valid_len_all))
        else:
            scope = torch.zeros(total, self.head_dim, device=device)

        extras = scope + (type_vec if self.use_type else 0.0) + global_gap
        seed = self.seed_proj(torch.cat([prototype, fused, anchor, extras], dim=-1))

        # ---- 6. Reliability bank ----
        if self.use_reliability_bank:
            pi = F.softmax(self.bank_selector(
                torch.cat([seed, anchor, torch.abs(seed - anchor)], dim=-1)), dim=-1)
            bank_proto = pi @ self.reliability_bank
            temp = self.reliability_temp.clamp(min=1e-2)
            r_gate = torch.sigmoid(self.reliability_gate(
                torch.cat([seed, bank_proto, anchor, torch.abs(bank_proto - seed)], dim=-1)) / temp)
            reliab = self.reliability_residual(r_gate * seed + (1.0 - r_gate) * bank_proto) + seed
        else:
            reliab = seed

        # ---- 7. Per-claim logit ----
        reliab_gap = torch.abs(reliab - anchor)
        logits_all = self.claim_scorer(
            torch.cat([prototype, reliab, reliab_gap, type_vec], dim=-1))  # (T, C)

        # ---- Split back to per-trace, cache states for the trace read-out. ----
        per_trace_logits: List[torch.Tensor] = []
        self._cached_states = []
        offset = 0
        for i in range(bsz):
            c = counts[i]
            if c == 0:
                per_trace_logits.append(torch.zeros(0, self.num_classes, device=device))
                self._cached_states.append({})
                continue
            sl = slice(offset, offset + c)
            per_trace_logits.append(logits_all[sl])
            self._cached_states.append(
                {"reliab": reliab[sl], "type": t_all[sl], "anchor": global_anchor[i]})
            offset += c
        return per_trace_logits

    @staticmethod
    def _claim_prototype_batched(raw, attention_mask, cm_all, trace_idx):
        """Masked mean-pool raw token features within each claim span (batched)."""
        valid = attention_mask.float()[trace_idx]                 # (T, L)
        cm = cm_all * valid
        denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        raw_rows = raw[trace_idx]                                  # (T, L, d_h)
        return torch.einsum("tl,tld->td", cm, raw_rows) / denom

    def _scope_descriptor_batched(self, claim_mask: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        """Batched version of `_scope_descriptor` over (T, L) masks with per-row valid_len."""
        T, L = claim_mask.shape
        device = claim_mask.device
        positions = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
        size = claim_mask.sum(dim=1).clamp(min=1.0)               # (T,)
        v = valid_len.clamp(min=1.0)                              # (T,)
        frac = (size / v).clamp(0.0, 1.0)
        log_ratio = torch.log1p(size) / torch.log1p(v)
        centroid = (claim_mask * positions).sum(dim=1) / (size * v)
        has_tok = claim_mask > 0
        any_tok = has_tok.any(dim=1)
        first = torch.where(any_tok, torch.argmax(has_tok.float(), dim=1).float(),
                            torch.zeros(T, device=device))
        rev = torch.flip(has_tok.float(), dims=[1])
        last = torch.where(any_tok, (L - 1 - torch.argmax(rev, dim=1)).float(),
                           torch.zeros(T, device=device))
        width = ((last - first + 1.0) / v).clamp(0.0, 1.0)
        return torch.stack([frac, log_ratio, centroid.clamp(0.0, 1.0), width], dim=-1)

    def forward_trace(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_logits_per_trace: List[torch.Tensor],
        claim_masks: List[torch.Tensor],
        claim_types: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """Learned trace-level logit from cached per-claim reliability states."""
        device = features.device
        outs = []
        for st in self._cached_states:
            if not st:
                outs.append(torch.zeros(self.num_classes, device=device))
                continue
            reliab = st["reliab"]                     # (C, d_h)
            t = st["type"]                            # (C,)
            anchor = st["anchor"]                     # (d_h,)
            # Attention pooling over claim states.
            attn_logits = self.trace_attn(torch.tanh(reliab + self.trace_query))  # (C, 1)
            attn = F.softmax(attn_logits, dim=0)
            pooled = (attn * reliab).sum(dim=0)       # (d_h,)
            # Conclusion state (last conclusion claim; fall back to last claim).
            conc_positions = (t == CONCLUSION_TYPE_ID).nonzero(as_tuple=False).flatten()
            conc_idx = int(conc_positions[-1].item()) if conc_positions.numel() else reliab.shape[0] - 1
            conc_state = reliab[conc_idx]
            trace_logit = self.trace_scorer(torch.cat([pooled, conc_state, anchor], dim=-1))
            outs.append(trace_logit)
        if not outs:
            return torch.zeros(0, self.num_classes, device=device)
        return torch.stack(outs, dim=0)
