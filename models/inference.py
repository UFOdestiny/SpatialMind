"""
inference.py - Phase 3 plug-and-play adapter.

Wraps a frozen LLM with a trained UQ head so that a single `generate()` call
produces both generated text and an uncertainty score.

Usage:
    from models.inference import CausalLMWithUncertainty

    adapter = CausalLMWithUncertainty(
        base_model=llm,
        ue_head=head,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    text, uncertainty = adapter.generate(input_ids, attention_mask)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from scipy.special import expit

log = logging.getLogger(__name__)


class CausalLMWithUncertainty(nn.Module):
    """Plug-and-play adapter: LLM generates text, UQ head scores uncertainty.

    Combines a frozen causal LLM with a trained binary UQ head and feature
    extractor. The `generate()` method produces text and extracts features
    from the generation process, then passes them through the UQ head to
    get an uncertainty score.
    """

    def __init__(
        self,
        base_model: nn.Module,
        ue_head: nn.Module,
        feature_extractor: nn.Module,
        tokenizer=None,
        max_new_tokens: int = 256,
    ):
        super().__init__()
        self.base_model = base_model
        self.ue_head = ue_head
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        # Freeze the LLM
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Head in eval mode
        self.ue_head.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> Tuple[str, float]:
        """Generate text and compute uncertainty score.

        Args:
            input_ids:      (1, seq_len) input token IDs
            attention_mask:  (1, seq_len) attention mask
            max_new_tokens: maximum number of tokens to generate

        Returns:
            (generated_text, uncertainty_score) where uncertainty_score
            is in [0, 1] — higher means more likely hallucination.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        max_new = max_new_tokens or self.max_new_tokens
        output_attentions = self.feature_extractor.output_attention()

        # Generate with hidden states and attentions
        gen_output = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            output_hidden_states=True,
            output_attentions=output_attentions,
            output_scores=True,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

        generated_ids = gen_output.sequences
        prompt_len = input_ids.shape[1]
        new_token_ids = generated_ids[:, prompt_len:]

        # Decode generated text
        generated_text = ""
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(
                new_token_ids[0], skip_special_tokens=True
            ).strip()

        # Extract features from generation output
        features, feat_mask = self._extract_generation_features(
            gen_output, input_ids, attention_mask, output_attentions,
        )

        # Run UQ head. The head returns a HeadOutput; with no claim masks it
        # treats the whole valid trace as a single claim, so claim_logits is
        # (1, 1, 1). If the head emits a learned trace logit, prefer it.
        head_out = self.ue_head(features, feat_mask)
        if getattr(head_out, "trace_logit", None) is not None:
            logit_val = head_out.trace_logit.reshape(-1)[0].cpu().item()
        else:
            logit_val = head_out.claim_logits.reshape(-1)[0].cpu().item()
        confidence = float(expit(logit_val))
        # Uncertainty (higher => more likely hallucination) is 1 - confidence.
        uncertainty = 1.0 - confidence

        return generated_text, uncertainty

    def _extract_generation_features(
        self,
        gen_output,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from generation output for the UQ head.

        Reconstructs hidden states (and optionally attentions) from the
        per-step generation outputs and feeds them through the feature extractor.
        """
        device = input_ids.device
        prompt_len = input_ids.shape[1]
        num_new_tokens = gen_output.sequences.shape[1] - prompt_len

        if num_new_tokens == 0:
            # No tokens generated, return zero features
            dummy_dim = 1
            try:
                # Try to get feature dim from a forward pass
                dummy_hs = [torch.zeros(1, 1, self.base_model.config.hidden_size, device=device)]
                dummy_logits = torch.zeros(1, 1, self.base_model.config.vocab_size, device=device)
                dummy_feat = self.feature_extractor(
                    hidden_states=dummy_hs, logits=dummy_logits,
                    attention_mask=torch.ones(1, 1, device=device),
                )
                dummy_dim = dummy_feat.shape[-1]
            except Exception:
                pass
            return (
                torch.zeros(1, 1, dummy_dim, device=device),
                torch.ones(1, 1, dtype=torch.long, device=device),
            )

        # Reconstruct per-layer hidden states for generated tokens
        # gen_output.hidden_states is a tuple of length num_new_tokens
        # Each element is a tuple of (n_layers+1,) tensors of shape (batch, 1, hidden_size)
        num_layers = len(gen_output.hidden_states[0])
        hidden_size = gen_output.hidden_states[0][0].shape[-1]

        # Stack hidden states: (n_layers, batch, num_new_tokens, hidden_size)
        # Step 0 contains all prompt tokens + first gen token; take only the last.
        all_hidden = []
        for layer_idx in range(num_layers):
            step0 = gen_output.hidden_states[0][layer_idx][:, -1:, :]
            rest = [gen_output.hidden_states[step][layer_idx] for step in range(1, num_new_tokens)]
            all_hidden.append(torch.cat([step0] + rest, dim=1))  # (batch, num_new_tokens, hidden_size)

        hidden_states_tuple = tuple(all_hidden)

        # Reconstruct logits: (batch, num_new_tokens, vocab_size)
        scores = gen_output.scores  # tuple of (batch, vocab_size) per step
        logits = torch.stack(scores, dim=1)

        # Reconstruct attentions if needed
        attentions = None
        if output_attentions and hasattr(gen_output, "attentions") and gen_output.attentions is not None:
            num_attn_layers = len(gen_output.attentions[0])
            attn_list = []
            for layer_idx in range(num_attn_layers):
                layer_attns = []
                for step in range(num_new_tokens):
                    layer_attns.append(gen_output.attentions[step][layer_idx])
                attn_list.append(torch.cat(layer_attns, dim=2))
            attentions = tuple(attn_list)

        # Build attention mask for generated tokens
        gen_mask = torch.ones(1, num_new_tokens, dtype=torch.long, device=device)

        # Run feature extractor
        features = self.feature_extractor(
            hidden_states=hidden_states_tuple,
            logits=logits,
            attentions=attentions,
            attention_mask=gen_mask,
        ).float()

        return features, gen_mask

    @classmethod
    def from_pretrained(
        cls,
        base_model: nn.Module,
        head_path: str,
        feature_extractor: nn.Module,
        tokenizer=None,
        max_new_tokens: int = 256,
    ):
        """Load a trained UQ head from disk and wrap the base model.

        Args:
            base_model:        Pre-loaded causal LM
            head_path:         Directory containing head_config.json + head_weights.pth
            feature_extractor: Pre-configured feature extractor
            tokenizer:         Tokenizer for decoding generated text
            max_new_tokens:    Max tokens to generate
        """
        import json
        from models.heads import build_head

        config_path = f"{head_path}/head_config.json"
        with open(config_path, "r") as f:
            head_cfg = json.load(f)

        head = build_head(
            head_type=head_cfg["head_type"],
            feature_dim=head_cfg["feature_dim"],
            num_classes=head_cfg.get("num_classes", 1),
            head_dim=head_cfg.get("head_dim", 256),
            n_layers=head_cfg.get("n_layers", 2),
            n_heads=head_cfg.get("n_heads", 8),
            dropout=head_cfg.get("dropout", 0.1),
            max_seq_len=head_cfg.get("max_seq_len", 512),
        )
        head.load_state_dict(
            torch.load(f"{head_path}/head_weights.pth", map_location="cpu")
        )

        return cls(
            base_model=base_model,
            ue_head=head,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
