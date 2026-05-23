"""HuggingFace generation engine for Phase 1.

Encapsulates all HF model.generate() logic: tokenization, generation,
hidden-state / attention reconstruction, and feature extraction.
"""

import logging
from typing import List

import torch

from config import Config
from engine.types import GenerationResult
from models.features.combined import CombinedExtractor
from utils.common import load_llm_from_path, load_tokenizer_from_path

log = logging.getLogger(__name__)


class HFGenerationEngine:
    """Pure HuggingFace generation engine using model.generate().

    Loads a causal LM and tokenizer, generates tokens with greedy decoding,
    reconstructs hidden states / attentions from per-step generation outputs,
    and runs the feature extractor.
    """

    def __init__(self, cfg: Config, model_path: str, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Deferred import keeps process startup light before generation is needed.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading tokenizer from %s", model_path)
        self.tokenizer = load_tokenizer_from_path(
            tokenizer_cls=AutoTokenizer,
            model_path=model_path,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.hf_cache_dir,
            padding_side=cfg.model.padding_side,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info("Loading HF model from %s", model_path)
        self.llm = load_llm_from_path(
            model_cls=AutoModelForCausalLM,
            model_path=model_path,
            torch_dtype_name=cfg.model.torch_dtype,
            device_map=cfg.model.device_map,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.hf_cache_dir,
            attn_implementation="eager",
        )
        self.llm.eval()

    @property
    def model_config(self):
        return self.llm.config

    def generate_batch(
        self,
        prompts: List[str],
        feature_extractor: CombinedExtractor,
        max_new_tokens: int,
        use_attention: bool,
    ) -> GenerationResult:
        """Tokenize prompts, generate tokens, extract features.

        Args:
            prompts: List of text prompts.
            feature_extractor: CombinedExtractor to produce feature vectors.
            max_new_tokens: Maximum tokens to generate per prompt.
            use_attention: Whether to output attention weights.

        Returns:
            GenerationResult with generated texts, features, token probs,
            and log-likelihoods.
        """
        cfg = self.cfg

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.model.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            gen_outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature if cfg.generation.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=cfg.generation.save_hidden_states,
                output_attentions=cfg.generation.save_attention_weights and use_attention,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode
        gen_token_ids = gen_outputs.sequences[:, prompt_len:]
        generated_texts = self.tokenizer.batch_decode(gen_token_ids, skip_special_tokens=True)

        # Extract features
        features, top_probs, log_likelihoods = reconstruct_features_from_generation(
            gen_outputs, prompt_len, feature_extractor,
            inputs.attention_mask, self.device,
        )

        return GenerationResult(
            generated_texts=generated_texts,
            generated_token_ids=gen_token_ids,
            features=features,
            top_probs=top_probs,
            log_likelihoods=log_likelihoods,
        )

    def generate_text_only(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate text without feature extraction (for judge.py).

        Args:
            prompts: List of text prompts.
            max_new_tokens: Maximum tokens to generate per prompt.

        Returns:
            List of generated text strings.
        """
        cfg = self.cfg

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.model.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_token_ids = gen_outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(gen_token_ids, skip_special_tokens=True)


def reconstruct_features_from_generation(
    gen_outputs,
    prompt_len: int,
    feature_extractor: CombinedExtractor,
    attention_mask: torch.Tensor,
    device: torch.device,
):
    """Reconstruct features from model.generate() outputs.

    During generation with return_dict_in_generate=True:
    - gen_outputs.hidden_states: tuple of (n_gen_tokens,), each is
      tuple of (n_layers+1,) tensors of shape (batch, 1, hidden_size)
    - gen_outputs.scores: tuple of (n_gen_tokens,) tensors of (batch, vocab_size)
    - gen_outputs.attentions: tuple of (n_gen_tokens,), each is
      tuple of (n_layers,) tensors of shape (batch, n_heads, 1, current_seq_len)
    """
    n_gen_tokens = len(gen_outputs.scores)
    if n_gen_tokens == 0:
        return None, None, None

    # Reconstruct logits from scores: (batch, gen_len, vocab_size)
    logits = torch.stack(gen_outputs.scores, dim=1)

    # Token probs and log-likelihoods for unsupervised baselines (compute in float32, store in bfloat16)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gen_token_ids = gen_outputs.sequences[:, prompt_len:]  # (batch, gen_len)
    log_likelihoods = log_probs.gather(
        dim=-1, index=gen_token_ids.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)  # (batch, gen_len)
    top_probs = torch.softmax(logits.float(), dim=-1).topk(4, dim=-1).values.to(torch.bfloat16)  # (batch, gen_len, 4)

    # Reconstruct hidden states: (n_layers+1,) x (batch, gen_len, hidden_size)
    n_layers = len(gen_outputs.hidden_states[0])
    hidden_states_per_layer = []
    for layer_idx in range(n_layers):
        step0 = gen_outputs.hidden_states[0][layer_idx][:, -1:, :]
        rest = [gen_outputs.hidden_states[t][layer_idx] for t in range(1, n_gen_tokens)]
        layer_hs = torch.cat([step0] + rest, dim=1)  # (batch, gen_len, hidden_size)
        hidden_states_per_layer.append(layer_hs)
    hidden_states_tuple = tuple(hidden_states_per_layer)

    # Reconstruct attentions if available
    attentions_tuple = None
    if gen_outputs.attentions is not None and len(gen_outputs.attentions) > 0:
        n_attn_layers = len(gen_outputs.attentions[0])
        batch_size = gen_outputs.sequences.shape[0]
        gen_seq_len = n_gen_tokens
        max_ctx = prompt_len + gen_seq_len
        n_heads = gen_outputs.attentions[0][0].shape[1]
        layer_attns = []
        for layer_idx in range(n_attn_layers):
            sample_attn = gen_outputs.attentions[0][layer_idx]
            attn_full = torch.zeros(
                batch_size, n_heads, gen_seq_len, max_ctx,
                device=device, dtype=sample_attn.dtype,
            )
            for t in range(gen_seq_len):
                a = gen_outputs.attentions[t][layer_idx]
                ctx_len_t = a.shape[-1]
                if a.dim() == 4:
                    # HF attention may be either (B,H,1,K) or full-seq (B,H,Q,K).
                    # We always take the latest query row as the generated token t.
                    step_attn = a[:, :, -1, :ctx_len_t]
                elif a.dim() == 3:
                    # Some backends already squeeze query dimension: (B,H,K).
                    step_attn = a[:, :, :ctx_len_t]
                else:
                    raise RuntimeError(
                        f"Unexpected attention tensor rank={a.dim()} for layer={layer_idx}, step={t}"
                    )
                attn_full[:, :, t, :ctx_len_t] = step_attn
            layer_attns.append(attn_full)
        attentions_tuple = tuple(layer_attns)

    # Build gen attention mask (all ones for generated tokens)
    gen_attn_mask = torch.ones(
        gen_token_ids.shape, dtype=torch.long, device=device,
    )

    # Run feature extractor
    features = feature_extractor(
        hidden_states=hidden_states_tuple,
        logits=logits,
        attentions=attentions_tuple,
        attention_mask=gen_attn_mask,
    ).to(torch.bfloat16)  # (batch, gen_len, feature_dim) - bfloat16 for memory efficiency

    return features, top_probs, log_likelihoods
