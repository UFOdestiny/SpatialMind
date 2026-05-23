"""Hybrid vLLM + HF generation engine for fast Phase 1.

Strategy:
  1. vLLM generates tokens fast (tensor-parallel, PagedAttention, continuous batching)
  2. HF forward pass on [prompt + generated] extracts hidden states, logits, attentions

CRITICAL: vLLM must be loaded BEFORE HF model because vLLM pre-allocates
GPU memory via its KV cache manager.

NOTE: VLLM_WORKER_MULTIPROC_METHOD=spawn must be set BEFORE `from vllm import LLM`.
vLLM V1 spawns EngineCoreProc via multiprocessing. Linux default is 'fork',
which inherits the parent's CUDA context and causes:
  "RuntimeError: Cannot re-initialize CUDA in forked subprocess."
'spawn' starts a clean process with no inherited CUDA state.
"""

import multiprocessing
import os

# Must be set before vLLM is imported to take effect.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already started

import logging
from typing import List

import torch

from config import Config
from engine.types import GenerationResult
from models.features.combined import CombinedExtractor
from utils.common import load_tokenizer_from_path, resolve_torch_dtype
from utils.efficiency import load_model_with_dtype

log = logging.getLogger(__name__)


class VLLMGenerationEngine:
    """Hybrid vLLM + HF engine for Phase 1 generation.

    vLLM handles fast token generation with PagedAttention, then an HF
    forward pass on [prompt + generated_text] extracts hidden states and
    attention weights for feature extraction.

    The HF forward pass on the concatenated sequence produces **identical**
    hidden states at generated positions due to the causal attention mask.
    """

    def __init__(
        self,
        cfg: Config,
        model_path: str,
        device: torch.device,
        text_only: bool = False,
    ):
        self.cfg = cfg
        self.device = device
        self.model_path = model_path
        self.text_only = bool(text_only)

        # Deferred import keeps process startup light before generation is needed.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Shared tokenizer
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

        attn_backend = os.getenv("VLLM_ATTENTION_BACKEND", cfg.vllm.attention_backend)

        # Step 1: Load vLLM engine FIRST (pre-allocates GPU memory)
        # Note: vLLM v1 auto-selects the attention backend (FLASH_ATTN on sm_90+).
        # cfg.vllm.attention_backend documents the expected backend for reference.
        log.info(
            "Loading vLLM engine from %s (attention_backend=%s)...",
            model_path,
            attn_backend,
        )
        from vllm import LLM, SamplingParams

        self._vllm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            max_model_len=cfg.vllm.max_model_len,
            enforce_eager=cfg.vllm.enforce_eager,
            swap_space=cfg.vllm.swap_space,
            dtype=cfg.vllm.dtype,
            seed=cfg.vllm.seed,
            trust_remote_code=cfg.model.trust_remote_code,
            disable_log_stats=True,
            block_size=32,  # Warining
            attention_backend=attn_backend,
        )
        self._SamplingParams = SamplingParams
        log.info("vLLM engine loaded successfully.")

        # Step 2: Optional HF model load for feature extraction.
        # For text-only workloads (claim extraction / judge), skip this to save
        # startup time and a large amount of GPU memory.
        self._hf_model = None
        if not self.text_only:
            log.info("Loading HF model for feature extraction from %s ...", model_path)
            torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)
            self._hf_model = load_model_with_dtype(
                AutoModelForCausalLM.from_pretrained,
                model_path,
                torch_dtype,
                device_map=cfg.model.device_map,
                trust_remote_code=cfg.model.trust_remote_code,
                cache_dir=cfg.model.hf_cache_dir,
                attn_implementation="eager",
            )
            self._hf_model.eval()
            log.info("HF model loaded for feature extraction.")
        else:
            log.info("Text-only mode enabled: skipping HF feature-extraction model load.")

    def __del__(self):
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    @property
    def model_config(self):
        if self._hf_model is None:
            raise RuntimeError("model_config is unavailable in text_only mode.")
        return self._hf_model.config

    def generate_batch(
        self,
        prompts: List[str],
        feature_extractor: CombinedExtractor,
        max_new_tokens: int,
        use_attention: bool,
    ) -> GenerationResult:
        """Generate tokens with vLLM, extract features with HF forward pass.

        Args:
            prompts: List of text prompts.
            feature_extractor: CombinedExtractor to produce feature vectors.
            max_new_tokens: Maximum tokens to generate per prompt.
            use_attention: Whether to extract attention weights.

        Returns:
            GenerationResult with generated texts, features, token probs,
            and log-likelihoods.
        """
        if self.text_only or self._hf_model is None:
            raise RuntimeError(
                "generate_batch() requires HF feature model; initialize engine with text_only=False."
            )

        # ---- Step 1: vLLM generation (fast) ----
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,  # greedy decoding
            n=1,
        )
        vllm_outputs = self._vllm_engine.generate(
            prompts, sampling_params, use_tqdm=False
        )

        generated_texts = []
        full_texts = []
        for output in vllm_outputs:
            gen_text = output.outputs[0].text
            generated_texts.append(gen_text)
            full_texts.append(output.prompt + gen_text)

        # ---- Step 2: HF forward pass for feature extraction ----
        # Tokenize full sequence [prompt + generated]
        full_inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.model.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        # Tokenize just prompts to determine where generated tokens start
        prompt_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg.model.model_max_length,
            return_tensors="pt",
        )
        # Per-sample prompt length (excluding padding)
        prompt_lens = prompt_inputs.attention_mask.sum(dim=1)  # (batch,)

        with torch.no_grad():
            hf_out = self._hf_model(
                input_ids=full_inputs.input_ids,
                attention_mask=full_inputs.attention_mask,
                output_hidden_states=True,
                output_attentions=use_attention,
                return_dict=True,
            )

        # ---- Step 3: Extract features for generated positions ----
        batch_features = []
        batch_top_probs = []
        batch_log_likelihoods = []

        for i in range(len(prompts)):
            p_len = prompt_lens[i].item()
            total_len = full_inputs.attention_mask[i].sum().item()
            gen_len = total_len - p_len

            if gen_len <= 0:
                batch_features.append(None)
                batch_top_probs.append(None)
                batch_log_likelihoods.append(None)
                continue

            # Slice hidden states for generated positions
            gen_hidden = tuple(
                layer_hs[i : i + 1, p_len:total_len, :]
                for layer_hs in hf_out.hidden_states
            )
            gen_logits = hf_out.logits[i : i + 1, p_len:total_len, :]

            gen_attentions = None
            if use_attention and hf_out.attentions is not None:
                gen_attentions = tuple(
                    layer_attn[i : i + 1, :, p_len:total_len, :total_len]
                    for layer_attn in hf_out.attentions
                )

            gen_mask = torch.ones(1, gen_len, dtype=torch.long, device=self.device)

            features = feature_extractor(
                hidden_states=gen_hidden,
                logits=gen_logits,
                attentions=gen_attentions,
                attention_mask=gen_mask,
            ).to(torch.bfloat16)
            batch_features.append(features)

            # Token probs and log-likelihoods from logits (keep float32 for numerical stability)
            log_probs = torch.log_softmax(gen_logits.float(), dim=-1)
            gen_token_ids = full_inputs.input_ids[i, p_len:total_len]
            ll = (
                log_probs[0]
                .gather(dim=-1, index=gen_token_ids.unsqueeze(-1))
                .squeeze(-1)
                .to(torch.bfloat16)
            )
            tp = torch.softmax(gen_logits[0].float(), dim=-1).topk(4, dim=-1).values.to(torch.bfloat16)
            batch_top_probs.append(tp)
            batch_log_likelihoods.append(ll)

        # ---- Step 4: Pad and stack batch tensors ----
        features_out, top_probs_out, log_ll_out = _pad_batch_results(
            batch_features,
            batch_top_probs,
            batch_log_likelihoods,
            self.device,
        )
        generated_token_ids = _pad_generated_token_ids(
            full_inputs, prompt_lens, self.device
        )

        return GenerationResult(
            generated_texts=generated_texts,
            generated_token_ids=generated_token_ids,
            features=features_out,
            top_probs=top_probs_out,
            log_likelihoods=log_ll_out,
        )

    def generate_text_only(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate text without feature extraction (for judge.py).

        Uses only the vLLM engine — skips the HF forward pass entirely.

        Args:
            prompts: List of text prompts.
            max_new_tokens: Maximum tokens to generate per prompt.

        Returns:
            List of generated text strings.
        """
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            n=1,
        )
        vllm_outputs = self._vllm_engine.generate(
            prompts, sampling_params, use_tqdm=False
        )
        return [output.outputs[0].text for output in vllm_outputs]


def _pad_batch_results(batch_features, batch_top_probs, batch_log_likelihoods, device):
    """Pad variable-length per-sample results into batch tensors."""
    valid = [i for i, f in enumerate(batch_features) if f is not None]
    if not valid:
        return None, None, None

    max_gen_len = max(batch_features[i].shape[1] for i in valid)
    feat_dim = batch_features[valid[0]].shape[-1]
    batch_size = len(batch_features)

    features_out = torch.zeros(batch_size, max_gen_len, feat_dim, device=device)
    top_probs_out = torch.zeros(batch_size, max_gen_len, 4, device=device)
    log_ll_out = torch.zeros(batch_size, max_gen_len, device=device)

    for i in valid:
        g = batch_features[i].shape[1]
        features_out[i, :g, :] = batch_features[i][0]
        if batch_top_probs[i] is not None:
            top_probs_out[i, :g, :] = batch_top_probs[i][:g]
        if batch_log_likelihoods[i] is not None:
            log_ll_out[i, :g] = batch_log_likelihoods[i][:g]

    return features_out, top_probs_out, log_ll_out


def _pad_generated_token_ids(full_inputs, prompt_lens, device):
    batch_size = full_inputs.input_ids.shape[0]
    gen_ids_per_sample = []
    max_len = 0
    for i in range(batch_size):
        p_len = int(prompt_lens[i].item())
        t_len = int(full_inputs.attention_mask[i].sum().item())
        ids = full_inputs.input_ids[i, p_len:t_len]
        gen_ids_per_sample.append(ids)
        max_len = max(max_len, ids.shape[0])
    if max_len == 0:
        return torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    out = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(gen_ids_per_sample):
        out[i, : ids.shape[0]] = ids.to(device)
    return out
