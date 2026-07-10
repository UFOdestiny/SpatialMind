#!/usr/bin/env python3
"""
generate.py - Phase 1: Generate LLM responses, extract features, cache to disk.

For each dataset sample:
  1. Build chat prompt via dataset's build_chat_messages()
  2. Generate tokens via engine backend (vLLM hybrid or pure HF)
  3. Parse final answer via dataset's parse_answer()
  4. Extract/align claims from generated chain-of-thought
  5. Build claim-level labels (`verified`) + sample-level summary label
  6. Extract hidden states, token probs, attention weights
  7. Save claim-level cached chunks to disk

Usage:
    # Single split (default dataset from config)
    python scripts/generate.py --split train --backend vllm

    # Multiple splits with one model load (saves ~5 min on cold start)
    python scripts/generate.py --split train,validation,test --max_samples 100,50,50

    # Explicit dataset
    python scripts/generate.py --dataset stepgame --split test
"""

import sys
import json
import gc
import logging
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.numpy_compat import (
    configure_protobuf_python_implementation,
    patch_numpy_core_multiarray,
    silence_transformers_chat_template_warning,
)
from utils.chain_judge import (
    build_chain_stage1_prompt,
    build_chain_stage2_prompt,
    parse_chain_stage1_output,
    parse_chain_stage2_output,
)

configure_protobuf_python_implementation()
patch_numpy_core_multiarray()

import torch

from config import Config, GLOBAL_SEED
from data.claims import CLAIM_TYPE2ID, PENDING_LABEL, extract_claims_from_generation, extract_claims_from_llm_output
from data.datasets import get_dataset
from models.features.hidden_states import HiddenStateExtractor
from models.features.token_probs import TokenProbExtractor
from models.features.attention import AttentionExtractor
from models.features.combined import CombinedExtractor
from engine import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
silence_transformers_chat_template_warning()
log = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "Verify answer correctness. Return JSON only."
)

JUDGE_USER_TEMPLATE = (
    "Q: {question}\n"
    "Answer: {ground_truth}\n"
    "Response: {generated_text}\n"
    "Claims: {claims_json}\n\n"
    "Return: {{\"answer\":\"correct|incorrect\",\"verified\":[0|1,...]}}\n"
    "verified: 1=correct, 0=hallucinated. Length must match claims."
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Random seed set to %d", seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Generate + cache features")
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split(s) to process. Comma-separated for multi-split: 'train,validation,test'",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g. 'stepgame', 'spartqa'). Default from config.")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["vllm", "hf"],
                        help="Generation backend: 'vllm' (fast, hybrid) or 'hf' (pure HuggingFace)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--claim_extractor_model",
        type=str,
        default=None,
        help="Optional model path for LLM-based claim extraction (decoupled from feature backbone).",
    )
    parser.add_argument(
        "--claim_extractor_backend",
        type=str,
        default=None,
        choices=["vllm", "hf"],
        help="Backend for claim extractor model (default: same as --backend).",
    )
    parser.add_argument(
        "--claim_extractor_max_new_tokens",
        type=int,
        default=1025,
        help="Max new tokens for claim extractor outputs.",
    )
    parser.add_argument(
        "--claim_labeler_max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for claim-labeler outputs (GT-based, non-judge datasets).",
    )
    parser.add_argument(
        "--judge_pending_with_claim_extractor",
        action="store_true",
        help="If set, free-form samples (label=-1) are judged inline using claim extractor model.",
    )
    parser.add_argument(
        "--judge_max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for inline judge responses.",
    )
    parser.add_argument(
        "--defer_claim_extraction",
        action="store_true",
        help="Skip LLM-based claim extraction during generation. Use claim_extract.py to process later.",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--free_form_batch_size",
        type=int,
        default=None,
        help="Override --batch_size for datasets with task_type='free_form' (e.g., bAbI).",
    )
    parser.add_argument(
        "--max_samples", type=str, default="0",
        help="Max samples per split. Single int (applied to all) or comma-sep per split: '100,50,50'",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max new tokens for backbone generation outputs (default from config).",
    )
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--k_hop_values", type=str, default=None,
                        help="Difficulty/k_hop filter. Comma-sep ints, or empty string for all.")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _parse_max_samples(value: str, n_splits: int) -> list:
    """Parse --max_samples into a per-split list.

    Single value 'N' is broadcast to all splits.
    Comma-separated 'N1,N2,...' must match n_splits.
    """
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        return [int(parts[0])] * n_splits
    if len(parts) != n_splits:
        raise ValueError(
            f"--max_samples has {len(parts)} value(s) but {n_splits} split(s) were given"
        )
    return [int(p) for p in parts]


def _parse_field_list(value: str, n_items: int, default_value: str, field_name: str) -> list:
    """Parse comma-separated field values with broadcast support."""
    if value is None:
        return [default_value] * n_items
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        return [parts[0]] * n_items
    if len(parts) != n_items:
        raise ValueError(
            f"{field_name} has {len(parts)} value(s) but {n_items} split(s) were given"
        )
    return parts


def build_feature_extractor(cfg: Config, hidden_size: int, num_layers: int, num_heads: int, device: torch.device):
    """Build the CombinedExtractor from config."""
    hs_layers_str = cfg.features.hidden_state_layers
    if hs_layers_str == "all":
        hs_layer_nums = None
    else:
        hs_layer_nums = [int(x.strip()) for x in hs_layers_str.split(",")]

    extractors = [
        HiddenStateExtractor(layer_nums=hs_layer_nums, hidden_size=hidden_size),
        TokenProbExtractor(top_n=cfg.features.top_n_probs, temperature=cfg.features.temperature),
    ]

    use_attention = bool(cfg.features.attention_layers)
    if use_attention:
        if str(cfg.features.attention_layers).strip().lower() == "all":
            attn_layer_nums = list(range(num_layers))
        else:
            attn_layer_nums = [int(x.strip()) for x in cfg.features.attention_layers.split(",")]
        extractors.append(
            AttentionExtractor(
                layer_nums=attn_layer_nums,
                head_nums=cfg.features.attention_heads,
                attn_history_sz=cfg.features.attn_history_sz,
                pool=cfg.features.pool_attention_layers,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        )

    feature_extractor = CombinedExtractor(extractors).to(device)
    return feature_extractor, use_attention


def _release_cuda_cache(device: torch.device, reason: str) -> None:
    """Release reclaimable CUDA memory after persistent chunk writes."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    collected = gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    log.debug("  CUDA cache cleanup after %s (gc_collected=%d)", reason, collected)


def _build_claim_extractor_prompt(question: str, model_response: str) -> str:
    """Build a strict JSON-only prompt for external claim extraction.

    Leakage-safe design: this prompt never includes ground-truth answers.
    Simplified to produce exactly 3-6 claims without explanations.
    """
    return (
        "Extract 3-6 atomic claims from the response.\n"
        "Return JSON: {\"claims\":[{\"type\":\"reasoning\",\"text\":\"...\"}, ...]}\n"
        "Rules:\n"
        "- 3-6 claims total, one conclusion (last)\n"
        "- No parentheses or explanations in text\n"
        "- Keep claims short and direct\n\n"
        f"Q: {question}\n"
        f"R: {model_response}\n\n"
        "JSON:"
    )


def _build_judge_prompt_with_claims(
    tokenizer,
    question: str,
    ground_truth: str,
    generated_text: str,
    claims_payload,
) -> str:
    claims_json = json.dumps(claims_payload or [], ensure_ascii=False)
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated_text=generated_text,
        claims_json=claims_json,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{JUDGE_SYSTEM_PROMPT}\n\n{user_msg}\nJSON:"


def _enforce_claim_label_consistency(
    claim_dicts,
    verified,
    sample_label: int,
):
    """
    Enforce robust sample/claim label consistency.

    Label convention: 1=correct/non-hallucinated, 0=incorrect/hallucinated.
    
    - Conclusion claims are forced to align with sample label.
    - If no conclusion exists and sample is hallucinated (0), at least one claim is set to 0.
    """
    if not isinstance(claim_dicts, list) or not isinstance(verified, list):
        return verified
    if not verified or len(verified) != len(claim_dicts):
        return verified

    sample_label = int(sample_label)
    if sample_label not in (0, 1):
        # For judge-required samples (label=-1), do not force consistency here.
        return verified

    # sample_label: 1=correct, 0=hallucinated
    target = sample_label
    conclusion_idx = [i for i, c in enumerate(claim_dicts) if str(c.get("claim_type", "")).lower() == "conclusion"]
    for i in conclusion_idx:
        verified[i] = target

    # Ensure there is at least one hallucinated claim for hallucinated samples (target=0).
    if target == 0 and all(int(v) != 0 for v in verified):
        if conclusion_idx:
            verified[conclusion_idx[-1]] = 0
        else:
            verified[-1] = 0

    return verified


def _ensure_reasoning_conclusion_contract(
    claim_dicts,
    verified,
    sample_label: int,
    fallback_conclusion_text: str = "",
):
    """Guarantee at least one conclusion claim and sane labels for reasoning/conclusion only.
    
    Label convention: 1=correct/non-hallucinated, 0=incorrect/hallucinated.
    """
    if not isinstance(claim_dicts, list):
        claim_dicts = []
    if not isinstance(verified, list):
        verified = []

    # Keep only reasoning/conclusion claims.
    filtered_claims = []
    filtered_verified = []
    for i, c in enumerate(claim_dicts):
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("claim_type", "")).strip().lower()
        if ctype not in {"reasoning", "conclusion"}:
            continue
        c = dict(c)
        c["claim_type"] = ctype
        c["claim_type_id"] = CLAIM_TYPE2ID.get(ctype, 2)
        filtered_claims.append(c)
        # Preserve labels including the pending sentinel (-1): reasoning claims are
        # unlabeled until the Stage-2 judge runs. Only 0/1 are final labels; anything
        # else (missing / unparseable) becomes pending so the judge will label it.
        v = verified[i] if i < len(verified) else PENDING_LABEL
        try:
            iv = int(v)
        except Exception:
            iv = PENDING_LABEL
        filtered_verified.append(iv if iv in (0, 1) else PENDING_LABEL)

    claim_dicts = filtered_claims
    verified = filtered_verified

    # Ensure exactly one conclusion at the end.
    conclusion_indices = [
        i for i, c in enumerate(claim_dicts)
        if str(c.get("claim_type", "")).strip().lower() == "conclusion"
    ]
    if conclusion_indices:
        keep = conclusion_indices[-1]
        new_claims = []
        new_verified = []
        for i, c in enumerate(claim_dicts):
            ctype = str(c.get("claim_type", "")).strip().lower()
            if ctype == "conclusion" and i != keep:
                continue
            new_claims.append(c)
            new_verified.append(verified[i])
        claim_dicts, verified = new_claims, new_verified
    else:
        # Synthesize one conclusion when missing.
        fallback_text = (fallback_conclusion_text or "").strip() or "unknown"
        claim_dicts.append(
            {
                "text": fallback_text,
                "claim_type": "conclusion",
                "claim_type_id": CLAIM_TYPE2ID["conclusion"],
                "aligned_token_ids": [0],
            }
        )
        # sample_label: 1=correct, 0=hallucinated
        if int(sample_label) in (0, 1):
            verified.append(int(sample_label))
        else:
            verified.append(1)  # Default to correct if unknown

    # Force conclusion label to sample label where available.
    if int(sample_label) in (0, 1):
        target = int(sample_label)
        for i, c in enumerate(claim_dicts):
            if str(c.get("claim_type", "")).strip().lower() == "conclusion":
                verified[i] = target
                break

    return claim_dicts, verified


def generate_split(
    cfg: Config,
    split: str,
    args,
    engine,
    feature_extractor,
    use_attention,
    max_samples: int,
    dataset_name: str,
    dataset_path: str,
    cache_dir: str,
):
    """Process a single split using a pre-built engine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_batch_size = args.batch_size or cfg.generation.batch_size
    free_form_batch_size = args.free_form_batch_size
    batch_size = configured_batch_size
    max_new_tokens = args.max_new_tokens or cfg.generation.max_new_tokens
    chunk_size = args.chunk_size or cfg.generation.chunk_size
    skip_existing = args.skip_existing and cfg.generation.skip_existing
    backend = args.backend or cfg.generation.backend
    model_path = args.model_path or cfg.model.pretrained_model_name_or_path
    claim_extractor_engine = getattr(args, "_claim_extractor_engine", None)
    claim_extractor_max_new_tokens = int(getattr(args, "claim_extractor_max_new_tokens", 1025) or 1025)
    claim_labeler_max_new_tokens = int(getattr(args, "claim_labeler_max_new_tokens", 512) or 512)
    inline_judge_enabled = bool(getattr(args, "judge_pending_with_claim_extractor", False))
    judge_max_new_tokens = int(getattr(args, "judge_max_new_tokens", 64) or 64)  # Increased from 32 to 64
    feature_dim = feature_extractor.feature_dim()

    split_dir = Path(cache_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Parse difficulty filter
    k_hop_values = None
    if args.k_hop_values is not None:
        if args.k_hop_values.strip():
            k_hop_values = [int(x) for x in args.k_hop_values.split(",")]
        # else: empty string means all difficulties (k_hop_values stays None)

    # Load raw dataset via registry (no tokenization — we handle it during generation)
    ds = get_dataset(
        name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        k_hop_values=k_hop_values,
        max_samples=max_samples,
    )

    task_type = str(getattr(ds, "task_type", "")).strip().lower()
    if task_type == "free_form" and free_form_batch_size is not None:
        if int(free_form_batch_size) > 0:
            batch_size = int(free_form_batch_size)
        else:
            log.warning(
                "Invalid --free_form_batch_size=%s (must be >0); using configured batch_size=%s.",
                free_form_batch_size, configured_batch_size,
            )

    log.info("=" * 70)
    log.info("SpatialMind Phase 1: Generate + Cache Features")
    log.info("=" * 70)
    log.info("Dataset:    %s", dataset_name)
    log.info("Split:      %s", split)
    log.info("Backend:    %s", backend)
    log.info("Batch size: %s", batch_size)
    if task_type == "free_form" and free_form_batch_size is not None:
        log.info(
            "Free-form batch override enabled: configured=%s -> effective=%s",
            configured_batch_size, batch_size,
        )
    log.info("Max new tok:%s", max_new_tokens)
    if claim_extractor_engine is not None:
        log.info("Claim max tok:%s", claim_extractor_max_new_tokens)
        log.info("Claim extractor input: QUESTION + MODEL_RESPONSE (no ground truth).")
        if inline_judge_enabled:
            log.info("Inline judge with claim extractor: enabled (judge max tok=%s).", judge_max_new_tokens)
    log.info("Cache dir:  %s", cache_dir)
    log.info("=" * 70)

    log.info("Generating for split=%s, %d samples, batch_size=%d, chunk_size=%d, backend=%s",
             split, len(ds), batch_size, chunk_size, backend)

    if ds.needs_judge:
        log.info("Dataset '%s' is free-form — samples will need LLM judge (label=-1).", dataset_name)
        if claim_extractor_engine is not None:
            log.info("Claim labeling: disabled in generation (needs_judge dataset).")
    elif claim_extractor_engine is not None:
        log.info(
            "Claim labeling: GT-supervised via extractor model (max tok=%s).",
            claim_labeler_max_new_tokens,
        )

    # Process in chunks
    current_chunk = []
    chunk_idx = 0
    total_correct = 0
    total_processed = 0
    total_pending = 0  # label=-1 (needs judge)
    consistency_overrides = 0
    claim_labeler_supervised_overrides = 0
    inline_judge_processed = 0
    inline_judge_unparseable = 0
    counted_existing_chunks = set()
    total_samples = len(ds)
    last_progress_log = 0

    for start_idx in range(0, len(ds), batch_size):
        end_idx = min(start_idx + batch_size, len(ds))
        
        # Log progress every 10%
        progress_pct = int((start_idx / total_samples) * 100)
        if progress_pct >= last_progress_log + 10:
            log.info("  Progress: %d%% (%d/%d samples)", progress_pct, start_idx, total_samples)
            last_progress_log = progress_pct

        # Check if this batch's chunk already exists (with integrity validation)
        chunk_for_start = start_idx // chunk_size
        chunk_path = split_dir / f"chunk_{chunk_for_start}.pt"
        if skip_existing and chunk_path.exists():
            if chunk_for_start in counted_existing_chunks:
                # Already verified this chunk, skip this batch
                continue
            try:
                existing = torch.load(chunk_path, map_location="cpu", weights_only=False)
                if not isinstance(existing, list) or len(existing) == 0:
                    log.warning("  Chunk %d is invalid (empty or wrong type), will regenerate", chunk_for_start)
                else:
                    total_processed += len(existing)
                    total_correct += sum(1 for s in existing if s.get("label", 1) == 1)
                    total_pending += sum(1 for s in existing if s.get("label", -1) == -1)
                    counted_existing_chunks.add(chunk_for_start)
                    log.info("  Skipping chunk %d (exists, %d samples)", chunk_for_start, len(existing))
                    chunk_idx = max(chunk_idx, chunk_for_start + 1)
                    current_chunk = []
                    continue
            except Exception as e:
                log.warning("  Chunk %d failed to load (%s), will regenerate", chunk_for_start, str(e))

        # Build prompts via dataset interface
        raw_items = [ds.data[i] for i in range(start_idx, min(end_idx, len(ds.data)))]
        prompts = []
        gt_strings = []
        batch_k_hops = []
        batch_questions = []

        for raw_item in raw_items:
            messages = ds.build_chat_messages(raw_item)
            try:
                text = engine.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = ds.build_prompt(raw_item)
            prompts.append(text)
            gt_strings.append(ds.get_ground_truth(raw_item))
            batch_k_hops.append(ds.get_difficulty(raw_item))
            batch_questions.append(ds.get_question(raw_item))

        # Generate + extract features via engine
        result = engine.generate_batch(
            prompts=prompts,
            feature_extractor=feature_extractor,
            max_new_tokens=max_new_tokens,
            use_attention=use_attention,
        )

        claim_extractor_outputs = None
        if claim_extractor_engine is not None:
            claim_prompts = []
            for q, g in zip(batch_questions, result.generated_texts):
                # LUH-style decoupling:
                # use an external model for claim extraction/verification while
                # keeping features from the backbone generation model.
                claim_prompts.append(_build_claim_extractor_prompt(q, g))
            try:
                claim_extractor_outputs = claim_extractor_engine.generate_text_only(
                    claim_prompts,
                    claim_extractor_max_new_tokens,
                )
            except Exception as e:
                log.warning("Claim extractor failed for batch %d-%d (%s); fallback to regex extraction.",
                            start_idx, end_idx, str(e))
                claim_extractor_outputs = None

        # Process each sample in batch (pass 1: extract claims)
        per_sample_rows = []
        per_sample_claims = []
        per_sample_verified = []
        for bi in range(len(prompts)):
            gen_text = result.generated_texts[bi]
            gen_token_ids = []
            gen_len = 0
            try:
                gen_len = len(engine.tokenizer(gen_text, add_special_tokens=False)["input_ids"])
            except Exception:
                gen_len = 0
            if result.generated_token_ids is not None:
                gen_token_ids = result.generated_token_ids[bi].tolist()
                if gen_len > 0:
                    gen_token_ids = gen_token_ids[:gen_len]
            if gen_len <= 0 and result.features is not None:
                gen_len = int(result.features.shape[1])

            predicted_answer = ds.parse_answer(gen_text)
            gt_answer = gt_strings[bi]
            label = ds.check_correctness(predicted_answer, gt_answer)
            answer_correct = (label == 1)
            if claim_extractor_outputs is not None and bi < len(claim_extractor_outputs):
                claims = extract_claims_from_llm_output(
                    llm_output=claim_extractor_outputs[bi],
                    generated_text=gen_text,
                    answer_correct=answer_correct,
                    tokenizer=engine.tokenizer,
                    generated_token_ids=gen_token_ids if gen_token_ids else None,
                )
                extraction_method = "llm"
            else:
                claims = extract_claims_from_generation(
                    generated_text=gen_text,
                    answer_correct=answer_correct,
                    tokenizer=engine.tokenizer,
                    generated_token_ids=gen_token_ids if gen_token_ids else None,
                )
                extraction_method = "regex"
            claim_dicts = [
                {
                    "text": c.text,
                    "claim_type": c.claim_type,
                    "claim_type_id": CLAIM_TYPE2ID.get(c.claim_type, 2),
                    "aligned_token_ids": c.aligned_token_ids,
                    "extracted_by": extraction_method,
                }
                for c in claims
            ]
            verified = [c.verified for c in claims]
            per_sample_rows.append(
                {
                    "gen_text": gen_text,
                    "gen_len": gen_len,
                    "predicted_answer": predicted_answer,
                    "ground_truth": gt_answer,
                    "label": label,
                    "answer_correct": answer_correct,
                    "k_hop": batch_k_hops[bi],
                    "question": batch_questions[bi],
                    "features": (
                        result.features[bi].cpu()
                        if result.features is not None
                        else torch.zeros(1, feature_dim, dtype=torch.bfloat16)
                    ),
                    "token_probs": (
                        result.top_probs[bi].cpu()
                        if result.top_probs is not None
                        else torch.zeros(1, 4, dtype=torch.bfloat16)
                    ),
                    "log_likelihoods": (
                        result.log_likelihoods[bi].cpu()
                        if result.log_likelihoods is not None
                        else torch.zeros(1, dtype=torch.bfloat16)
                    ),
                }
            )
            per_sample_claims.append(claim_dicts)
            per_sample_verified.append(verified)

        # Optional pass 2: two-stage chain judge (all non-judge datasets).
        if claim_extractor_engine is not None and not ds.needs_judge:
            # Stage 1 (strict): answer + conclusion verification.
            stage1_prompts = []
            stage1_indices = []
            stage1_conclusion_pos = {}
            for bi, claim_dicts in enumerate(per_sample_claims):
                if not claim_dicts:
                    continue
                conclusion_pos = -1
                conclusion_text = ""
                for ci, c in enumerate(claim_dicts):
                    if str(c.get("claim_type", "")).strip().lower() == "conclusion":
                        conclusion_pos = ci
                        conclusion_text = str(c.get("text", "")).strip()
                stage1_conclusion_pos[bi] = conclusion_pos
                stage1_prompts.append(
                    build_chain_stage1_prompt(
                        tokenizer=claim_extractor_engine.tokenizer,
                        question=per_sample_rows[bi]["question"],
                        ground_truth=per_sample_rows[bi]["ground_truth"],
                        generated_text=per_sample_rows[bi]["gen_text"],
                        conclusion_claim=conclusion_text,
                    )
                )
                stage1_indices.append(bi)

            stage1_answer_labels = {}
            if stage1_prompts:
                try:
                    stage1_outputs = claim_extractor_engine.generate_text_only(
                        stage1_prompts,
                        claim_labeler_max_new_tokens,
                    )
                    for out_text, bi in zip(stage1_outputs, stage1_indices):
                        st1 = parse_chain_stage1_output(out_text)
                        answer_label = st1.get("answer_label")
                        if answer_label in (0, 1):
                            stage1_answer_labels[bi] = int(answer_label)
                            per_sample_rows[bi]["label"] = int(answer_label)
                            per_sample_rows[bi]["answer_correct"] = bool(int(answer_label) == 1)
                        # Conclusion claim correctness IS the final-answer correctness:
                        # tie it to the resolved answer label, not the judge's independent
                        # semantic conclusion_verified verdict.
                        cpos = stage1_conclusion_pos.get(bi, -1)
                        if cpos >= 0 and cpos < len(per_sample_verified[bi]) and answer_label in (0, 1):
                            per_sample_verified[bi][cpos] = int(answer_label)
                except Exception as e:
                    log.warning(
                        "Stage-1 chain judge failed for batch %d-%d (%s); keeping existing labels.",
                        start_idx, end_idx, str(e),
                    )

            # Stage 2 (chain): reasoning-claim verification.
            stage2_prompts = []
            stage2_indices = []
            stage2_reasoning_pos = {}
            for bi, claim_dicts in enumerate(per_sample_claims):
                if not claim_dicts:
                    continue
                reasoning_texts = []
                reasoning_pos = []
                conclusion_text = ""
                for ci, c in enumerate(claim_dicts):
                    ctype = str(c.get("claim_type", "")).strip().lower()
                    if ctype == "reasoning":
                        reasoning_pos.append(ci)
                        reasoning_texts.append(str(c.get("text", "")))
                    elif ctype == "conclusion":
                        conclusion_text = str(c.get("text", ""))
                if not reasoning_texts:
                    continue
                stage2_reasoning_pos[bi] = reasoning_pos
                stage2_prompts.append(
                    build_chain_stage2_prompt(
                        tokenizer=claim_extractor_engine.tokenizer,
                        question=per_sample_rows[bi]["question"],
                        generated_text=per_sample_rows[bi]["gen_text"],
                        reasoning_claims=reasoning_texts,
                        conclusion_claim=conclusion_text,
                        answer_label=stage1_answer_labels.get(bi, per_sample_rows[bi]["label"]),
                    )
                )
                stage2_indices.append(bi)
            if stage2_prompts:
                try:
                    stage2_outputs = claim_extractor_engine.generate_text_only(
                        stage2_prompts,
                        claim_labeler_max_new_tokens,
                    )
                    for out_text, bi in zip(stage2_outputs, stage2_indices):
                        reasoning_pos = stage2_reasoning_pos.get(bi, [])
                        answer_label = stage1_answer_labels.get(bi, per_sample_rows[bi]["label"])
                        parsed_reasoning, _ = parse_chain_stage2_output(
                            out_text,
                            expected_len=len(reasoning_pos),
                            answer_label=answer_label,
                        )
                        if parsed_reasoning is None:
                            continue
                        for local_i, claim_i in enumerate(reasoning_pos):
                            if 0 <= claim_i < len(per_sample_verified[bi]):
                                per_sample_verified[bi][claim_i] = int(parsed_reasoning[local_i])
                except Exception as e:
                    log.warning(
                        "Stage-2 chain judge failed for batch %d-%d (%s); keeping existing reasoning labels.",
                        start_idx, end_idx, str(e),
                    )

        # Optional pass 2.5: inline judge for free-form datasets using same extractor model.
        if (
            claim_extractor_engine is not None
            and ds.needs_judge
            and inline_judge_enabled
        ):
            # Two-stage chain judge for free-form datasets (batched for speed).
            judge_prompts = []
            judge_indices = []
            for bi, row in enumerate(per_sample_rows):
                if int(row["label"]) == -1:
                    claims_payload = []
                    for ci, c in enumerate(per_sample_claims[bi] or []):
                        claims_payload.append(
                            {
                                "idx": ci,
                                "claim_type": str(c.get("claim_type", "reasoning")),
                                "text": str(c.get("text", "")),
                            }
                        )
                    judge_prompts.append(
                        _build_judge_prompt_with_claims(
                            tokenizer=claim_extractor_engine.tokenizer,
                            question=row["question"],
                            ground_truth=row["ground_truth"],
                            generated_text=row["gen_text"],
                            claims_payload=claims_payload,
                        )
                    )
                    judge_indices.append(bi)
            if judge_prompts:
                try:
                    stage1_outputs = claim_extractor_engine.generate_text_only(
                        judge_prompts,
                        judge_max_new_tokens,
                    )
                    inline_judge_processed += len(judge_indices)

                    stage1_answer_labels = {}
                    stage2_prompts = []
                    stage2_indices = []
                    stage2_reasoning_pos = {}

                    for out_text, bi in zip(stage1_outputs, judge_indices):
                        st1 = parse_chain_stage1_output(out_text)
                        answer_label = st1.get("answer_label")
                        if answer_label in (0, 1):
                            per_sample_rows[bi]["label"] = int(answer_label)
                            per_sample_rows[bi]["answer_correct"] = bool(int(answer_label) == 1)
                            stage1_answer_labels[bi] = int(answer_label)
                        else:
                            inline_judge_unparseable += 1
                            continue

                        # Conclusion claim correctness IS the final-answer correctness:
                        # tie it to the resolved answer label, not the judge's independent
                        # semantic conclusion_verified verdict.
                        for ci, c in enumerate(per_sample_claims[bi] or []):
                            if str(c.get("claim_type", "")).strip().lower() == "conclusion":
                                if ci < len(per_sample_verified[bi]):
                                    per_sample_verified[bi][ci] = int(answer_label)
                                break

                        # Build stage-2 prompts for reasoning chain checks.
                        reasoning_texts = []
                        reasoning_pos = []
                        conclusion_text = ""
                        for ci, c in enumerate(per_sample_claims[bi] or []):
                            ctype = str(c.get("claim_type", "")).strip().lower()
                            if ctype == "reasoning":
                                reasoning_texts.append(str(c.get("text", "")))
                                reasoning_pos.append(ci)
                            elif ctype == "conclusion":
                                conclusion_text = str(c.get("text", ""))
                        if reasoning_texts:
                            stage2_prompts.append(
                                build_chain_stage2_prompt(
                                    tokenizer=claim_extractor_engine.tokenizer,
                                    question=per_sample_rows[bi]["question"],
                                    generated_text=per_sample_rows[bi]["gen_text"],
                                    reasoning_claims=reasoning_texts,
                                    conclusion_claim=conclusion_text,
                                    answer_label=stage1_answer_labels.get(bi),
                                )
                            )
                            stage2_indices.append(bi)
                            stage2_reasoning_pos[bi] = reasoning_pos

                    if stage2_prompts:
                        stage2_outputs = claim_extractor_engine.generate_text_only(
                            stage2_prompts,
                            judge_max_new_tokens,
                        )
                        for out_text, bi in zip(stage2_outputs, stage2_indices):
                            reasoning_pos = stage2_reasoning_pos.get(bi, [])
                            answer_label = stage1_answer_labels.get(bi)
                            parsed_reasoning, _ = parse_chain_stage2_output(
                                out_text,
                                expected_len=len(reasoning_pos),
                                answer_label=answer_label,
                            )
                            if parsed_reasoning is None:
                                continue
                            for ri, ci in enumerate(reasoning_pos):
                                if ci < len(per_sample_verified[bi]):
                                    per_sample_verified[bi][ci] = int(parsed_reasoning[ri])
                except Exception as e:
                    log.warning(
                        "Inline judge failed for batch %d-%d (%s); pending labels remain -1.",
                        start_idx, end_idx, str(e),
                    )

        # Pass 3: consistency + persist
        chunk_saved_in_batch = False
        for bi in range(len(prompts)):
            row = per_sample_rows[bi]
            claim_dicts = per_sample_claims[bi]
            verified = per_sample_verified[bi]

            claim_dicts, verified = _ensure_reasoning_conclusion_contract(
                claim_dicts,
                verified,
                sample_label=row["label"],
                fallback_conclusion_text=row.get("predicted_answer", "") or row.get("gen_text", ""),
            )
            per_sample_claims[bi] = claim_dicts
            per_sample_verified[bi] = verified

            pre_verified = list(verified)
            verified = _enforce_claim_label_consistency(
                claim_dicts,
                verified,
                sample_label=row["label"],
            )
            if verified != pre_verified:
                consistency_overrides += 1

            if row["label"] == 1:
                total_correct += 1
            elif row["label"] == -1:
                total_pending += 1

            attn_mask = torch.zeros(result.features.shape[1], dtype=torch.bfloat16) if result.features is not None else torch.ones(1, dtype=torch.bfloat16)
            if result.features is not None:
                eff_len = max(1, min(int(row["gen_len"]), int(result.features.shape[1])))
                attn_mask[:eff_len] = 1

            sample_dict = {
                "features": row["features"],
                "attention_mask": attn_mask,
                "label": row["label"],
                "answer_correct": row["answer_correct"],
                "k_hop": row["k_hop"],
                "predicted_answer": row["predicted_answer"],
                "ground_truth": row["ground_truth"],
                "question": row["question"],
                "generated_text": row["gen_text"],
                "dataset_name": dataset_name,
                "claims": claim_dicts,
                "verified": verified,
                "token_probs": row["token_probs"],
                "log_likelihoods": row["log_likelihoods"],
            }
            current_chunk.append(sample_dict)
            total_processed += 1

            # Save chunk when full
            if len(current_chunk) >= chunk_size:
                chunk_path = split_dir / f"chunk_{chunk_idx}.pt"
                torch.save(current_chunk, chunk_path)
                
                # Calculate detailed chunk statistics
                labeled = [s for s in current_chunk if s["label"] >= 0]
                incorrect_rate = sum(1 for s in labeled if int(s.get("label", 1)) == 0) / max(len(labeled), 1)
                pending_count = sum(1 for s in current_chunk if s["label"] == -1)
                
                # Claim type statistics
                claim_type_counts = {"reasoning": 0, "conclusion": 0}
                type_label_counter = defaultdict(Counter)
                total_claims = 0
                for s in current_chunk:
                    claims = s.get("claims", [])
                    verified = s.get("verified", [])
                    total_claims += len(claims)
                    for ci, c in enumerate(claims):
                        ctype = c.get("claim_type", "reasoning") if isinstance(c, dict) else "reasoning"
                        claim_type_counts[ctype] = claim_type_counts.get(ctype, 0) + 1
                        label = "invalid"
                        if ci < len(verified):
                            try:
                                iv = int(verified[ci])
                                if iv in (-1, 0, 1):
                                    label = str(iv)
                            except Exception:
                                pass
                        type_label_counter[ctype][label] += 1
                type_label_ratio = {}
                for ctype, counter in type_label_counter.items():
                    denom = float(sum(counter.values()))
                    if denom <= 0:
                        continue
                    type_label_ratio[ctype] = {
                        k: round(float(v) / denom, 4)
                        for k, v in sorted(counter.items(), key=lambda x: x[0])
                    }
                
                avg_claims = total_claims / max(len(current_chunk), 1)
                log.info(
                    "  Chunk %d: %d samples, incorrect_rate=%.2f, pending=%d, avg_claims=%.1f, types=%s, type_label_ratio=%s",
                    chunk_idx, len(current_chunk), incorrect_rate, pending_count, avg_claims,
                    {k: v for k, v in claim_type_counts.items() if v > 0},
                    type_label_ratio,
                )
                
                current_chunk = []
                chunk_idx += 1
                chunk_saved_in_batch = True

        del result, raw_items, prompts, gt_strings, batch_k_hops, batch_questions
        if chunk_saved_in_batch:
            _release_cuda_cache(device, reason=f"chunk {chunk_idx - 1}")

    # Save remaining samples
    if current_chunk:
        chunk_path = split_dir / f"chunk_{chunk_idx}.pt"
        torch.save(current_chunk, chunk_path)
        
        # Final chunk statistics
        labeled = [s for s in current_chunk if s["label"] >= 0]
        incorrect_rate = sum(1 for s in labeled if int(s.get("label", 1)) == 0) / max(len(labeled), 1)
        total_claims = sum(len(s.get("claims", [])) for s in current_chunk)
        type_label_counter = defaultdict(Counter)
        for s in current_chunk:
            claims = s.get("claims", [])
            verified = s.get("verified", [])
            for ci, c in enumerate(claims):
                ctype = c.get("claim_type", "reasoning") if isinstance(c, dict) else "reasoning"
                label = "invalid"
                if ci < len(verified):
                    try:
                        iv = int(verified[ci])
                        if iv in (-1, 0, 1):
                            label = str(iv)
                    except Exception:
                        pass
                type_label_counter[ctype][label] += 1
        type_label_ratio = {}
        for ctype, counter in type_label_counter.items():
            denom = float(sum(counter.values()))
            if denom <= 0:
                continue
            type_label_ratio[ctype] = {
                k: round(float(v) / denom, 4)
                for k, v in sorted(counter.items(), key=lambda x: x[0])
            }
        avg_claims = total_claims / max(len(current_chunk), 1)
        log.info(
            "  Final chunk %d: %d samples, incorrect_rate=%.2f, avg_claims=%.1f, type_label_ratio=%s",
            chunk_idx, len(current_chunk), incorrect_rate, avg_claims, type_label_ratio
        )
        chunk_idx += 1
        _release_cuda_cache(device, reason=f"final chunk {chunk_idx - 1}")

    # Write manifest with chunk sample counts for validation
    hidden_size = engine.model_config.hidden_size
    max_new_tokens_val = args.max_new_tokens or cfg.generation.max_new_tokens
    total_labeled = total_processed - total_pending
    total_incorrect = total_labeled - total_correct
    correct_rate = total_correct / max(total_labeled, 1)
    incorrect_rate = total_incorrect / max(total_labeled, 1)
    
    # Collect chunk sample counts for resume validation, and count pending claims
    # (verified == -1). Reasoning claims are pending after extraction and must be
    # labeled by the Stage-2 judge, so total_claim_pending drives judge scheduling.
    chunk_sample_counts = []
    total_claim_pending = 0
    for ci in range(chunk_idx):
        cp = split_dir / f"chunk_{ci}.pt"
        if cp.exists():
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                chunk_sample_counts.append(len(data))
                for s in data:
                    total_claim_pending += sum(
                        1 for v in s.get("verified", []) if int(v) == -1
                    )
            except Exception:
                chunk_sample_counts.append(-1)  # Invalid chunk
        else:
            chunk_sample_counts.append(0)
    
    manifest = {
        "split": split,
        "dataset_name": dataset_name,
        "granularity": "claim",
        "total_count": total_processed,
        "chunk_size": chunk_size,
        "num_chunks": chunk_idx,
        "chunk_sample_counts": chunk_sample_counts,  # For resume validation
        "feature_dim": feature_dim,
        "model_path": model_path,
        "claim_extractor_model": getattr(args, "claim_extractor_model", None) or "",
        "claim_extractor_max_new_tokens": (
            claim_extractor_max_new_tokens
            if getattr(args, "claim_extractor_model", None)
            else None
        ),
        "claim_extractor_prompt_uses_ground_truth": (
            False if getattr(args, "claim_extractor_model", None) else None
        ),
        "hidden_size": hidden_size,
        "max_new_tokens": max_new_tokens_val,
        "backend": backend,
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "total_pending": total_pending,
        "total_claim_pending": total_claim_pending,
        "claim_label_consistency_overrides": int(consistency_overrides),
        "claim_labeler_supervised_overrides": int(claim_labeler_supervised_overrides),
        "claim_labeler_max_new_tokens": (
            claim_labeler_max_new_tokens
            if (getattr(args, "claim_extractor_model", None) and not ds.needs_judge)
            else None
        ),
        "judge_inline_with_claim_extractor": (
            bool(getattr(args, "claim_extractor_model", None))
            and ds.needs_judge
            and inline_judge_enabled
        ),
        "judge_samples_processed": int(inline_judge_processed) if ds.needs_judge else None,
        "judge_unparseable": int(inline_judge_unparseable) if ds.needs_judge else None,
        "judge_model": (
            (getattr(args, "claim_extractor_model", None) or "")
            if (ds.needs_judge and inline_judge_enabled and claim_extractor_engine is not None)
            else ""
        ),
    }
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "Phase 1 complete: split=%s, total=%d, correct=%d (%.1f%%), incorrect=%d (%.1f%%), pending=%d, claim_overrides=%d, claim_labeler_overrides=%d",
        split, total_processed, total_correct, correct_rate * 100,
        total_incorrect, incorrect_rate * 100, total_pending,
        consistency_overrides, claim_labeler_supervised_overrides,
    )

    if total_pending > 0:
        log.warning(
            "  %d samples have label=-1 (pending judgment). "
            "Run scripts/judge.py to assign labels before Phase 2 training.",
            total_pending,
        )

    return manifest


def main():
    args = parse_args()
    cfg = Config()
    
    # Set global random seed for reproducibility
    set_seed(GLOBAL_SEED)

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    max_samples_list = _parse_max_samples(args.max_samples, len(splits))
    backend = args.backend or cfg.generation.backend
    model_path = args.model_path or cfg.model.pretrained_model_name_or_path
    claim_extractor_model = args.claim_extractor_model
    claim_extractor_backend = args.claim_extractor_backend or backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Deferred claim extraction: skip loading claim_extractor model now,
    # run claim_extract.py separately after generation completes.
    defer_claim_extraction = getattr(args, "defer_claim_extraction", False)
    if defer_claim_extraction and claim_extractor_model:
        log.info("Deferred claim extraction enabled: claim_extractor model will not be loaded.")
        log.info("Run claim_extract.py after generation to process claims.")
        claim_extractor_model = None  # Prevent loading

    dataset_names = _parse_field_list(
        args.dataset,
        len(splits),
        cfg.dataset.dataset_name,
        "--dataset",
    )
    dataset_paths = _parse_field_list(
        args.dataset_path,
        len(splits),
        cfg.dataset.dataset_path,
        "--dataset_path",
    )
    cache_dirs = _parse_field_list(
        args.cache_dir,
        len(splits),
        cfg.generation.cache_dir,
        "--cache_dir",
    )

    # Build engine ONCE — shared across all splits.
    # With vLLM this saves ~5 min of cold-start overhead per additional split.
    log.info("Initializing %s backend from %s", backend.upper(), model_path)
    engine = get_engine(backend=backend, cfg=cfg, model_path=model_path, device=device)
    args._claim_extractor_engine = None
    if claim_extractor_model:
        log.info(
            "Initializing claim extractor backend %s from %s",
            claim_extractor_backend.upper(),
            claim_extractor_model,
        )
        log.info("Claim extractor prompt mode: leakage-safe (no ground truth provided).")
        claim_engine_kwargs = {
            "backend": claim_extractor_backend,
            "cfg": cfg,
            "model_path": claim_extractor_model,
            "device": device,
        }
        if claim_extractor_backend == "vllm":
            claim_engine_kwargs["text_only"] = True
        args._claim_extractor_engine = get_engine(**claim_engine_kwargs)

    feature_extractor, use_attention = build_feature_extractor(
        cfg,
        engine.model_config.hidden_size,
        engine.model_config.num_hidden_layers,
        engine.model_config.num_attention_heads,
        device,
    )
    log.info("Feature dimension: %d (attention=%s)", feature_extractor.feature_dim(), use_attention)

    n_tasks = len(splits)
    for i, (split, max_samples, dataset_name, dataset_path, cache_dir) in enumerate(
        zip(splits, max_samples_list, dataset_names, dataset_paths, cache_dirs),
        start=1,
    ):
        log.info(
            "Task %d/%d: dataset=%s split=%s cache_dir=%s",
            i, n_tasks, dataset_name, split, cache_dir,
        )
        manifest = generate_split(
            cfg=cfg,
            split=split,
            args=args,
            engine=engine,
            feature_extractor=feature_extractor,
            use_attention=use_attention,
            max_samples=max_samples,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
        )
        log.info("Results: %s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
