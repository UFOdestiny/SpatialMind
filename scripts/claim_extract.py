#!/usr/bin/env python3
"""
claim_extract.py - Post-generation claim extraction using LLM.

This script processes cached features from generate.py and extracts claims
using a separate LLM (e.g., Qwen2.5-14B-Instruct). This allows:
1. Running generation with high gpu_memory_utilization (0.85)
2. Running claim extraction separately with high utilization
3. Better throughput than loading both models simultaneously

Usage:
    # Extract claims from cached features
    python scripts/claim_extract.py \
        --cache_dir /path/to/cache \
        --split train,validation,test \
        --claim_extractor_model /path/to/Qwen2.5-14B-Instruct \
        --backend vllm

    # Use HF backend for smaller GPU
    python scripts/claim_extract.py \
        --cache_dir /path/to/cache \
        --split train \
        --claim_extractor_model /path/to/model \
        --backend hf
"""

import sys
import json
import gc
import time
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.numpy_compat import (
    configure_protobuf_python_implementation,
    patch_numpy_core_multiarray,
    silence_transformers_chat_template_warning,
)

configure_protobuf_python_implementation()
patch_numpy_core_multiarray()

import torch

from config import Config, GLOBAL_SEED
from data.claims import CLAIM_TYPE2ID, extract_claims_from_llm_output
from engine import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
silence_transformers_chat_template_warning()
log = logging.getLogger(__name__)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_claim_extractor_prompt(question: str, generated_text: str) -> str:
    """Build a strict JSON-only prompt for LLM-based claim extraction.

    Leakage-safe design: this prompt never includes ground-truth answers.
    Consistent with generate.py prompt format for reliable JSON parsing.
    """
    return (
        "Extract 3-6 atomic claims from the response.\n"
        "Return JSON: {\"claims\":[{\"type\":\"reasoning\",\"text\":\"...\"}, ...]}\n"
        "Rules:\n"
        "- 3-6 claims total, one conclusion (last)\n"
        "- No parentheses or explanations in text\n"
        "- Keep claims short and direct\n\n"
        f"Q: {question}\n"
        f"R: {generated_text}\n\n"
        "JSON:"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Post-generation claim extraction")
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Path to cached features directory (from generate.py)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split(s) to process. Comma-separated: 'train,validation,test'",
    )
    parser.add_argument(
        "--claim_extractor_model", type=str, required=True,
        help="Model path for LLM-based claim extraction.",
    )
    parser.add_argument(
        "--backend", type=str, default="vllm",
        choices=["vllm", "hf"],
        help="Backend for claim extractor model.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=384,
        help="Max new tokens for claim extractor outputs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for claim extraction.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip samples that already have LLM-extracted claims.",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
    )
    return parser.parse_args()


def load_chunk(chunk_path: Path) -> List[dict]:
    """Load a chunk file."""
    return torch.load(chunk_path, map_location="cpu", weights_only=False)


def save_chunk(chunk_path: Path, data: List[dict]):
    """Save a chunk file."""
    torch.save(data, chunk_path)


def _write_manifest(manifest_path: Path, manifest: dict):
    """Atomically write manifest updates to reduce interruption corruption."""
    tmp_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with open(tmp_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_manifest_path.replace(manifest_path)


def _update_claim_manifest(
    manifest: dict,
    manifest_path: Path,
    *,
    model_name: str,
    num_chunks: int,
    chunks_completed: int,
    total_processed: int,
    total_skipped: int,
    total_claims: int,
    status: str,
):
    """Persist claim extraction progress/state into manifest."""
    avg_claims = total_claims / max(total_processed, 1)
    manifest["claim_extractor_applied"] = status == "complete"
    manifest["claim_extractor_model"] = model_name
    manifest["claim_extraction_stats"] = {
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "total_claims": total_claims,
        "avg_claims_per_sample": avg_claims,
        "chunks_completed": chunks_completed,
        "chunks_total": num_chunks,
        "status": status,
    }
    manifest["claim_extraction_progress"] = {
        "status": status,
        "chunks_completed": chunks_completed,
        "chunks_total": num_chunks,
        "updated_unix": int(time.time()),
    }
    _write_manifest(manifest_path, manifest)


def needs_claim_extraction(sample: dict) -> bool:
    """Check if sample needs LLM-based claim extraction."""
    claims = sample.get("claims", [])
    if not claims:
        return True
    # Check if claims were extracted by regex (no LLM marker)
    for c in claims:
        # Handle both dict and SpatialClaim dataclass
        if isinstance(c, dict):
            if c.get("extracted_by") == "llm":
                return False
        elif hasattr(c, "extracted_by") and getattr(c, "extracted_by", None) == "llm":
            return False
    return True


def process_split(
    split_dir: Path,
    engine,
    max_new_tokens: int,
    batch_size: int,
    skip_existing: bool,
) -> dict:
    """Process all chunks in a split directory."""
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        log.warning("Manifest not found: %s", manifest_path)
        return {"error": "manifest_not_found"}

    with open(manifest_path) as f:
        manifest = json.load(f)

    num_chunks = manifest.get("num_chunks", 0)
    if num_chunks == 0:
        log.warning("No chunks in manifest: %s", manifest_path)
        return {"error": "no_chunks"}

    total_processed = 0
    total_skipped = 0
    total_claims = 0
    missing_chunks = 0
    chunks_completed = 0
    model_name = str(engine.model_path) if hasattr(engine, "model_path") else "unknown"

    # Write an explicit running state up-front for interruption-safe resume.
    _update_claim_manifest(
        manifest,
        manifest_path,
        model_name=model_name,
        num_chunks=num_chunks,
        chunks_completed=chunks_completed,
        total_processed=total_processed,
        total_skipped=total_skipped,
        total_claims=total_claims,
        status="running",
    )

    for chunk_idx in range(num_chunks):
        chunk_path = split_dir / f"chunk_{chunk_idx}.pt"
        if not chunk_path.exists():
            log.warning("Chunk not found: %s", chunk_path)
            missing_chunks += 1
            continue

        log.info("Processing chunk %d/%d: %s", chunk_idx + 1, num_chunks, chunk_path)
        chunk_data = load_chunk(chunk_path)
        chunk_modified = False

        # Collect samples needing extraction
        samples_to_process = []
        sample_indices = []
        for i, sample in enumerate(chunk_data):
            if skip_existing and not needs_claim_extraction(sample):
                total_skipped += 1
                continue
            samples_to_process.append(sample)
            sample_indices.append(i)

        if not samples_to_process:
            log.info("  All samples already processed, skipping chunk.")
            chunks_completed += 1
            _update_claim_manifest(
                manifest,
                manifest_path,
                model_name=model_name,
                num_chunks=num_chunks,
                chunks_completed=chunks_completed,
                total_processed=total_processed,
                total_skipped=total_skipped,
                total_claims=total_claims,
                status="running",
            )
            del chunk_data
            gc.collect()
            continue

        log.info("  %d samples need claim extraction", len(samples_to_process))

        # Process in batches
        chunk_had_batch_error = False
        for batch_start in range(0, len(samples_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(samples_to_process))
            batch_samples = samples_to_process[batch_start:batch_end]
            batch_indices = sample_indices[batch_start:batch_end]

            # Build prompts
            prompts = []
            for sample in batch_samples:
                question = sample.get("question", "")
                generated_text = sample.get("generated_text", "")
                prompts.append(_build_claim_extractor_prompt(question, generated_text))

            # Extract claims via LLM
            try:
                outputs = engine.generate_text_only(prompts, max_new_tokens)
            except Exception as e:
                log.error("Claim extraction failed: %s", e)
                chunk_had_batch_error = True
                continue

            # Parse outputs and update samples
            for j, (sample, output, orig_idx) in enumerate(zip(batch_samples, outputs, batch_indices)):
                generated_text = sample.get("generated_text", "")
                answer_correct = sample.get("answer_correct", False)
                
                claims = extract_claims_from_llm_output(
                    llm_output=output,
                    generated_text=generated_text,
                    answer_correct=answer_correct,
                    tokenizer=engine.tokenizer,
                    generated_token_ids=None,
                )

                # Convert claims to dict format for storage (SpatialClaim -> dict)
                claims_dict = []
                for c in claims:
                    if hasattr(c, '__dict__'):
                        # SpatialClaim dataclass
                        ctype = str(c.claim_type or "").strip().lower()
                        claim_d = {
                            "text": c.text,
                            "claim_type": ctype,
                            "claim_type_id": CLAIM_TYPE2ID.get(ctype, 2),
                            "aligned_token_ids": c.aligned_token_ids,
                            "verified": c.verified,
                            "extracted_by": "llm",
                        }
                    elif isinstance(c, dict):
                        claim_d = dict(c)
                        ctype = str(claim_d.get("claim_type", "")).strip().lower()
                        claim_d["claim_type"] = ctype
                        claim_d["claim_type_id"] = CLAIM_TYPE2ID.get(ctype, 2)
                        claim_d["extracted_by"] = "llm"
                    else:
                        claim_d = {
                            "text": str(c),
                            "claim_type": "reasoning",
                            "claim_type_id": CLAIM_TYPE2ID.get("reasoning", 1),
                            "extracted_by": "llm",
                        }
                    claims_dict.append(claim_d)

                # Update sample in chunk_data
                chunk_data[orig_idx]["claims"] = claims_dict
                chunk_data[orig_idx]["verified"] = [-1] * len(claims_dict)  # Pending verification (-1)
                total_processed += 1
                total_claims += len(claims_dict)
                chunk_modified = True

        # Save modified chunk with statistics
        if chunk_modified:
            save_chunk(chunk_path, chunk_data)
            # Calculate chunk statistics
            chunk_claims = [len(s.get("claims", [])) for s in chunk_data]
            claim_types = {}
            type_label_counter = defaultdict(Counter)
            for s in chunk_data:
                claims = s.get("claims", [])
                verified = s.get("verified", [])
                for ci, c in enumerate(claims):
                    ctype = c.get("claim_type", "unknown") if isinstance(c, dict) else "unknown"
                    claim_types[ctype] = claim_types.get(ctype, 0) + 1
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
            avg_claims = sum(chunk_claims) / max(len(chunk_claims), 1)
            log.info(
                "  Saved chunk %d: %d samples, avg_claims=%.1f, types=%s, type_label_ratio=%s",
                chunk_idx, len(chunk_data), avg_claims, claim_types, type_label_ratio
            )

        # Release memory
        remaining_after_chunk = sum(1 for s in chunk_data if needs_claim_extraction(s))
        if remaining_after_chunk == 0:
            chunks_completed += 1
        else:
            log.warning(
                "  Chunk %d still has %d sample(s) requiring extraction; will resume later.",
                chunk_idx,
                remaining_after_chunk,
            )
            if chunk_had_batch_error:
                log.warning("  Chunk %d had batch errors during extraction.", chunk_idx)

        del chunk_data
        gc.collect()

        _update_claim_manifest(
            manifest,
            manifest_path,
            model_name=model_name,
            num_chunks=num_chunks,
            chunks_completed=chunks_completed,
            total_processed=total_processed,
            total_skipped=total_skipped,
            total_claims=total_claims,
            status="running",
        )

    final_status = (
        "complete"
        if (missing_chunks == 0 and chunks_completed == num_chunks)
        else "incomplete"
    )
    _update_claim_manifest(
        manifest,
        manifest_path,
        model_name=model_name,
        num_chunks=num_chunks,
        chunks_completed=chunks_completed,
        total_processed=total_processed,
        total_skipped=total_skipped,
        total_claims=total_claims,
        status=final_status,
    )

    return {
        "processed": total_processed,
        "skipped": total_skipped,
        "claims": total_claims,
        "missing_chunks": missing_chunks,
        "status": final_status,
    }


def main():
    args = parse_args()
    cfg = Config()
    set_seed(GLOBAL_SEED)

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    cache_dir = Path(args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cache_dir.exists():
        log.error("Cache directory not found: %s", cache_dir)
        sys.exit(1)

    log.info("Initializing claim extractor: %s (%s)", args.claim_extractor_model, args.backend)
    engine_kwargs = {
        "backend": args.backend,
        "cfg": cfg,
        "model_path": args.claim_extractor_model,
        "device": device,
    }
    if args.backend == "vllm":
        engine_kwargs["text_only"] = True
    
    engine = get_engine(**engine_kwargs)
    log.info("Claim extractor ready")

    results = {}
    has_failure = False
    for split in splits:
        split_dir = cache_dir / split
        if not split_dir.exists():
            log.warning("Split directory not found: %s", split_dir)
            results[split] = {"error": "not_found"}
            has_failure = True
            continue

        log.info("Processing split: %s", split)
        result = process_split(
            split_dir,
            engine,
            args.max_new_tokens,
            args.batch_size,
            args.skip_existing,
        )
        results[split] = result
        if result.get("error") or result.get("status") != "complete":
            has_failure = True
        log.info("Split %s: processed=%d, skipped=%d, claims=%d",
                 split, result.get("processed", 0), result.get("skipped", 0), result.get("claims", 0))

    log.info("Claim extraction complete")
    for split, result in results.items():
        log.info("  %s: %s", split, result)

    if has_failure:
        log.error("Claim extraction finished with incomplete results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
