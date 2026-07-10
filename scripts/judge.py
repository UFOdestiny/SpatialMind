#!/usr/bin/env python3
"""
judge.py - Phase 1.5: LLM-as-Judge for free-form answer evaluation.

For datasets where check_correctness() returns -1 (e.g. SpartQA), this script
uses a judge LLM to evaluate whether the generated answer is correct.

Flow:
  1. Load chunk files from {cache_dir}/{split}/
  2. Find samples with label == -1 (pending judgment)
  3. Build judge prompts with question, ground truth, and generated answer
  4. Generate judge responses via engine.generate_text_only()
  5. Parse: "correct" -> 1, "incorrect" -> 0
  6. Update label field in chunk data, save back to disk
  7. Update manifest with judge statistics

Usage:
    python scripts/judge.py --cache_dir /path/to/cached_features --split test \\
        --judge_model /path/to/judge_model --judge_backend vllm

    # Force re-judge all samples (overwrite existing labels)
    python scripts/judge.py --cache_dir /path/to/cached_features --split test \\
        --judge_model /path/to/judge_model --force
"""

import sys
import json
import logging
import argparse
from collections import Counter, defaultdict
from pathlib import Path

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
from config import Config
from engine import get_engine
from utils.chain_judge import (
    build_chain_stage1_prompt,
    build_chain_stage2_prompt,
    build_chain_stage2_schema,
    parse_chain_stage1_output,
    parse_chain_stage2_output,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
silence_transformers_chat_template_warning()
log = logging.getLogger(__name__)


def _compute_claim_type_label_ratio(samples):
    """Compute per-claim-type label ratios from claim verified labels."""
    type_label_counter = defaultdict(Counter)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        claims = sample.get("claims", []) or []
        verified = sample.get("verified", []) or []
        if not isinstance(claims, list):
            continue
        if not isinstance(verified, list):
            verified = []
        for ci, claim in enumerate(claims):
            ctype = "unknown"
            if isinstance(claim, dict):
                ctype = str(claim.get("claim_type", "unknown")).strip().lower() or "unknown"
            label = "invalid"
            if ci < len(verified):
                try:
                    iv = int(verified[ci])
                    if iv in (-1, 0, 1):
                        label = str(iv)
                except Exception:
                    pass
            type_label_counter[ctype][label] += 1

    ratio = {}
    for ctype, counter in type_label_counter.items():
        denom = float(sum(counter.values()))
        if denom <= 0:
            continue
        ordered = {}
        for key in ("-1", "0", "1", "invalid"):
            if key in counter:
                ordered[key] = round(float(counter[key]) / denom, 6)
        for key, count in counter.items():
            if key not in ordered:
                ordered[key] = round(float(count) / denom, 6)
        ratio[ctype] = ordered
    return ratio

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1.5: LLM-as-Judge")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Cache directory/directories containing chunk files (comma-separated supported)")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to judge (default: test)")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Path to judge model (overrides config)")
    parser.add_argument("--judge_backend", type=str, default=None,
                        choices=["vllm", "hf"],
                        help="Judge backend (overrides config)")
    parser.add_argument("--judge_batch_size", type=int, default=None,
                        help="Batch size for judge generation (overrides config)")
    parser.add_argument("--judge_max_new_tokens", type=int, default=None,
                        help="Max new tokens for judge (overrides config)")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge all samples, not just label=-1")
    return parser.parse_args()



def main():
    args = parse_args()
    cfg = Config()

    judge_model = args.judge_model or cfg.judge.judge_model_path
    judge_backend = args.judge_backend or cfg.judge.judge_backend
    judge_batch_size = args.judge_batch_size or cfg.judge.judge_batch_size
    judge_max_new_tokens = args.judge_max_new_tokens or cfg.judge.judge_max_new_tokens

    # Support comma-separated cache dirs and splits
    cache_dirs = [c.strip() for c in args.cache_dir.split(",") if c.strip()]
    splits = [s.strip() for s in args.split.split(",")]
    pending_targets = []

    for cache_dir in cache_dirs:
        for split in splits:
            split_dir = Path(cache_dir) / split
            if not split_dir.exists():
                log.info("Split directory not found: %s — skipping.", split_dir)
                continue
            # Do not trust manifest-only pending counts here: they can be stale
            # after intermediate scripts mutate labels. _judge_split will scan
            # chunk data and skip quickly if there are truly no pending labels.
            pending_targets.append((cache_dir, split, split_dir))

    if not pending_targets:
        log.info("No cache/split target needs judgment. Done.")
        return

    if not judge_model:
        log.error("Samples need judgment but no judge model specified. "
                  "Use --judge_model or set judge.judge_model_path in config.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading judge engine once (%s) from %s", judge_backend, judge_model)
    engine_kwargs = {
        "backend": judge_backend,
        "cfg": cfg,
        "model_path": judge_model,
        "device": device,
    }
    if judge_backend == "vllm":
        engine_kwargs["text_only"] = True
    engine = get_engine(**engine_kwargs)

    for cache_dir, split, split_dir in pending_targets:
        _judge_split(
            cache_dir, split, split_dir, args, engine, judge_model,
            judge_batch_size, judge_max_new_tokens,
        )


def _judge_split(cache_dir, split, split_dir, args, engine, judge_model,
                 judge_batch_size, judge_max_new_tokens):
    """Run judge on a single split. Memory-efficient: processes one chunk at a time.

    The conclusion claim label is always tied to the resolved sample label (the
    final-answer correctness), never the judge's independent semantic verdict —
    see the conclusion-labeling block below. Reasoning claims are still verified
    by the judge (Stage 2), since intermediate spatial steps have no deterministic
    label.
    """
    chunk_files = sorted(split_dir.glob("chunk_*.pt"))
    if not chunk_files:
        log.info("No chunk files found in %s — skipping.", split_dir)
        return

    # Phase 1: Scan chunks to find pending samples (lightweight scan)
    log.info("Scanning %d chunks for pending samples...", len(chunk_files))
    pending_by_chunk = {}  # chunk_idx -> list of local_idx
    total_samples = 0
    # Track label distribution by claim type: {type: {label: count}}
    label_stats_by_type = {}

    for chunk_idx, chunk_path in enumerate(chunk_files):
        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        total_samples += len(chunk_data)
        pending_indices = []
        for local_idx, sample in enumerate(chunk_data):
            sample_label = sample.get("label", -1)
            verified = sample.get("verified", [])
            claims = sample.get("claims", []) or []
            claim_types = sample.get("claim_types", []) or []
            # Count label stats per claim type
            for i, ctype in enumerate(claim_types):
                if ctype not in label_stats_by_type:
                    label_stats_by_type[ctype] = {"0": 0, "1": 0, "-1": 0}
                v = verified[i] if i < len(verified) else -1
                try:
                    v_int = int(v)
                except Exception:
                    v_int = -1
                label_stats_by_type[ctype][str(v_int)] = label_stats_by_type[ctype].get(str(v_int), 0) + 1
            has_pending_claim = False
            if isinstance(verified, list):
                for v in verified:
                    try:
                        if int(v) == -1:
                            has_pending_claim = True
                            break
                    except Exception:
                        continue
            if args.force or sample_label == -1 or has_pending_claim:
                pending_indices.append(local_idx)
        if pending_indices:
            pending_by_chunk[chunk_idx] = pending_indices
        del chunk_data  # Free memory immediately

    total_pending = sum(len(v) for v in pending_by_chunk.values())
    log.info("Split '%s': %d total samples, %d need judgment across %d chunks.",
             split, total_samples, total_pending, len(pending_by_chunk))

    # Log label distribution by claim type
    if label_stats_by_type:
        log.info("  Label distribution by claim type:")
        for ctype in sorted(label_stats_by_type.keys()):
            stats = label_stats_by_type[ctype]
            total_claims = sum(stats.values())
            if total_claims > 0:
                pct_0 = stats["0"] / total_claims * 100
                pct_1 = stats["1"] / total_claims * 100
                pct_pending = stats["-1"] / total_claims * 100
                log.info("    %s: total=%d, correct=%.1f%%, incorrect=%.1f%%, pending=%.1f%%",
                         ctype, total_claims, pct_1, pct_0, pct_pending)

    if not pending_by_chunk:
        # Still refresh manifest stats so legacy manifests gain total_claim_pending.
        log.info("  No samples need judgment. Refreshing manifest statistics.")

    log.info("=" * 70)
    log.info("SpatialMind Phase 1.5: LLM-as-Judge (streaming mode)")
    log.info("=" * 70)
    log.info("Cache dir:    %s", cache_dir)
    log.info("Split dir:    %s", split_dir)
    log.info("Judge model:  %s", judge_model)
    log.info("Batch size:   %s", judge_batch_size)
    log.info("Force:        %s", args.force)
    log.info("=" * 70)

    # Phase 2: Process each chunk individually
    total_correct = 0
    total_incorrect = 0
    total_unparseable = 0
    total_processed = 0

    for chunk_idx in sorted(pending_by_chunk.keys()):
        chunk_path = chunk_files[chunk_idx]
        pending_indices = pending_by_chunk[chunk_idx]
        log.info("Processing chunk %d/%d (%d pending samples)...",
                 chunk_idx + 1, len(chunk_files), len(pending_indices))

        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        pending_samples = [(chunk_idx, idx, chunk_data[idx]) for idx in pending_indices]

        # Process this chunk's samples in batches
        for batch_start in range(0, len(pending_samples), judge_batch_size):
            batch = pending_samples[batch_start:batch_start + judge_batch_size]
            stage1_answer_label = {}

            # ---------------- Stage 1: strict answer/conclusion ----------------
            stage1_prompts = []
            stage1_meta = []
            for ci, local_idx, sample in batch:
                question = sample.get("question", "")
                ground_truth = sample.get("ground_truth", sample.get("ground_truth_dir", ""))
                generated_text = sample.get("generated_text", "")
                claims = sample.get("claims", []) or []
                conclusion_pos = -1
                conclusion_text = ""
                for c_idx, c in enumerate(claims):
                    if not isinstance(c, dict):
                        continue
                    if str(c.get("claim_type", "")).strip().lower() == "conclusion":
                        conclusion_pos = c_idx
                        conclusion_text = str(c.get("text", "")).strip()

                existing_label = sample.get("label", -1)
                try:
                    existing_label = int(existing_label)
                except Exception:
                    existing_label = -1
                preserve_sample_label = (not args.force) and existing_label in (0, 1)

                stage1_prompts.append(
                    build_chain_stage1_prompt(
                        tokenizer=engine.tokenizer,
                        question=question,
                        ground_truth=ground_truth,
                        generated_text=generated_text,
                        conclusion_claim=conclusion_text,
                    )
                )
                stage1_meta.append((ci, local_idx, conclusion_pos, existing_label, preserve_sample_label))

            if stage1_prompts:
                stage1_outputs = engine.generate_text_only(stage1_prompts, judge_max_new_tokens)

                for response, (ci, local_idx, conclusion_pos, existing_label, preserve_sample_label) in zip(stage1_outputs, stage1_meta):
                    parsed = parse_chain_stage1_output(response)
                    answer_label = parsed.get("answer_label")
                    key = (ci, local_idx)
                    sample = chunk_data[local_idx]

                    if preserve_sample_label:
                        sample["label"] = existing_label
                        sample["answer_correct"] = bool(existing_label == 1)
                        stage1_answer_label[key] = int(answer_label) if answer_label in (0, 1) else existing_label
                    else:
                        stage1_answer_label[key] = answer_label
                        if answer_label in (0, 1):
                            sample["label"] = int(answer_label)
                            sample["answer_correct"] = bool(int(answer_label) == 1)
                            if int(answer_label) == 1:
                                total_correct += 1
                            else:
                                total_incorrect += 1
                        else:
                            sample["label"] = -1
                            sample["answer_correct"] = False
                            total_unparseable += 1
                            log.warning(
                                "  Stage-1 unparseable for chunk=%d, idx=%d: %r",
                                ci, local_idx, response[:120],
                            )

                    claims = sample.get("claims", []) or []
                    if claims:
                        verified = sample.get("verified", [])
                        if not isinstance(verified, list) or len(verified) != len(claims):
                            verified = [-1] * len(claims)
                        if conclusion_pos >= 0 and conclusion_pos < len(verified):
                            # The conclusion claim asserts "the answer is X"; its
                            # correctness IS the final-answer correctness. Tie it to the
                            # resolved sample label for BOTH task types:
                            #   * classification: sample["label"] = deterministic
                            #     parse_answer + check_correctness (exact match).
                            #   * free-form:     sample["label"] = judge answer_label
                            #     (generated answer aligned with the true sample label).
                            # Never use the judge's independent semantic conclusion_verified,
                            # which can decouple the conclusion label from ground truth.
                            resolved = int(sample.get("label", -1))
                            if resolved in (0, 1):
                                verified[conclusion_pos] = resolved
                        sample["verified"] = verified

            # ---------------- Stage 2: reasoning chain consistency ----------------
            stage2_prompts = []
            stage2_meta = []
            for ci, local_idx, sample in batch:
                sample = chunk_data[local_idx]  # Re-fetch updated sample
                claims = sample.get("claims", []) or []
                # NOTE:
                # `batch` already contains samples selected as needing judgment
                # (label == -1 and/or pending claim labels before stage-1).
                # Do not re-gate stage-2 on current `verified` values here:
                # stage-1 may remove "-1" sentinels (e.g., by fixing conclusion),
                # which would incorrectly skip reasoning verification for BaBi.
                reasoning_positions = []
                reasoning_texts = []
                conclusion_text = ""
                for c_idx, c in enumerate(claims):
                    if not isinstance(c, dict):
                        continue
                    ctype = str(c.get("claim_type", "")).strip().lower()
                    if ctype == "reasoning":
                        reasoning_positions.append(c_idx)
                        reasoning_texts.append(str(c.get("text", "")))
                    elif ctype == "conclusion":
                        conclusion_text = str(c.get("text", ""))
                if not reasoning_texts:
                    continue
                key = (ci, local_idx)
                answer_label = stage1_answer_label.get(key)
                stage2_prompts.append(
                    build_chain_stage2_prompt(
                        tokenizer=engine.tokenizer,
                        question=sample.get("question", ""),
                        generated_text=sample.get("generated_text", ""),
                        reasoning_claims=reasoning_texts,
                        conclusion_claim=conclusion_text,
                        answer_label=answer_label,
                    )
                )
                stage2_meta.append((ci, local_idx, reasoning_positions))

            if stage2_prompts:
                # Guided decoding: constrain each output to a JSON schema with the
                # exact reasoning-claim count (values 0/1). Prompts have different
                # claim counts, so group by count and decode each group with its
                # schema, then scatter results back to the original order.
                stage2_outputs = [None] * len(stage2_prompts)
                by_count = {}
                for idx, (_, _, r_pos) in enumerate(stage2_meta):
                    by_count.setdefault(len(r_pos), []).append(idx)
                for count, idxs in by_count.items():
                    schema = build_chain_stage2_schema(count) if count > 0 else None
                    group_out = engine.generate_text_only(
                        [stage2_prompts[i] for i in idxs],
                        judge_max_new_tokens,
                        structured_json_schema=schema,
                    )
                    for j, i in enumerate(idxs):
                        stage2_outputs[i] = group_out[j]
                stage2_parse_stats = {"exact": 0, "padded": 0, "truncated": 0, "fallback": 0, "inferred": 0, "failed": 0}
                for response, (ci, local_idx, reasoning_positions) in zip(stage2_outputs, stage2_meta):
                    key = (ci, local_idx)
                    answer_label = stage1_answer_label.get(key)
                    parsed_reasoning, parse_status = parse_chain_stage2_output(
                        response,
                        expected_len=len(reasoning_positions),
                        answer_label=answer_label,
                        log_failures=True,
                    )
                    stage2_parse_stats[parse_status] = stage2_parse_stats.get(parse_status, 0) + 1
                    if parsed_reasoning is None:
                        continue
                    sample = chunk_data[local_idx]
                    claims = sample.get("claims", []) or []
                    if not claims:
                        continue
                    verified = sample.get("verified", [])
                    if not isinstance(verified, list) or len(verified) != len(claims):
                        verified = [-1] * len(claims)
                    for ri, claim_pos in enumerate(reasoning_positions):
                        if 0 <= claim_pos < len(verified):
                            verified[claim_pos] = int(parsed_reasoning[ri])
                    sample["verified"] = verified

                if sum(stage2_parse_stats.values()) > 0:
                    log.info("  Stage-2 parse stats: %s", stage2_parse_stats)

            total_processed += len(batch)
            log.info(
                "  Progress: %d/%d samples (correct=%d, incorrect=%d, unparseable=%d)",
                total_processed, total_pending, total_correct, total_incorrect, total_unparseable,
            )

        # Save this chunk immediately after processing
        chunk_claim_type_label_ratio = _compute_claim_type_label_ratio(chunk_data)
        log.info("  Chunk %d claim_type_label_ratio=%s", chunk_idx, chunk_claim_type_label_ratio)
        torch.save(chunk_data, chunk_path)
        log.info("  Saved updated chunk: %s", chunk_path.name)
        del chunk_data  # Free memory before loading next chunk

    # Phase 3: Update manifest (re-scan chunks for final stats)
    log.info("Updating manifest with final statistics...")
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    all_labels = []
    unparseable_samples = []
    total_claim_pending = 0
    # Use streaming counter instead of accumulating all samples
    type_label_counter = defaultdict(Counter)

    for chunk_idx, chunk_path in enumerate(chunk_files):
        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for sample_idx, sample in enumerate(chunk_data):
            label = sample.get("label", 1)
            all_labels.append(label)
            if label == -1:
                unparseable_samples.append({"chunk": chunk_idx, "idx": sample_idx})
            verified = sample.get("verified", [])
            if isinstance(verified, list):
                for v in verified:
                    try:
                        if int(v) == -1:
                            total_claim_pending += 1
                    except Exception:
                        continue
            # Accumulate claim type/label stats (streaming, no sample storage)
            claims = sample.get("claims", []) or []
            if isinstance(claims, list) and isinstance(verified, list):
                for ci, claim in enumerate(claims):
                    ctype = "unknown"
                    if isinstance(claim, dict):
                        ctype = str(claim.get("claim_type", "unknown")).strip().lower() or "unknown"
                    lbl = "invalid"
                    if ci < len(verified):
                        try:
                            iv = int(verified[ci])
                            if iv in (-1, 0, 1):
                                lbl = str(iv)
                        except Exception:
                            pass
                    type_label_counter[ctype][lbl] += 1
        del chunk_data

    total_count = len(all_labels)
    n_correct = sum(1 for l in all_labels if l == 1)
    n_incorrect = sum(1 for l in all_labels if l == 0)
    n_pending = sum(1 for l in all_labels if l == -1)
    total_labeled = total_count - n_pending
    correct_rate = n_correct / max(total_labeled, 1)
    incorrect_rate = n_incorrect / max(total_labeled, 1)

    # Compute ratio from streaming counter
    overall_claim_type_label_ratio = {}
    for ctype, counter in type_label_counter.items():
        denom = float(sum(counter.values()))
        if denom <= 0:
            continue
        ordered = {}
        for key in ("-1", "0", "1", "invalid"):
            if key in counter:
                ordered[key] = round(float(counter[key]) / denom, 6)
        for key, count in counter.items():
            if key not in ordered:
                ordered[key] = round(float(count) / denom, 6)
        overall_claim_type_label_ratio[ctype] = ordered

    manifest.update({
        "total_count": total_count,
        "total_correct": n_correct,
        "total_incorrect": n_incorrect,
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "total_pending": n_pending,
        "judge_model": judge_model,
        "judge_samples_processed": total_pending,
        "judge_unparseable": total_unparseable,
        "total_claim_pending": total_claim_pending,
        "claim_type_label_ratio": overall_claim_type_label_ratio,
        "judge_unparseable_samples": unparseable_samples[:100],
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 70)
    log.info(
        "Judge complete: total=%d, correct=%d (%.1f%%), incorrect=%d (%.1f%%), pending=%d, claim_pending=%d, unparseable=%d",
        total_count, n_correct, correct_rate * 100, n_incorrect, incorrect_rate * 100,
        n_pending, total_claim_pending, total_unparseable
    )
    log.info("Claim type label ratio: %s", overall_claim_type_label_ratio)
    log.info("Updated manifest: %s", manifest_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
