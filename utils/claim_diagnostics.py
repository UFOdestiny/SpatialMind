#!/usr/bin/env python3
"""
Claim extraction / labeling diagnostics for cached feature data.

Reads cached chunks under:
  <cache_dir>/<split>/chunk_*.pt

Outputs:
  - Human-readable report text
  - Machine-readable JSON summary
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch


KNOWN_CLAIM_TYPES = {"reasoning", "conclusion"}
KNOWN_TYPE_IDS = {1, 2}
GENERIC_CLAIM_PATTERNS = [
    "reasoning:",
    "conclusion:",
    "reasoning using",
    "merge intermediate reasoning",
]


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_int_or_none(v: Any):
    try:
        return int(v)
    except Exception:
        return None


def _attention_len(sample: Dict[str, Any]) -> int:
    am = sample.get("attention_mask")
    if am is None:
        return 0
    try:
        if hasattr(am, "sum"):
            return int(am.sum().item())
        return int(sum(int(x) for x in am))
    except Exception:
        return 0


def _short_text(x: str, n: int = 180) -> str:
    s = (x or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _quantile_int(sorted_vals: List[int], q: float) -> int:
    if not sorted_vals:
        return 0
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(sorted_vals) - 1) * q))
    return int(sorted_vals[idx])


def _has_explicit_conclusion_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return ("conclusion:" in t) or ("the answer is" in t) or ("final answer" in t)


def _is_generic_claim_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(p in t for p in GENERIC_CLAIM_PATTERNS)


def _sample_has_conclusion_claim(claims: Any) -> bool:
    if not isinstance(claims, list):
        return False
    for c in claims:
        if isinstance(c, dict) and str(c.get("claim_type", "")).strip().lower() == "conclusion":
            return True
    return False


def _extract_conclusion_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    lower = t.lower()
    marker = "conclusion:"
    idx = lower.rfind(marker)
    if idx >= 0:
        return t[idx + len(marker):].strip()
    m = re.search(r"(?:the answer is|final answer\s*[:\-]?)\s*(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def parse_args():
    p = argparse.ArgumentParser(description="Detailed claim extraction diagnostics")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--splits", type=str, default="train,validation,test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_bad_examples", type=int, default=50)
    p.add_argument("--examples_per_khop", type=int, default=3)
    return p.parse_args()


def _sample_preview(sample: Dict[str, Any], max_claims: int = 6) -> Dict[str, Any]:
    claims = sample.get("claims", []) or []
    verified = sample.get("verified", []) or []
    claim_previews = []
    for i, c in enumerate(claims[:max_claims]):
        if not isinstance(c, dict):
            continue
        v = verified[i] if i < len(verified) else None
        claim_previews.append(
            {
                "idx": i,
                "claim_type": c.get("claim_type"),
                "claim_text": _short_text(str(c.get("text", "")), n=140),
                "claim_label": _to_int_or_none(v),
            }
        )
    return {
        "k_hop": sample.get("k_hop", sample.get("task", "NA")),
        "input_question": _short_text(sample.get("question", ""), n=220),
        "output_generated": _short_text(sample.get("generated_text", ""), n=260),
        "generation_len": _attention_len(sample),
        "ground_truth": _short_text(sample.get("ground_truth", ""), n=120),
        "predicted_answer": _short_text(sample.get("predicted_answer", ""), n=120),
        "sample_label": _to_int_or_none(sample.get("label")),
        "answer_correct": bool(sample.get("answer_correct", False)),
        "num_claims": len(claims),
        "num_verified": len(verified),
        "claims_preview": claim_previews,
    }


def analyze_split(split_dir: Path, max_bad_examples: int, examples_per_khop: int) -> Dict[str, Any]:
    manifest_path = split_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    max_new_tokens_manifest = _to_int_or_none(manifest.get("max_new_tokens"))

    chunk_paths = sorted(split_dir.glob("chunk_*.pt"))
    sample_count = 0
    claim_count = 0
    chunk_stats = []

    sample_label_counter = Counter()
    claim_label_counter = Counter()
    claim_type_counter = Counter()
    claim_type_label_counter = defaultdict(Counter)
    difficulty_counter = Counter()
    difficulty_claim_label_counter = defaultdict(Counter)
    examples_by_khop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    issues = Counter()
    bad_examples = []
    consistency = Counter()
    generic_claim_text_count = 0
    generic_claim_text_examples = []
    generation_lens: List[int] = []
    samples_with_generation_len = 0
    samples_missing_generation_len = 0
    token_cap_hit_count = 0
    token_cap_hit_no_conclusion_count = 0
    sample_missing_conclusion_claim = 0

    for cidx, cpath in enumerate(chunk_paths):
        data = _safe_torch_load(cpath)
        if not isinstance(data, list):
            issues["chunk_not_list"] += 1
            continue

        chunk_samples = 0
        chunk_claims = 0
        chunk_claim_labels = Counter()

        for sidx, sample in enumerate(data):
            if not isinstance(sample, dict):
                issues["sample_not_dict"] += 1
                continue

            sample_count += 1
            chunk_samples += 1

            s_label = _to_int_or_none(sample.get("label"))
            if s_label is None:
                issues["sample_label_missing_or_invalid"] += 1
                sample_label_counter["invalid"] += 1
            else:
                sample_label_counter[s_label] += 1

            difficulty = sample.get("k_hop", sample.get("task", "NA"))
            difficulty_key = str(difficulty)
            difficulty_counter[difficulty_key] += 1
            if len(examples_by_khop[difficulty_key]) < max(0, int(examples_per_khop)):
                examples_by_khop[difficulty_key].append(_sample_preview(sample))

            claims = sample.get("claims", [])
            verified = sample.get("verified", [])
            sample_label = _to_int_or_none(sample.get("label"))
            has_conclusion_claim_any = _sample_has_conclusion_claim(claims)
            if not has_conclusion_claim_any:
                sample_missing_conclusion_claim += 1
                if _extract_conclusion_text(str(sample.get("generated_text", ""))):
                    issues["missing_conclusion_claim_despite_generated_conclusion"] += 1

            if not claims:
                issues["empty_claims"] += 1
            if not verified:
                issues["empty_verified"] += 1
            if isinstance(claims, list) and isinstance(verified, list) and len(claims) != len(verified):
                issues["claims_verified_length_mismatch"] += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append(
                        {
                            "kind": "claims_verified_length_mismatch",
                            "chunk": cidx,
                            "sample": sidx,
                            "question": _short_text(sample.get("question", "")),
                            "ground_truth": _short_text(sample.get("ground_truth", "")),
                            "predicted_answer": _short_text(sample.get("predicted_answer", "")),
                            "n_claims": len(claims) if isinstance(claims, list) else -1,
                            "n_verified": len(verified) if isinstance(verified, list) else -1,
                        }
                    )

            if not isinstance(claims, list):
                issues["claims_not_list"] += 1
                continue
            if not isinstance(verified, list):
                issues["verified_not_list"] += 1
                continue

            if sample_label in (0, 1) and verified:
                verified_clean = [_to_int_or_none(v) for v in verified]
                verified01 = [v for v in verified_clean if v in (0, 1)]
                if verified01:
                    # Convention: sample/claim label 1=correct, 0=hallucination.
                    if sample_label == 0 and all(v == 1 for v in verified01):
                        consistency["sample_hallucinated_but_all_claims_correct"] += 1
                    if sample_label == 1 and any(v == 0 for v in verified01):
                        consistency["sample_correct_but_has_hallucinated_claim"] += 1

                    conclusion_labels = []
                    for i, c in enumerate(claims):
                        if not isinstance(c, dict):
                            continue
                        if str(c.get("claim_type", "")).strip().lower() == "conclusion" and i < len(verified_clean):
                            if verified_clean[i] in (0, 1):
                                conclusion_labels.append(verified_clean[i])
                    if not conclusion_labels:
                        consistency["sample_missing_conclusion_claim"] += 1
                    else:
                        if sample_label == 0 and all(v == 1 for v in conclusion_labels):
                            consistency["sample_hallucinated_but_conclusion_correct"] += 1
                        if sample_label == 1 and any(v == 0 for v in conclusion_labels):
                            consistency["sample_correct_but_conclusion_hallucinated"] += 1

            a_len = _attention_len(sample)
            if a_len > 0:
                generation_lens.append(a_len)
                samples_with_generation_len += 1
            else:
                samples_missing_generation_len += 1

            if max_new_tokens_manifest and max_new_tokens_manifest > 0 and a_len >= max_new_tokens_manifest:
                token_cap_hit_count += 1
                has_conclusion_claim = _sample_has_conclusion_claim(claims)
                has_conclusion_text = _has_explicit_conclusion_text(str(sample.get("generated_text", "")))
                if (not has_conclusion_claim) and (not has_conclusion_text):
                    token_cap_hit_no_conclusion_count += 1

            for i, claim in enumerate(claims):
                claim_count += 1
                chunk_claims += 1

                if not isinstance(claim, dict):
                    issues["claim_not_dict"] += 1
                    continue

                ctype = str(claim.get("claim_type", "")).strip().lower()
                ctype_id = _to_int_or_none(claim.get("claim_type_id"))
                ctext = str(claim.get("text", ""))
                if ctype not in KNOWN_CLAIM_TYPES:
                    issues["unknown_claim_type"] += 1
                if ctype_id not in KNOWN_TYPE_IDS:
                    issues["unknown_claim_type_id"] += 1
                if _is_generic_claim_text(ctext):
                    issues["generic_claim_text"] += 1
                    generic_claim_text_count += 1
                    if len(generic_claim_text_examples) < 10:
                        generic_claim_text_examples.append(
                            {
                                "chunk": cidx,
                                "sample": sidx,
                                "claim_type": ctype,
                                "claim_text": _short_text(ctext, n=180),
                                "question": _short_text(sample.get("question", ""), n=180),
                            }
                        )

                claim_type_counter[ctype if ctype else "missing"] += 1

                label = _to_int_or_none(verified[i]) if i < len(verified) else None
                if label not in (0, 1, -1):
                    issues["claim_label_invalid"] += 1
                    claim_label_counter["invalid"] += 1
                    chunk_claim_labels["invalid"] += 1
                else:
                    claim_label_counter[label] += 1
                    chunk_claim_labels[label] += 1
                    claim_type_label_counter[ctype if ctype else "missing"][label] += 1
                    difficulty_claim_label_counter[str(difficulty)][label] += 1

                aligned = claim.get("aligned_token_ids", [])
                if not isinstance(aligned, list):
                    issues["aligned_token_ids_not_list"] += 1
                    continue
                if len(aligned) == 0:
                    issues["empty_aligned_token_ids"] += 1

                bad_alignment = False
                for t in aligned:
                    tid = _to_int_or_none(t)
                    if tid is None or tid < 0:
                        bad_alignment = True
                        break
                    if a_len > 0 and tid >= a_len:
                        bad_alignment = True
                        break
                if bad_alignment:
                    issues["aligned_token_ids_out_of_range"] += 1
                    if len(bad_examples) < max_bad_examples:
                        bad_examples.append(
                            {
                                "kind": "aligned_token_ids_out_of_range",
                                "chunk": cidx,
                                "sample": sidx,
                                "question": _short_text(sample.get("question", "")),
                                "claim_text": _short_text(claim.get("text", "")),
                                "attention_len": a_len,
                                "aligned_token_ids_head": aligned[:20],
                            }
                        )

        chunk_stats.append(
            {
                "chunk": cidx,
                "path": str(cpath),
                "samples": chunk_samples,
                "claims": chunk_claims,
                "claim_labels": dict(chunk_claim_labels),
            }
        )

    labeled_claims = claim_label_counter.get(0, 0) + claim_label_counter.get(1, 0)
    correct_rate = (claim_label_counter.get(1, 0) / labeled_claims) if labeled_claims > 0 else None
    hallucination_rate = (claim_label_counter.get(0, 0) / labeled_claims) if labeled_claims > 0 else None
    single_class = (
        labeled_claims > 0
        and (claim_label_counter.get(0, 0) == 0 or claim_label_counter.get(1, 0) == 0)
    )

    warnings = []
    if single_class:
        warnings.append("claim labels collapse to a single class (AUC metrics become invalid)")
    if labeled_claims > 0:
        p_correct = claim_label_counter.get(1, 0) / labeled_claims
        if p_correct >= 0.99 or p_correct <= 0.01:
            warnings.append(f"extreme claim-label imbalance detected: p(correct)={p_correct:.4f}")
    if issues.get("claims_verified_length_mismatch", 0) > 0:
        warnings.append("claims/verified length mismatch exists")
    if issues.get("generic_claim_text", 0) > 0:
        warnings.append("generic/template claim texts detected (e.g., section headers extracted as claims)")
    if issues.get("empty_claims", 0) > 0:
        warnings.append("some samples contain empty claims")
    if consistency.get("sample_hallucinated_but_all_claims_correct", 0) > 0:
        warnings.append("found samples with label=0 but all claim labels are 1")
    if consistency.get("sample_hallucinated_but_conclusion_correct", 0) > 0:
        warnings.append("found samples with label=0 but conclusion label(s) are all 1")
    if sample_missing_conclusion_claim > 0:
        warnings.append(
            f"samples missing conclusion claim: {sample_missing_conclusion_claim}"
        )
    if (
        max_new_tokens_manifest
        and samples_with_generation_len > 0
        and token_cap_hit_count / samples_with_generation_len >= 0.80
    ):
        warnings.append(
            "high generation token-cap hit rate detected "
            f"({token_cap_hit_count}/{samples_with_generation_len})"
        )
    if token_cap_hit_count > 0 and token_cap_hit_no_conclusion_count / token_cap_hit_count >= 0.30:
        warnings.append(
            "many cap-hit samples miss explicit conclusion "
            f"({token_cap_hit_no_conclusion_count}/{token_cap_hit_count})"
        )

    generation_length_stats: Dict[str, Any]
    if generation_lens:
        sorted_lens = sorted(generation_lens)
        generation_length_stats = {
            "count": len(sorted_lens),
            "min": int(sorted_lens[0]),
            "p50": _quantile_int(sorted_lens, 0.50),
            "p90": _quantile_int(sorted_lens, 0.90),
            "p99": _quantile_int(sorted_lens, 0.99),
            "max": int(sorted_lens[-1]),
            "mean": round(float(sum(sorted_lens)) / float(len(sorted_lens)), 4),
        }
    else:
        generation_length_stats = {"count": 0}

    token_cap_hit_rate = (
        token_cap_hit_count / samples_with_generation_len
        if samples_with_generation_len > 0 and max_new_tokens_manifest and max_new_tokens_manifest > 0
        else None
    )
    token_cap_hit_no_conclusion_rate = (
        token_cap_hit_no_conclusion_count / token_cap_hit_count
        if token_cap_hit_count > 0
        else None
    )

    claim_type_label_ratio = {}
    for ctype, label_counter in claim_type_label_counter.items():
        denom = float(sum(label_counter.values()))
        if denom <= 0:
            continue
        claim_type_label_ratio[ctype] = {
            label: round(float(count) / denom, 6)
            for label, count in label_counter.items()
        }

    return {
        "manifest": manifest,
        "num_chunks": len(chunk_paths),
        "samples": sample_count,
        "claims": claim_count,
        "sample_labels": dict(sample_label_counter),
        "claim_labels": dict(claim_label_counter),
        "claim_types": dict(claim_type_counter),
        "claim_type_label_distribution": {
            k: dict(v) for k, v in claim_type_label_counter.items()
        },
        "claim_type_label_ratio": claim_type_label_ratio,
        "difficulty_distribution": dict(difficulty_counter),
        "difficulty_claim_label_distribution": {
            k: dict(v) for k, v in difficulty_claim_label_counter.items()
        },
        "examples_by_khop": dict(examples_by_khop),
        "consistency_checks": dict(consistency),
        "sample_missing_conclusion_claim": int(sample_missing_conclusion_claim),
        "issues": dict(issues),
        "generic_claim_text_count": generic_claim_text_count,
        "generic_claim_text_examples": generic_claim_text_examples,
        "warnings": warnings,
        "claim_correct_rate": correct_rate,
        "claim_hallucination_rate": hallucination_rate,
        "max_new_tokens_from_manifest": max_new_tokens_manifest,
        "generation_length_stats": generation_length_stats,
        "samples_with_generation_len": samples_with_generation_len,
        "samples_missing_generation_len": samples_missing_generation_len,
        "token_cap_hit_count": token_cap_hit_count,
        "token_cap_hit_rate": token_cap_hit_rate,
        "token_cap_hit_no_conclusion_count": token_cap_hit_no_conclusion_count,
        "token_cap_hit_no_conclusion_rate": token_cap_hit_no_conclusion_rate,
        "single_class_claim_labels": single_class,
        "chunk_stats": chunk_stats,
        "bad_examples": bad_examples,
    }


def render_text_report(cache_dir: Path, splits: List[str], result: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Claim Diagnostics Report")
    lines.append("")
    lines.append(f"- cache_dir: {cache_dir}")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- splits: {', '.join(splits)}")
    lines.append("")

    for split in splits:
        info = result.get("splits", {}).get(split)
        if not info:
            lines.append(f"## Split: {split}")
            lines.append("missing")
            lines.append("")
            continue

        lines.append(f"## Split: {split}")
        lines.append(f"- chunks: {info.get('num_chunks', 0)}")
        lines.append(f"- samples: {info.get('samples', 0)}")
        lines.append(f"- claims: {info.get('claims', 0)}")
        lines.append(f"- claim_correct_rate(label=1): {info.get('claim_correct_rate')}")
        lines.append(f"- claim_hallucination_rate(label=0): {info.get('claim_hallucination_rate')}")
        lines.append(f"- max_new_tokens(manifest): {info.get('max_new_tokens_from_manifest')}")
        lines.append(f"- generation_len_stats: {info.get('generation_length_stats', {})}")
        lines.append(
            f"- token_cap_hit_rate: {info.get('token_cap_hit_rate')} "
            f"({info.get('token_cap_hit_count', 0)}/{info.get('samples_with_generation_len', 0)})"
        )
        lines.append(
            f"- token_cap_hit_no_conclusion_rate: {info.get('token_cap_hit_no_conclusion_rate')} "
            f"({info.get('token_cap_hit_no_conclusion_count', 0)}/{info.get('token_cap_hit_count', 0)})"
        )
        lines.append(f"- single_class_claim_labels: {info.get('single_class_claim_labels')}")

        manifest = info.get("manifest", {})
        if manifest:
            lines.append("- manifest:")
            lines.append(f"  - total_count={manifest.get('total_count')}")
            lines.append(f"  - total_correct={manifest.get('total_correct')}")
            lines.append(f"  - total_incorrect={manifest.get('total_incorrect')}")
            lines.append(f"  - correct_rate={manifest.get('correct_rate')}")
            lines.append(f"  - incorrect_rate={manifest.get('incorrect_rate')}")
            lines.append(f"  - total_pending={manifest.get('total_pending')}")
            lines.append(f"  - model_path={manifest.get('model_path', '')}")
            lines.append(f"  - claim_extractor_model={manifest.get('claim_extractor_model', '')}")
            lines.append(f"  - judge_model={manifest.get('judge_model', '')}")

        lines.append(f"- sample_labels: {info.get('sample_labels', {})}")
        lines.append(f"- claim_labels: {info.get('claim_labels', {})}")
        lines.append(f"- claim_types: {info.get('claim_types', {})}")
        lines.append(f"- claim_type_label_ratio: {info.get('claim_type_label_ratio', {})}")
        lines.append(
            f"- generic_claim_text_count: {info.get('generic_claim_text_count', 0)}"
        )
        lines.append(f"- issues: {info.get('issues', {})}")
        lines.append(f"- consistency_checks: {info.get('consistency_checks', {})}")
        if info.get("warnings"):
            lines.append("- warnings:")
            for w in info["warnings"]:
                lines.append(f"  - {w}")

        bad_examples = info.get("bad_examples", [])
        lines.append(f"- bad_examples_shown: {len(bad_examples)}")
        for ex in bad_examples[:10]:
            lines.append(
                f"  - [{ex.get('kind')}] chunk={ex.get('chunk')} sample={ex.get('sample')} "
                f"q={ex.get('question', '')}"
            )

        ex_by_khop = info.get("examples_by_khop", {})
        if ex_by_khop:
            lines.append("- examples_by_k_hop:")
            for kh in sorted(ex_by_khop.keys(), key=lambda x: (len(x), x)):
                lines.append(f"  - k_hop={kh} (n={len(ex_by_khop[kh])})")
                for idx, ex in enumerate(ex_by_khop[kh], 1):
                    lines.append(
                        f"    - sample#{idx}: sample_label={ex.get('sample_label')} "
                        f"answer_correct={ex.get('answer_correct')} "
                        f"pred={ex.get('predicted_answer','')} gt={ex.get('ground_truth','')}"
                    )
                    lines.append(f"      input: {ex.get('input_question','')}")
                    lines.append(f"      output: {ex.get('output_generated','')}")
                    cp = ex.get("claims_preview", [])
                    if cp:
                        lines.append("      claims_preview:")
                        for c in cp:
                            lines.append(
                                f"        - [{c.get('claim_type')}] label={c.get('claim_label')} "
                                f"text={c.get('claim_text')}"
                            )
        generic_examples = info.get("generic_claim_text_examples", [])
        if generic_examples:
            lines.append("- generic_claim_text_examples:")
            for ex in generic_examples[:5]:
                lines.append(
                    f"  - chunk={ex.get('chunk')} sample={ex.get('sample')} "
                    f"type={ex.get('claim_type')} text={ex.get('claim_text')}"
                )
        lines.append("")

    global_warnings = result.get("global_warnings", [])
    if global_warnings:
        lines.append("## Global Warnings")
        for w in global_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out: Dict[str, Any] = {"cache_dir": str(cache_dir), "splits": {}}

    global_warnings = []
    for split in splits:
        split_dir = cache_dir / split
        if not split_dir.exists():
            out["splits"][split] = {"missing": True}
            global_warnings.append(f"missing split directory: {split_dir}")
            continue
        out["splits"][split] = analyze_split(
            split_dir,
            max_bad_examples=args.max_bad_examples,
            examples_per_khop=args.examples_per_khop,
        )
        if out["splits"][split].get("single_class_claim_labels"):
            global_warnings.append(f"{split}: single-class claim labels detected")
        cap_hit_rate = out["splits"][split].get("token_cap_hit_rate")
        if isinstance(cap_hit_rate, (int, float)) and cap_hit_rate >= 0.80:
            global_warnings.append(f"{split}: high token-cap hit rate ({cap_hit_rate:.4f})")

    out["global_warnings"] = global_warnings

    json_path = output_dir / "claim_diagnostics.json"
    txt_path = output_dir / "claim_diagnostics.txt"
    txt = render_text_report(cache_dir, splits, out)

    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")

    print(txt)
    print(f"[saved] {txt_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
