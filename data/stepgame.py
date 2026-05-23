"""
stepgame.py - StepGame dataset for spatial reasoning UQ experiments.

All StepGame-specific logic lives here: labels, prompt construction,
answer parsing, and correctness checks.

StepGame is a classification task — check_correctness() returns 1 or 0 directly.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import torch
from datasets import (
    Dataset as HFDataset,
    DatasetDict,
    DatasetInfo,
    Features,
    load_from_disk,
)

from config import GLOBAL_SEED
from data.base import BaseTaskDataset

log = logging.getLogger(__name__)

# 9 spatial relation labels
LABELS = [
    "upper-right", "lower-right", "upper-left", "lower-left",
    "above", "left", "below", "right", "overlap",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL_LIST_TEXT = ", ".join(LABELS)

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning.\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    f"Conclusion: The answer is <one label from [{LABEL_LIST_TEXT}]>.\n"
    "Keep conclusion as the last line."
)


def build_prompt(story: list, question: str) -> str:
    """Build a text prompt from story sentences and question."""
    context = " ".join(story)
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Allowed labels: [{LABEL_LIST_TEXT}]\n"
        "Format:\n"
        "Reasoning:\n"
        "- <atomic inference>\n"
        "Conclusion: The answer is <one allowed label>.\n"
        "Conclusion must be last line.\n"
        "Answer:"
    )


def build_chat_messages(story: list, question: str) -> list:
    """Build chat-format messages for instruction-tuned models."""
    context = " ".join(story)
    user_msg = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Allowed labels: [{LABEL_LIST_TEXT}]"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Arrow loading helpers (handles legacy HF metadata edge cases)
# ---------------------------------------------------------------------------

def _resolve_split_dir(dataset_path: str, split: str) -> Path:
    """Resolve the on-disk directory that contains split arrow shards."""
    dataset_root = Path(dataset_path)
    split_dir = dataset_root / split
    if split_dir.is_dir():
        return split_dir

    if dataset_root.name == split and dataset_root.is_dir():
        return dataset_root

    raise ValueError(
        f"Cannot resolve split directory for split='{split}' from dataset_path='{dataset_path}'."
    )


def _read_arrow_table_without_hf_metadata(arrow_path: Path) -> pa.Table:
    """Read a HF arrow shard and drop schema metadata to avoid legacy feature parsing."""
    with pa.memory_map(str(arrow_path), "r") as source:
        try:
            reader = pa.ipc.RecordBatchFileReader(source)
        except pa.ArrowInvalid:
            reader = pa.ipc.RecordBatchStreamReader(source)
        table = reader.read_all()
    return table.replace_schema_metadata(None)


def _load_split_from_arrow(dataset_path: str, split: str) -> HFDataset:
    """Fallback loader that builds a Dataset directly from split arrow shards."""
    split_dir = _resolve_split_dir(dataset_path, split)
    data_files = sorted(split_dir.glob("data-*.arrow"))
    if not data_files:
        raise FileNotFoundError(f"No data-*.arrow files found under {split_dir}")

    tables = [_read_arrow_table_without_hf_metadata(path) for path in data_files]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    features = Features.from_arrow_schema(table.schema)
    info = DatasetInfo(features=features)
    return HFDataset(arrow_table=table, info=info, split=split)


def _load_from_disk_with_compat(dataset_path: str, split: str):
    """Load a split with fallback for legacy List metadata in older datasets artifacts."""
    try:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            return ds[split]
        return ds
    except ValueError as exc:
        if "Feature type 'List' not found" not in str(exc):
            raise

        log.warning(
            "Detected legacy HF feature metadata at %s; falling back to metadata-free arrow "
            "loading for split '%s'.",
            dataset_path,
            split,
        )
        return _load_split_from_arrow(dataset_path, split)


# ---------------------------------------------------------------------------
# StepGame dataset class
# ---------------------------------------------------------------------------

class StepGameDataset(BaseTaskDataset):
    """StepGame spatial reasoning dataset.

    Each item yields:
        input_ids:      tokenized prompt (LongTensor)  — if tokenizer provided
        attention_mask:  attention mask (LongTensor)    — if tokenizer provided
        label:          class index 0-8 (LongTensor)
        k_hop:          reasoning hops (int)
    """

    name = "stepgame"
    task_type = "classification"
    needs_judge = False
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "k_hop"

    eval_config = {
        "split": "test",
        "k_hop_values": None,    # test has k_hop 1-10, evaluate all
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        tokenizer=None,
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        max_length: int = 1024,  # Increased from 512 for 9-hop support
    ):
        super().__init__(dataset_path, split, k_hop_values, max_samples)

        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading StepGame split='{split}' from {dataset_path}")
        ds = _load_from_disk_with_compat(dataset_path, split)

        # Filter by k_hop if specified
        if k_hop_values:
            ds = ds.filter(lambda x: x["k_hop"] in k_hop_values)
            log.info(f"  Filtered to k_hop={k_hop_values}: {len(ds)} samples")

        # Subsample if requested
        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.shuffle(seed=GLOBAL_SEED)
            ds = ds.select(range(max_samples))
            log.info(f"  Subsampled to {max_samples} samples (after shuffle, seed={GLOBAL_SEED})")

        self.data = ds
        log.info(f"  Final dataset size: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    # --- BaseTaskDataset interface ---

    def build_chat_messages(self, raw_item: dict) -> list:
        return build_chat_messages(raw_item["story"], raw_item["question"])

    def build_prompt(self, raw_item: dict) -> str:
        return build_prompt(raw_item["story"], raw_item["question"])

    def get_question(self, raw_item: dict) -> str:
        return raw_item["question"]

    def get_ground_truth(self, raw_item: dict) -> str:
        return raw_item["label"]

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("k_hop", 0)

    def parse_answer(self, generated_text: str) -> str:
        """Extract spatial direction from LLM generated text."""
        text = (generated_text or "").strip()
        if not text:
            return "unknown"

        text_lower = text.lower()
        line_candidates = []
        for pattern in (
            r"conclusion\s*:\s*([^\n\r]+)",
            r"the answer is\s*([^\n\r]+)",
            r"final answer\s*[:\-]?\s*([^\n\r]+)",
        ):
            matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
            if matches:
                line_candidates.append(matches[-1])
        line_candidates.append(text_lower)

        # Match longer labels first to avoid substring collisions
        # (e.g., "upper-right" incorrectly matched as "right").
        labels_by_len = sorted(LABELS, key=len, reverse=True)

        def _normalize_for_match(x: str) -> str:
            return re.sub(r"[\s_]+", "-", x.lower())

        def _find_last_label_with_boundaries(x: str) -> Optional[str]:
            best_label = None
            best_pos = -1
            for label in labels_by_len:
                pat = rf"(?<![a-z-]){re.escape(label)}(?![a-z-])"
                for m in re.finditer(pat, x):
                    if m.start() > best_pos:
                        best_pos = m.start()
                        best_label = label
            return best_label

        for candidate in line_candidates:
            normalized = _normalize_for_match(candidate)
            picked = _find_last_label_with_boundaries(normalized)
            if picked is not None:
                return picked
        return "unknown"

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Exact string match: 1=correct, 0=hallucination."""
        return 1 if predicted == ground_truth else 0

    # --- __getitem__ for Phase 1 tokenized access (backward compat) ---

    def __getitem__(self, idx):
        item = self.data[idx]
        story = item["story"]
        question = item["question"]
        label_str = item["label"]
        k_hop = item["k_hop"]

        # Map label string to class id
        label_id = LABEL2ID.get(label_str, -1)
        if label_id == -1:
            for key in LABEL2ID:
                if key in label_str.lower() or label_str.lower() in key:
                    label_id = LABEL2ID[key]
                    break
            if label_id == -1:
                raise ValueError(
                    f"Unknown label '{label_str}' at idx {idx}. "
                    f"Valid labels are: {list(LABEL2ID.keys())}"
                )

        # Tokenize
        if self.tokenizer is not None:
            messages = build_chat_messages(story, question)
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = build_prompt(story, question)

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            text = build_prompt(story, question)
            input_ids = text
            attention_mask = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "k_hop": k_hop,
        }
