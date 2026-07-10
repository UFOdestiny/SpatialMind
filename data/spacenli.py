"""
spacenli.py - SpaceNLI spatial natural language inference dataset
(tasksource/SpaceNLI; Abzianidze, Zwarts & Winter, 2023).

SpaceNLI is a different task shape from StepGame/SpartQA/SpaRTUN/bAbI: instead
of a story + question, each item is a PREMISE set + HYPOTHESIS pair, and the
model must classify the inference as one of {entailment, contradiction,
neutral} — standard 3-way NLI, specialized to spatial reasoning patterns
(topological, directional, projective relations, etc).

The HF dataset ships a single 32000-row "train" split with no pre-built
train/val/test partition. This adapter creates a seeded 80/10/10 split
(stratified by label so class balance is preserved) inside __init__ whenever
train/validation/test is requested, using the same GLOBAL_SEED the other
adapters use for their subsampling shuffles.

`prem_num` (2-4 premises per problem) is exposed as difficulty_field — more
premises means a longer inference chain, the closest analogue to StepGame's
k_hop for this dataset.

Since the label is exact and deterministic, no LLM judge is needed
(needs_judge = False).
"""

import logging
import re
from typing import List, Optional

import torch
from datasets import load_from_disk, DatasetDict, ClassLabel

from config import GLOBAL_SEED
from data.base import BaseTaskDataset

log = logging.getLogger(__name__)

LABELS = ["contradiction", "entailment", "neutral"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning and natural language inference.\n"
    "Given premises and a hypothesis, decide whether the hypothesis is\n"
    "entailed by, contradicted by, or neutral (unrelated/underdetermined) with\n"
    "respect to the premises.\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    "Conclusion: The answer is <entailment/contradiction/neutral>.\n"
    "Keep conclusion as the last line."
)

# Seeded 80/10/10 split over the single HF "train" partition.
_TRAIN_FRAC = 0.8
_VAL_FRAC_OF_REST = 0.5  # remaining 20% split evenly -> 10% val, 10% test


def build_prompt(premises: str, hypothesis: str) -> str:
    return (
        f"Premises: {premises}\n"
        f"Hypothesis: {hypothesis}\n"
        "Format:\n"
        "Reasoning:\n"
        "- <atomic inference>\n"
        "Conclusion: The answer is <entailment/contradiction/neutral>.\n"
        "Conclusion must be last line.\n"
        "Answer:"
    )


def build_chat_messages(premises: str, hypothesis: str) -> list:
    user_msg = f"Premises: {premises}\nHypothesis: {hypothesis}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


class SpaceNLIDataset(BaseTaskDataset):
    """SpaceNLI spatial NLI dataset (3-way entailment/contradiction/neutral).

    A classification task with deterministic correctness checking, following
    the same protocol as StepGame's label matching.
    """

    name = "spacenli"
    task_type = "classification"
    needs_judge = False
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "prem_num"

    eval_config = {
        "split": "test",
        "k_hop_values": None,    # evaluate all premise-count levels
        "ood_split": "test",
        "ood_max_samples": 0,         # all samples for OOD eval
    }

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        tokenizer=None,
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        max_length: int = 1024,
        **kwargs,
    ):
        super().__init__(dataset_path, split, k_hop_values, max_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading SpaceNLI split='{split}' from {dataset_path}")
        raw = load_from_disk(dataset_path)
        if isinstance(raw, DatasetDict):
            # HF ships a single "train" split; if the on-disk dataset already
            # has train/validation/test (e.g. from a prior manual split), use
            # them directly instead of re-splitting.
            if split in raw and len(raw.keys()) > 1:
                ds = raw[split]
            else:
                full = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
                ds = self._make_split(full, split)
        else:
            ds = self._make_split(raw, split)

        # Filter by premise count (difficulty) if specified.
        if k_hop_values:
            ds = ds.filter(lambda x: x["prem_num"] in k_hop_values)
            log.info(f"  Filtered to prem_num={k_hop_values}: {len(ds)} samples")

        # Subsample if requested.
        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.shuffle(seed=GLOBAL_SEED)
            ds = ds.select(range(max_samples))
            log.info(f"  Subsampled to {max_samples} samples (after shuffle, seed={GLOBAL_SEED})")

        self.data = ds
        log.info(f"  Final dataset size: {len(self.data)} samples")

    @staticmethod
    def _make_split(full_ds, split: str):
        """Seeded 80/10/10 stratified split of the single HF train partition."""
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split '{split}'. Expected train/validation/test.")
        if not isinstance(full_ds.features["label"], ClassLabel):
            full_ds = full_ds.cast_column("label", ClassLabel(names=LABELS))
        first = full_ds.train_test_split(
            test_size=(1.0 - _TRAIN_FRAC), seed=GLOBAL_SEED, stratify_by_column="label"
        )
        if split == "train":
            return first["train"]
        second = first["test"].train_test_split(
            test_size=_VAL_FRAC_OF_REST, seed=GLOBAL_SEED, stratify_by_column="label"
        )
        return second["train"] if split == "validation" else second["test"]

    def __len__(self):
        return len(self.data)

    # --- BaseTaskDataset interface ---

    def build_chat_messages(self, raw_item: dict) -> list:
        return build_chat_messages(raw_item["premises"], raw_item["hypothesis"])

    def build_prompt(self, raw_item: dict) -> str:
        return build_prompt(raw_item["premises"], raw_item["hypothesis"])

    def get_question(self, raw_item: dict) -> str:
        return raw_item["hypothesis"]

    def get_ground_truth(self, raw_item: dict) -> str:
        label = raw_item["label"]
        # After cast_column, label is an int index into LABELS; the manual
        # split path (on-disk train/validation/test) may instead carry the
        # original string label — handle both.
        if isinstance(label, int):
            return LABELS[label]
        return str(label)

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("prem_num", 0)

    def parse_answer(self, generated_text: str) -> str:
        """Extract one of {entailment, contradiction, neutral} from LLM output."""
        text = (generated_text or "").strip()
        if not text:
            return "unknown"
        text_lower = text.lower()

        line_candidates = []
        for pattern in (
            r"conclusion\s*:\s*(?:the answer is\s*)?([^\n\r\.]+)",
            r"the answer is\s*([^\n\r\.]+)",
            r"final answer\s*[:\-]?\s*([^\n\r\.]+)",
        ):
            matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
            if matches:
                line_candidates.append(matches[-1])
        line_candidates.append(text_lower)

        labels_by_len = sorted(LABELS, key=len, reverse=True)
        for candidate in line_candidates:
            for label in labels_by_len:
                if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", candidate):
                    return label
        return "unknown"

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Exact label match: 1=correct, 0=hallucination."""
        return 1 if predicted == ground_truth else 0
