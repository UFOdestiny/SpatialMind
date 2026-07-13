"""
spartqa_yn.py - SpartQA yes/no spatial reasoning dataset
(tasksource/spartqa-yn; Mirzaee et al., 2021, the AUTO/synthetic portion).

Unlike the human-authored `spartqa-mchoice` adapter, this is the machine-
generated block-world variant: highly templated stories ("Block B is below A.
Block A contains a medium yellow square.") that the explicit spatial-relation
parser covers densely (~8 relations/story, 100% story coverage). It provides a
HIGH-symbolizability yes/no task, complementing StepGame (directional) and
filling the gap left by the low-coverage free-form datasets.

Each item: story (context), question, answer in {Yes, No, DK}. Deterministic
3-way classification, so no LLM judge is needed (needs_judge = False).
"""

import logging
import re
from typing import List, Optional

import torch
from datasets import load_from_disk, DatasetDict

from config import GLOBAL_SEED
from data.base import BaseTaskDataset

log = logging.getLogger(__name__)

# Canonical labels. "DK" = don't know / underdetermined (the neutral case).
LABELS = ["Yes", "No", "DK"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning.\n"
    "Given a story describing a spatial layout and a yes/no question, decide\n"
    "whether the answer is Yes, No, or DK (cannot be determined from the story).\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    "Conclusion: The answer is <Yes/No/DK>.\n"
    "Keep conclusion as the last line."
)


def build_prompt(story: str, question: str) -> str:
    return (
        f"Context: {story}\n"
        f"Question: {question}\n"
        "Format:\n"
        "Reasoning:\n"
        "- <atomic inference>\n"
        "Conclusion: The answer is <Yes/No/DK>.\n"
        "Conclusion must be last line.\n"
        "Answer:"
    )


def build_chat_messages(story: str, question: str) -> list:
    user_msg = f"Context: {story}\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


class SpartQAYNDataset(BaseTaskDataset):
    """SpartQA yes/no spatial reasoning dataset (3-way Yes/No/DK).

    A classification task with deterministic correctness checking, following
    the same protocol as StepGame's label matching.
    """

    name = "spartqa_yn"
    task_type = "classification"
    needs_judge = False
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "difficulty"  # no native difficulty; placeholder

    eval_config = {
        "split": "test",
        "k_hop_values": None,
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
        max_length: int = 1024,
        **kwargs,
    ):
        super().__init__(dataset_path, split, k_hop_values, max_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading SpartQA-YN split='{split}' from {dataset_path}")
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            ds = ds[split]

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

    def get_context(self, raw_item: dict) -> str:
        return str(raw_item["story"])

    def get_ground_truth(self, raw_item: dict) -> str:
        return str(raw_item["answer"]).strip()

    def get_difficulty(self, raw_item: dict) -> int:
        return 0

    def parse_answer(self, generated_text: str) -> str:
        """Extract one of {Yes, No, DK} from LLM output."""
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

        # Match "dk"/"don't know" -> DK; "yes" -> Yes; "no" -> No.
        for candidate in line_candidates:
            if re.search(r"(?<![a-z])(dk|don'?t know|cannot|can not|unknown|undetermined)(?![a-z])", candidate):
                return "DK"
            if re.search(r"(?<![a-z])yes(?![a-z])", candidate):
                return "Yes"
            if re.search(r"(?<![a-z])no(?![a-z])", candidate):
                return "No"
        return "unknown"

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Exact label match: 1=correct, 0=hallucination."""
        return 1 if predicted == ground_truth else 0
