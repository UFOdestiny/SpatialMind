"""
spartun.py - SpaRTUN spatial reasoning dataset (tasksource/SpaRTUN).

SpaRTUN is a synthetic spatial QA benchmark from the same authors/pipeline as
SpartQA (Mirzaee & Kordjamshidi, 2022), sharing the story/question/
candidate_answers shape but mixing two question types:
  - YN (yes/no):        candidate_answers = ["Yes", "No"]
  - FR (free relation):  candidate_answers = 15-way relation vocabulary
                         (left/right/above/below/.../ntppi)

Unlike SpartQA, `answer` is a LIST of correct candidate strings (usually
singleton, occasionally multi-label for FR questions with several correct
relations). Ground truth / correctness is therefore string-set membership
rather than a single integer index.

The HF dataset ships split names train/dev/test (not train/validation/test);
this adapter accepts "validation" as an alias for "dev".

Since the task is classification (selecting from fixed options), correctness
is determined by exact-match membership — no LLM judge needed (needs_judge = False).
"""

import logging
import re
from typing import List, Optional

import torch
from datasets import load_from_disk, DatasetDict

from config import GLOBAL_SEED
from data.base import BaseTaskDataset

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning.\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    "Conclusion: The answer is <option number>.\n"
    "Use only provided options and keep conclusion as the last line."
)

# HF split name -> local alias accepted by this adapter.
_SPLIT_ALIASES = {"validation": "dev"}


def build_prompt(story: str, question: str, candidate_answers: List[str]) -> str:
    """Build a text prompt with numbered options."""
    options_text = "\n".join(
        f"{i + 1}. {opt.strip()}" for i, opt in enumerate(candidate_answers)
    )
    n = len(candidate_answers)
    return (
        f"Context: {story}\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n"
        "Format:\n"
        "Reasoning:\n"
        "- <atomic inference>\n"
        f"Conclusion: The answer is <1-{n}>.\n"
        "Conclusion must be last line.\n"
        "Answer:"
    )


def build_chat_messages(story: str, question: str, candidate_answers: List[str]) -> list:
    """Build chat-format messages for instruction-tuned models."""
    options_text = "\n".join(
        f"{i + 1}. {opt.strip()}" for i, opt in enumerate(candidate_answers)
    )
    user_msg = (
        f"Context: {story}\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


class SpaRTUNDataset(BaseTaskDataset):
    """SpaRTUN spatial QA dataset (YN + FR questions, multi-label answers).

    A classification task with deterministic (set-membership) correctness
    checking, same protocol as SpartQA.
    """

    name = "spartun"
    task_type = "classification"
    needs_judge = False
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "difficulty"  # no native difficulty; placeholder

    eval_config = {
        "split": "test",
        "k_hop_values": None,    # evaluate all samples
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

        log.info(f"Loading SpaRTUN split='{split}' from {dataset_path}")
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            resolved_split = split if split in ds else _SPLIT_ALIASES.get(split, split)
            if resolved_split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            ds = ds[resolved_split]

        # SpaRTUN has no native difficulty field, skip filtering
        # (k_hop_values parameter is ignored for this dataset)

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
        return build_chat_messages(
            raw_item["story"],
            raw_item["question"],
            raw_item["candidate_answers"],
        )

    def build_prompt(self, raw_item: dict) -> str:
        return build_prompt(
            raw_item["story"],
            raw_item["question"],
            raw_item["candidate_answers"],
        )

    def get_question(self, raw_item: dict) -> str:
        return raw_item["question"]

    def get_context(self, raw_item: dict) -> str:
        return str(raw_item["story"])

    def get_ground_truth(self, raw_item: dict) -> str:
        """Return the pipe-joined 0-indexed positions of correct options.

        `answer` is a list of correct candidate STRINGS (usually singleton,
        occasionally multi-label for FR questions with several correct
        relations). We resolve each to its 0-indexed position within
        candidate_answers so ground truth and parse_answer's output are both
        option indices — the same convention SpartQA uses (1-indexed in the
        prompt, 0-indexed internally), just multi-valued here.
        """
        candidates = [c.strip().lower() for c in raw_item["candidate_answers"]]
        answers = raw_item["answer"] or []
        indices = []
        for a in answers:
            a_norm = str(a).strip().lower()
            if a_norm in candidates:
                indices.append(str(candidates.index(a_norm)))
        return "|".join(indices)

    def get_difficulty(self, raw_item: dict) -> int:
        """SpaRTUN has no difficulty levels; return 0."""
        return 0

    def parse_answer(self, generated_text: str) -> str:
        """Extract the selected option number (0-indexed) from LLM output.

        LLM outputs 1-indexed (human readable); convert to 0-indexed to match
        get_ground_truth's convention.
        """
        text = (generated_text or "").strip()
        if not text:
            return "unknown"

        for pattern in (
            r"conclusion\s*:\s*(?:the answer is\s*)?(\d+)",
            r"the answer is\s*(\d+)",
            r"final answer\s*[:\-]?\s*(\d+)",
            r"option\s*(\d+)",
        ):
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                return str(int(matches[-1]) - 1)

        matches = re.findall(r"(?<!\d)(\d+)(?!\d)", text)
        if matches:
            return str(int(matches[-1]) - 1)
        return "unknown"

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """1 iff the predicted (0-indexed) option is among the ground-truth set."""
        if not ground_truth:
            return 0
        return 1 if predicted in ground_truth.split("|") else 0
