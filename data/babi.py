"""
babi.py - bAbI QA dataset for spatial reasoning UQ experiments.

bAbI (Muennighoff/babi) is a free-form QA task. Answers are short strings
(room names, person names, etc.) with many possible values, so correctness
is evaluated via LLM-as-judge (needs_judge = True).

Columns: passage, question, answer, task
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
    "You are an expert in spatial and tracking reasoning.\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    "Conclusion: The answer is <short phrase>.\n"
    "Keep conclusion as the last line."
)


class BabiDataset(BaseTaskDataset):
    """bAbI QA dataset.

    Free-form short answers — requires LLM-as-judge for correctness labeling.
    """

    name = "babi"
    task_type = "free_form"
    needs_judge = True
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "task"

    eval_config = {
        "split": "test",
        "k_hop_values": None,    # evaluate all tasks
        "ood_split": "test",
        "ood_max_samples": 2000,      # cap OOD eval to 2k for efficiency
    }

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        max_length: int = 1024,  # Increased for compatibility
        **kwargs,
    ):
        super().__init__(dataset_path, split, k_hop_values, max_samples)
        self.max_length = max_length

        log.info(f"Loading bAbI split='{split}' from {dataset_path}")
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            ds = ds[split]

        # Filter by task (difficulty) if specified
        if k_hop_values:
            ds = ds.filter(lambda x: x["task"] in k_hop_values)
            log.info(f"  Filtered to task={k_hop_values}: {len(ds)} samples")

        # Subsample if requested
        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.shuffle(seed=GLOBAL_SEED)
            ds = ds.select(range(max_samples))
            log.info(f"  Subsampled to {max_samples} samples (after shuffle, seed={GLOBAL_SEED})")

        self.data = ds
        log.info(f"  Final dataset size: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "passage": item["passage"],
            "question": item["question"],
            "answer": item["answer"],
            "task": item.get("task", 0),
        }

    # --- BaseTaskDataset interface ---

    def build_chat_messages(self, raw_item: dict) -> list:
        passage = raw_item["passage"]
        if isinstance(passage, list):
            passage = "\n".join(passage)
        question = raw_item["question"]
        user_msg = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Respond with Reasoning / Conclusion format."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def build_prompt(self, raw_item: dict) -> str:
        passage = raw_item["passage"]
        if isinstance(passage, list):
            passage = "\n".join(passage)
        question = raw_item["question"]
        return (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Format:\n"
            "Reasoning:\n"
            "- <atomic inference>\n"
            "Conclusion: The answer is <short phrase>.\n"
            "Conclusion must be last line.\n"
            "Answer:"
        )

    def get_question(self, raw_item: dict) -> str:
        return raw_item["question"]

    def get_ground_truth(self, raw_item: dict) -> str:
        return raw_item["answer"]

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("task", 0)

    def parse_answer(self, generated_text: str) -> str:
        """Extract concise answer text with conclusion-first preference."""
        text = (generated_text or "").strip()
        if not text:
            return ""

        patterns = (
            r"conclusion\s*:\s*the answer is\s*([^\n\r\.]+)",
            r"conclusion\s*:\s*([^\n\r\.]+)",
            r"the answer is\s*([^\n\r\.]+)",
            r"final answer\s*[:\-]?\s*([^\n\r\.]+)",
        )
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" .,:;\"'").strip()
        first_line = text.splitlines()[0].strip()
        return first_line.strip(" .,:;\"'").strip()

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Always returns -1 (needs LLM judge)."""
        return -1
