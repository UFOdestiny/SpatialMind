"""
spartqa.py - SpartQA multiple-choice spatial reasoning dataset.

SpartQA is a 4-option multiple-choice QA task.
Each sample has:
  - story: context/background text
  - question: the question
  - candidate_answers: list of 4 answer options
  - answer: integer index (0-3) of the correct option

Since the task is classification (selecting from fixed options), correctness
is determined by exact index match — no LLM judge needed (needs_judge = False).
"""

import logging
import re
from typing import List, Optional

import torch
from datasets import load_from_disk, DatasetDict

from config import GLOBAL_SEED
from data.base import BaseTaskDataset

log = logging.getLogger(__name__)

# 4 answer option indices
NUM_OPTIONS = 4
LABELS = ["1", "2", "3", "4"]  # display labels (1-indexed for user clarity)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning.\n"
    "Output format:\n"
    "Reasoning:\n"
    "- <atomic inference>\n"
    "Conclusion: The answer is <option number 1/2/3/4>.\n"
    "Use only provided options and keep conclusion as the last line."
)


def build_prompt(story: str, question: str, candidate_answers: List[str]) -> str:
    """Build a text prompt with numbered options."""
    options_text = "\n".join(
        f"{i+1}. {opt.strip()}" for i, opt in enumerate(candidate_answers)
    )
    return (
        f"Context: {story}\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n"
        "Format:\n"
        "Reasoning:\n"
        "- <atomic inference>\n"
        "Conclusion: The answer is <1/2/3/4>.\n"
        "Conclusion must be last line.\n"
        f"Answer:"
    )


def build_chat_messages(story: str, question: str, candidate_answers: List[str]) -> list:
    """Build chat-format messages for instruction-tuned models."""
    options_text = "\n".join(
        f"{i+1}. {opt.strip()}" for i, opt in enumerate(candidate_answers)
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


class SpartQADataset(BaseTaskDataset):
    """SpartQA multiple-choice spatial reasoning dataset.

    A 4-class classification task with deterministic correctness checking.
    """

    name = "spartqa"
    task_type = "classification"
    needs_judge = False
    system_prompt = SYSTEM_PROMPT
    difficulty_field = "difficulty"  # spartqa doesn't have native difficulty; placeholder

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
        max_length: int = 1024,  # Increased for compatibility
        **kwargs,
    ):
        super().__init__(dataset_path, split, k_hop_values, max_samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

        log.info(f"Loading SpartQA split='{split}' from {dataset_path}")
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
            ds = ds[split]

        # SpartQA doesn't have a native difficulty field, skip filtering
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

    def __getitem__(self, idx):
        item = self.data[idx]
        story = item["story"]
        question = item["question"]
        candidate_answers = item["candidate_answers"]
        answer_idx = item["answer"]  # integer 0-3

        # Tokenize if tokenizer is provided
        if self.tokenizer is not None:
            messages = build_chat_messages(story, question, candidate_answers)
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = build_prompt(story, question, candidate_answers)

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
            text = build_prompt(story, question, candidate_answers)
            input_ids = text
            attention_mask = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(answer_idx, dtype=torch.long),  # 0-indexed for training
            "difficulty": 0,  # placeholder since spartqa has no difficulty levels
        }

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

    def get_ground_truth(self, raw_item: dict) -> str:
        """Return 0-indexed answer string for consistent comparison with parse_answer."""
        answer_idx = raw_item["answer"]
        return str(answer_idx)  # 0-indexed string, e.g. "0", "1", "2", "3"

    def get_difficulty(self, raw_item: dict) -> int:
        """SpartQA has no difficulty levels; return 0."""
        return 0

    def parse_answer(self, generated_text: str) -> str:
        """Extract the selected option number (0-3) from LLM output.
        
        LLM outputs 1-4 (human readable), we convert to 0-3 (internal index).
        """
        text = generated_text.strip()

        # Prefer explicit conclusion/final-answer lines to avoid picking
        # unrelated digits from reasoning steps.
        for pattern in (
            r"conclusion\s*:\s*(?:the answer is\s*)?([1-4])(?:\D|$)",
            r"the answer is\s*([1-4])(?:\D|$)",
            r"final answer\s*[:\-]?\s*([1-4])(?:\D|$)",
            r"option\s*([1-4])(?:\D|$)",
        ):
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                return str(int(matches[-1]) - 1)

        # Fallback: use a standalone option number and take the last one
        # (usually nearest to the final conclusion).
        matches = re.findall(r"(?<!\d)([1-4])(?!\d)", text)
        if matches:
            return str(int(matches[-1]) - 1)

        # Fallback: check if full answer text matches any candidate
        # (this would require access to candidates, so just return "unknown")
        return "unknown"

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Exact match on option number: 1=correct, 0=hallucination."""
        return 1 if predicted == ground_truth else 0
