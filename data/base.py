"""
base.py - Abstract base class for task datasets.

All task datasets (StepGame, bAbI, SpartQA, etc.) inherit from BaseTaskDataset.
This abstraction decouples Phase 1 generation from any specific dataset.

Key method: check_correctness() returns:
    1  = correct / non-hallucination
    0  = incorrect / hallucination
    -1 = needs LLM judge (for free-form tasks)

Each dataset also embeds its own evaluation config (difficulty_field,
default eval splits, etc.) so that evaluate.py is fully dataset-agnostic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class BaseTaskDataset(Dataset, ABC):
    """Abstract base for task datasets used in Phase 1 generation.

    Subclasses must implement loading logic and the abstract methods below.
    The generation pipeline calls these methods to build prompts, parse
    answers, and determine correctness without knowing dataset internals.

    Class-level attributes (override in subclasses):
        name:               Registry key, e.g. "stepgame", "babi"
        task_type:          "classification" or "free_form"
        needs_judge:        Whether correctness requires LLM-as-judge
        system_prompt:      System prompt for instruction-tuned models
        difficulty_field:   Name of the difficulty column in cached data (e.g. "k_hop", "task")
        eval_config:        Default eval parameters embedded in the dataset.
                            Keys: split, k_hop_values, ood_split, ood_max_samples
    """

    name: str = ""
    task_type: str = ""
    needs_judge: bool = False
    system_prompt: str = ""
    difficulty_field: str = "k_hop"

    # Default evaluation config — override per dataset.
    # evaluate.py reads these so no dataset-specific CLI args are needed.
    eval_config: Dict = {
        "split": "test",              # default eval split
        "k_hop_values": None,    # None = all; list of ints to filter
        "ood_split": "test",          # split to use when this dataset is OOD target
        "ood_max_samples": 0,         # 0 = all samples for OOD eval
    }

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        k_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
    ):
        """
        Args:
            dataset_path: Path to the dataset on disk.
            split: Data split ("train", "validation", "test").
            k_hop_values: Filter samples by difficulty level (e.g. k_hop).
                None or empty list means all difficulties.
            max_samples: Subsample to at most this many samples. 0 = no limit.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.k_hop_values = k_hop_values
        self.max_samples = max_samples

    @abstractmethod
    def build_chat_messages(self, raw_item: dict) -> list:
        """Build chat-format messages for instruction-tuned models.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """

    def build_prompt(self, raw_item: dict) -> str:
        """Build a plain-text prompt (fallback when chat template unavailable)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement build_prompt(). "
            "Override this method or ensure the tokenizer supports chat templates."
        )

    def get_question(self, raw_item: dict) -> str:
        """Extract the question text from a raw item (used by judge)."""
        return raw_item.get("question", "")

    def get_context(self, raw_item: dict) -> str:
        """Return the trusted scene/premise text used by constraint analysis."""
        value = raw_item.get("story", raw_item.get("passage", raw_item.get("premises", "")))
        if isinstance(value, (list, tuple)):
            return " ".join(str(x) for x in value)
        return str(value or "")

    def get_ground_truth(self, raw_item: dict) -> str:
        """Extract the ground truth answer string from a raw item."""
        return raw_item.get("label", "")

    def get_difficulty(self, raw_item: dict) -> int:
        """Extract difficulty level from a raw item."""
        return raw_item.get(self.difficulty_field, 0)

    @abstractmethod
    def parse_answer(self, generated_text: str) -> str:
        """Dataset-specific parsing of the LLM's generated text.

        For classification tasks, extract the predicted label.
        For free-form tasks, normalize/strip the text.
        """

    def check_correctness(self, predicted: str, ground_truth: str) -> int:
        """Determine if the prediction is correct.

        Returns:
            1  = correct / non-hallucination
            0  = incorrect / hallucination
            -1 = needs LLM judge (for free-form tasks)
        """
        if self.needs_judge:
            return -1
        return 1 if predicted == ground_truth else 0
