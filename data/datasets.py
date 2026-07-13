"""
datasets.py - Task dataset registry.

Usage:
    from data.datasets import get_dataset, get_dataset_cls

    ds = get_dataset("stepgame", dataset_path="...", split="test")
    cls = get_dataset_cls("babi")  # get class without instantiation
"""

from data.stepgame import (
    StepGameDataset,
    LABELS,
    LABEL2ID,
    ID2LABEL,
    build_prompt,
    build_chat_messages,
)
from data.babi import BabiDataset
from data.spartqa import SpartQADataset
from data.spartqa_yn import SpartQAYNDataset
from data.spartun import SpaRTUNDataset
from data.spacenli import SpaceNLIDataset
from data.base import BaseTaskDataset

DATASET_REGISTRY = {
    "stepgame": StepGameDataset,
    "StepGame": StepGameDataset,      # folder name alias
    "babi": BabiDataset,
    "spartqa": SpartQADataset,
    "SpartQA": SpartQADataset,        # folder name alias
    "spartqa_yn": SpartQAYNDataset,
    "SpartQA_YN": SpartQAYNDataset,   # folder name alias
    "spartun": SpaRTUNDataset,
    "SpaRTUN": SpaRTUNDataset,        # folder name alias
    "spacenli": SpaceNLIDataset,
    "SpaceNLI": SpaceNLIDataset,      # folder name alias
}


def get_dataset(name: str, **kwargs) -> BaseTaskDataset:
    """Instantiate a dataset by registry name.

    Args:
        name: Dataset name (e.g. "stepgame", "babi").
        **kwargs: Passed to the dataset constructor.

    Returns:
        An instance of the requested dataset.
    """
    cls = DATASET_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(set(DATASET_REGISTRY.keys())))
        raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")
    return cls(**kwargs)


def get_dataset_cls(name: str) -> type:
    """Get dataset class by name (without instantiation).

    Useful for reading class-level attributes like eval_config, needs_judge, etc.
    """
    cls = DATASET_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(set(DATASET_REGISTRY.keys())))
        raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")
    return cls
