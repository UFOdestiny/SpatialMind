"""Compatibility helpers for runtime dependency mismatches."""

import os
from importlib import import_module


def configure_protobuf_python_implementation() -> bool:
    """Enable pure-Python protobuf backend when not explicitly configured.

    Returns:
        True if this function set the environment variable, False otherwise.
    """
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
        return False

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    return True


def patch_numpy_core_multiarray() -> bool:
    """Patch numpy._core.multiarray when missing.

    Returns:
        True if a patch was applied, False otherwise.
    """
    try:
        np_core = import_module("numpy._core")
    except ModuleNotFoundError:
        return False

    if hasattr(np_core, "multiarray"):
        return False

    np_core.multiarray = import_module("numpy.core.multiarray")
    return True
