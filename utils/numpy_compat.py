"""Compatibility helpers for runtime dependency mismatches."""

import logging
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


class _SubstringLogFilter(logging.Filter):
    """Drop log records whose message contains any of the given substrings."""

    def __init__(self, substrings):
        super().__init__()
        self._substrings = tuple(substrings)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(s in msg for s in self._substrings)


def silence_transformers_chat_template_warning() -> bool:
    """Suppress the repeated transformers apply_chat_template(tokenize=False) warning.

    transformers emits a ``MistralCommonBackend.apply_chat_template(..., tokenize=False)
    is unsafe`` warning on every call, flooding logs during generation/judging. We
    intentionally build the prompt string ourselves, so the warning is noise.

    The warning originates from a ``transformers`` child logger. transformers installs
    its own StreamHandler on the ``transformers`` logger with ``propagate=False``, so
    the record never reaches the root handler and a logger-level filter on the parent
    is skipped for propagated child records. Filters must therefore live on the
    *handlers* that actually emit the record: the ``transformers`` logger's handlers
    plus the root handlers (in case transformers' own logging is disabled/reset).

    Call this after ``logging.basicConfig`` so the root handler already exists. Import
    of transformers is triggered so its handler exists to receive the filter.

    Returns:
        True if the filter was installed on at least one target, False otherwise.
    """
    substrings = ("apply_chat_template",)

    def _install(target) -> bool:
        for existing in getattr(target, "filters", []):
            if isinstance(existing, _SubstringLogFilter):
                return False
        target.addFilter(_SubstringLogFilter(substrings))
        return True

    handlers = list(logging.getLogger().handlers)

    tf_logger = logging.getLogger("transformers")
    try:  # ensure transformers' own handler is created before we filter it
        from transformers.utils import logging as _tf_logging

        _tf_logging.get_logger()
        tf_logger = logging.getLogger("transformers")
    except Exception:
        pass
    handlers.extend(tf_logger.handlers)

    installed = False
    for handler in handlers:
        installed = _install(handler) or installed
    # Also filter at the logger level for records logged directly to it.
    installed = _install(tf_logger) or installed
    return installed
