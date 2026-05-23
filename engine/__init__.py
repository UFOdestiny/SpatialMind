"""Generation engine backends for Phase 1 feature caching."""

def get_engine(backend: str, **kwargs):
    """Factory: return the appropriate generation engine.

    Args:
        backend: "vllm" for hybrid vLLM+HF, "hf" for pure HuggingFace.
        **kwargs: Passed to engine constructor (cfg, model_path, device, ...).
    """
    if backend == "vllm":
        from engine.vllm_engine import VLLMGenerationEngine
        return VLLMGenerationEngine(**kwargs)
    elif backend == "hf":
        from engine.hf_engine import HFGenerationEngine
        return HFGenerationEngine(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'vllm' or 'hf'.")
