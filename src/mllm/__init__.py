"""MLLM clients for MMAD evaluation."""
from .base import BaseLLMClient, INSTRUCTION
from .factory import get_llm_client, MODEL_REGISTRY, list_llm_models

__all__ = [
    "BaseLLMClient",
    "INSTRUCTION",
    # Factory
    "get_llm_client",
    "MODEL_REGISTRY",
    "list_llm_models",
    # Lazy-loaded clients
    "get_gpt4_client",
    "get_claude_client",
    "get_gemini_client",
    "get_qwen_client",
    "get_internvl_client",
    "get_llava_client",
    "get_gemma3_client",
]


# Lazy imports for API clients (avoid import errors if dependencies not installed)
def get_gpt4_client(*args, **kwargs):
    """Get GPT-4o/GPT-4V client."""
    from .openai_client import GPT4Client
    return GPT4Client(*args, **kwargs)


def get_claude_client(*args, **kwargs):
    """Get Claude client."""
    from .claude_client import ClaudeClient
    return ClaudeClient(*args, **kwargs)


def get_gemini_client(*args, **kwargs):
    """Get Gemini client (FREE tier available!)."""
    from .gemini_client import GeminiClient
    return GeminiClient(*args, **kwargs)


def get_qwen_client(*args, **kwargs):
    """Get Qwen2.5-VL client (requires transformers, qwen-vl-utils)."""
    from .qwen_client import QwenVLClient
    return QwenVLClient(*args, **kwargs)


def get_internvl_client(*args, **kwargs):
    """Get InternVL2 client (requires transformers)."""
    from .internvl_client import InternVLClient
    return InternVLClient(*args, **kwargs)


def get_llava_client(*args, **kwargs):
    """Get LLaVA client (requires transformers or llava package)."""
    from .llava_client import LLaVAClient
    return LLaVAClient(*args, **kwargs)


def get_gemma3_client(*args, **kwargs):
    """Get Gemma3 client (requires transformers; quantization requires bitsandbytes + CUDA)."""
    from .gemma3_client import Gemma3Client
    return Gemma3Client(*args, **kwargs)
