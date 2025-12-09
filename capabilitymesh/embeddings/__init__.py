"""Embeddings for semantic capability search.

This module provides pluggable embedding implementations:
- KeywordEmbedder: Simple keyword-based matching (no ML, zero dependencies)
- LocalEmbedder: Local embeddings using sentence-transformers
- OpenAIEmbedder: Cloud embeddings using OpenAI API
"""

from .base import Embedder, EmbeddingResult
from .keyword import KeywordEmbedder

# Optional embedders (may not be available)
_LOCAL_EMBEDDER_AVAILABLE = False
_OPENAI_EMBEDDER_AVAILABLE = False

try:
    from .local import LocalEmbedder
    _LOCAL_EMBEDDER_AVAILABLE = True
except ImportError:
    LocalEmbedder = None

try:
    from .openai import OpenAIEmbedder
    _OPENAI_EMBEDDER_AVAILABLE = True
except ImportError:
    OpenAIEmbedder = None


def auto_select_embedder(**kwargs) -> Embedder:
    """Auto-select the best available embedder.

    Selection priority:
    1. LocalEmbedder (if sentence-transformers is installed)
    2. OpenAIEmbedder (if openai library + API key available)
    3. KeywordEmbedder (always available as fallback)

    Args:
        **kwargs: Arguments to pass to the embedder constructor

    Returns:
        Best available Embedder instance
    """
    # Try LocalEmbedder first (best quality, no API costs)
    if _LOCAL_EMBEDDER_AVAILABLE:
        try:
            return LocalEmbedder(**kwargs)
        except Exception:
            pass

    # Try OpenAIEmbedder (high quality, requires API key)
    if _OPENAI_EMBEDDER_AVAILABLE:
        try:
            return OpenAIEmbedder(**kwargs)
        except Exception:
            pass

    # Fallback to KeywordEmbedder (always works)
    return KeywordEmbedder(**kwargs)


__all__ = [
    "Embedder",
    "EmbeddingResult",
    "KeywordEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "auto_select_embedder",
]
