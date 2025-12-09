"""OpenAI embedder using OpenAI API."""

import os
from typing import List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from .base import Embedder, EmbeddingResult


class OpenAIEmbedder(Embedder):
    """Cloud-based embedding using OpenAI API.

    Uses OpenAI's text-embedding models for high-quality semantic embeddings.
    Requires an OpenAI API key and incurs API costs.

    Default model: 'text-embedding-3-small'
    - High quality embeddings
    - 1536 dimensions
    - Fast API response
    - Cost: ~$0.02 per 1M tokens

    Requires:
    - pip install openai
    - OPENAI_API_KEY environment variable or api_key parameter
    """

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """Initialize OpenAI embedder.

        Args:
            model_name: Name of OpenAI embedding model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

        Raises:
            ImportError: If openai library is not installed
            ValueError: If API key is not provided
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            )

        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self._dimension = self.SUPPORTED_MODELS[model_name]
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )

        embedding = response.data[0].embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            metadata={
                "dimension": self._dimension,
                "usage": response.usage.model_dump(),
            },
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed (max 2048 texts per request)

        Returns:
            List of EmbeddingResult objects
        """
        # OpenAI supports batching up to 2048 inputs
        if len(texts) > 2048:
            # Split into chunks
            results = []
            for i in range(0, len(texts), 2048):
                chunk = texts[i:i + 2048]
                chunk_results = await self.embed_batch(chunk)
                results.extend(chunk_results)
            return results

        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        results = []
        for text, data in zip(texts, response.data):
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=data.embedding,
                    model=self.model_name,
                    metadata={
                        "dimension": self._dimension,
                        "usage": response.usage.model_dump(),
                    },
                )
            )

        return results

    def get_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Embedding dimension
        """
        return self._dimension

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name
