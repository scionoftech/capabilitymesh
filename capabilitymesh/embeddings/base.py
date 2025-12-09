"""Base embedder abstraction for semantic search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    embedding: List[float]
    model: str
    metadata: Optional[dict] = None


class Embedder(ABC):
    """Abstract base class for text embedders.

    Embedders convert text into vector representations for semantic similarity
    comparison.
    """

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text into a vector.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with the embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        pass

    async def similarity(
        self, text1: str, text2: str
    ) -> float:
        """Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        results = await self.embed_batch([text1, text2])
        return self.cosine_similarity(
            results[0].embedding, results[1].embedding
        )

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1.0 and 1.0 (typically 0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model.

        Returns:
            Model name
        """
        pass
