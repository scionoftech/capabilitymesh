"""Local embedder using sentence-transformers."""

from typing import List

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .base import Embedder, EmbeddingResult


class LocalEmbedder(Embedder):
    """Local embedding using sentence-transformers.

    Uses pre-trained transformer models that run locally without API calls.
    Provides high-quality semantic embeddings with zero runtime costs.

    Default model: 'all-MiniLM-L6-v2'
    - Fast inference (small model)
    - Good quality embeddings
    - 384 dimensions
    - ~80MB download

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedder.

        Args:
            model_name: Name of sentence-transformers model

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for LocalEmbedder. "
                "Install it with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # sentence-transformers is sync, but we wrap in async interface
        embedding = self.model.encode(text, convert_to_numpy=True)

        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            metadata={"dimension": self._dimension},
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts (batched for efficiency).

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        # Batch encoding is much faster
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        results = []
        for text, embedding in zip(texts, embeddings):
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=embedding.tolist(),
                    model=self.model_name,
                    metadata={"dimension": self._dimension},
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
