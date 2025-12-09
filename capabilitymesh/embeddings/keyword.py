"""Keyword-based embedder using TF-IDF (no ML dependencies)."""

import math
import re
from collections import Counter
from typing import Dict, List, Set

from .base import Embedder, EmbeddingResult


class KeywordEmbedder(Embedder):
    """Simple keyword-based embedder using TF-IDF.

    This embedder doesn't require any ML libraries and works with pure Python.
    It's suitable for basic semantic matching when ML models aren't available.

    The approach:
    1. Build a vocabulary from all seen texts
    2. Use TF-IDF to create sparse vectors
    3. Normalize to unit vectors for cosine similarity

    Limitations:
    - Not as accurate as ML-based embedders
    - Requires building vocabulary over time
    - Sparse vectors (mostly zeros)
    """

    def __init__(self, dimension: int = 512):
        """Initialize keyword embedder with fixed dimension.

        Uses feature hashing to map tokens to a fixed-size vector space.
        This ensures consistent embedding dimensions across all texts.

        Args:
            dimension: Fixed dimension for embeddings (default: 512)
        """
        self.dimension = dimension
        self.idf: Dict[str, float] = {}
        self.document_count = 0

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text using TF-IDF.

        Updates IDF scores with this text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with TF-IDF vector
        """
        # Update IDF scores with this text
        self._update_vocabulary([text])

        # Calculate TF-IDF
        embedding = self._tfidf_vector(text)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model="keyword-tfidf",
            metadata={"dimension": self.dimension},
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts using TF-IDF.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        # Update vocabulary with all texts
        self._update_vocabulary(texts)

        # Calculate TF-IDF for each text
        results = []
        for text in texts:
            embedding = self._tfidf_vector(text)
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model="keyword-tfidf",
                    metadata={"dimension": self.dimension},
                )
            )

        return results

    def get_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Fixed embedding dimension
        """
        return self.dimension

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return "keyword-tfidf"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _hash_token(self, token: str) -> int:
        """Hash a token to an index in the fixed-size vector.

        Args:
            token: Token to hash

        Returns:
            Index in the range [0, dimension)
        """
        # Use Python's built-in hash and modulo to map to dimension
        return hash(token) % self.dimension

    def _update_vocabulary(self, texts: List[str]) -> None:
        """Update IDF scores with new texts.

        Args:
            texts: List of texts to update IDF scores
        """
        # Tokenize all texts and count document frequencies
        document_term_freq: Dict[str, int] = {}

        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)

            # Count document frequency
            for token in unique_tokens:
                document_term_freq[token] = document_term_freq.get(token, 0) + 1

        # Update IDF (inverse document frequency)
        self.document_count += len(texts)
        for token, doc_freq in document_term_freq.items():
            # IDF = log((N + 1) / (df + 1)) + 1
            # Adding 1 ensures IDF is never 0, even for the first document
            self.idf[token] = math.log(
                (self.document_count + 1) / (doc_freq + 1)
            ) + 1.0

    def _tfidf_vector(self, text: str) -> List[float]:
        """Calculate TF-IDF vector for text using feature hashing.

        Args:
            text: Text to vectorize

        Returns:
            TF-IDF vector (normalized) with fixed dimension
        """
        # Initialize vector with zeros
        vector = [0.0] * self.dimension

        # Tokenize and count term frequencies
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        term_freq = Counter(tokens)

        # Calculate TF-IDF using feature hashing
        for token, freq in term_freq.items():
            # Hash token to get index
            idx = self._hash_token(token)

            # TF = freq / total_tokens
            tf = freq / len(tokens)

            # IDF (use default of 1.0 if not seen before)
            idf = self.idf.get(token, 1.0)

            # TF-IDF (accumulate if hash collision occurs)
            vector[idx] += tf * idf

        # Normalize to unit vector
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector
