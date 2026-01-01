"""Tests for LocalEmbedder."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from capabilitymesh.embeddings.base import EmbeddingResult


class TestLocalEmbedderImportError:
    """Test LocalEmbedder when sentence-transformers is not available."""

    def test_import_error_when_sentence_transformers_not_available(self):
        """Test that ImportError is raised when sentence-transformers is not installed."""
        # Mock sentence-transformers as not available
        with patch("capabilitymesh.embeddings.local.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            from capabilitymesh.embeddings.local import LocalEmbedder

            with pytest.raises(ImportError) as exc_info:
                LocalEmbedder()

            assert "sentence-transformers is required" in str(exc_info.value)
            assert "pip install sentence-transformers" in str(exc_info.value)


class TestLocalEmbedderWithMock:
    """Test LocalEmbedder with mocked sentence-transformers."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
        return mock_model

    @pytest.fixture
    def embedder(self, mock_sentence_transformer):
        """Create a LocalEmbedder with mocked dependencies."""
        with patch("capabilitymesh.embeddings.local.SENTENCE_TRANSFORMERS_AVAILABLE", True):
            with patch(
                "capabilitymesh.embeddings.local.SentenceTransformer",
                return_value=mock_sentence_transformer
            ):
                from capabilitymesh.embeddings.local import LocalEmbedder
                return LocalEmbedder(model_name="test-model")

    @pytest.mark.asyncio
    async def test_initialization(self, mock_sentence_transformer):
        """Test LocalEmbedder initialization."""
        with patch("capabilitymesh.embeddings.local.SENTENCE_TRANSFORMERS_AVAILABLE", True):
            with patch(
                "capabilitymesh.embeddings.local.SentenceTransformer",
                return_value=mock_sentence_transformer
            ) as mock_st:
                from capabilitymesh.embeddings.local import LocalEmbedder

                embedder = LocalEmbedder(model_name="custom-model")

                assert embedder.model_name == "custom-model"
                assert embedder._dimension == 384
                mock_st.assert_called_once_with("custom-model")

    @pytest.mark.asyncio
    async def test_default_model_name(self, mock_sentence_transformer):
        """Test LocalEmbedder uses default model name."""
        with patch("capabilitymesh.embeddings.local.SENTENCE_TRANSFORMERS_AVAILABLE", True):
            with patch(
                "capabilitymesh.embeddings.local.SentenceTransformer",
                return_value=mock_sentence_transformer
            ) as mock_st:
                from capabilitymesh.embeddings.local import LocalEmbedder

                embedder = LocalEmbedder()

                assert embedder.model_name == "all-MiniLM-L6-v2"
                mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder, mock_sentence_transformer):
        """Test embedding a single text."""
        import numpy as np

        # Setup mock to return a numpy array
        mock_array = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_array

        result = await embedder.embed("Hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "Hello world"
        assert result.embedding == [0.1, 0.2, 0.3, 0.4]
        assert result.model == "test-model"
        assert result.metadata["dimension"] == 384

        mock_sentence_transformer.encode.assert_called_once_with(
            "Hello world", convert_to_numpy=True
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder, mock_sentence_transformer):
        """Test embedding multiple texts in batch."""
        import numpy as np

        # Setup mock to return multiple embeddings
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ])
        mock_sentence_transformer.encode.return_value = mock_embeddings

        texts = ["First text", "Second text", "Third text"]
        results = await embedder.embed_batch(texts)

        assert len(results) == 3

        for i, (text, result) in enumerate(zip(texts, results)):
            assert isinstance(result, EmbeddingResult)
            assert result.text == text
            assert result.embedding == mock_embeddings[i].tolist()
            assert result.model == "test-model"
            assert result.metadata["dimension"] == 384

        mock_sentence_transformer.encode.assert_called_once_with(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedder, mock_sentence_transformer):
        """Test embedding empty text."""
        import numpy as np

        mock_array = np.array([0.0, 0.0, 0.0, 0.0])
        mock_sentence_transformer.encode.return_value = mock_array

        result = await embedder.embed("")

        assert result.text == ""
        assert result.embedding == [0.0, 0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, embedder, mock_sentence_transformer):
        """Test embedding empty list of texts."""
        import numpy as np

        mock_sentence_transformer.encode.return_value = np.array([])

        results = await embedder.embed_batch([])

        assert results == []

    def test_get_dimension(self, embedder):
        """Test getting embedding dimension."""
        assert embedder.get_dimension() == 384

    def test_get_model_name(self, embedder):
        """Test getting model name."""
        assert embedder.get_model_name() == "test-model"

    @pytest.mark.asyncio
    async def test_embed_special_characters(self, embedder, mock_sentence_transformer):
        """Test embedding text with special characters."""
        import numpy as np

        mock_array = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_array

        special_text = "Hello! @#$%^&*() 你好 émoji 🚀"
        result = await embedder.embed(special_text)

        assert result.text == special_text
        mock_sentence_transformer.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_long_text(self, embedder, mock_sentence_transformer):
        """Test embedding very long text."""
        import numpy as np

        mock_array = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_array

        long_text = "word " * 1000
        result = await embedder.embed(long_text)

        assert result.text == long_text
        assert len(result.embedding) == 4

    @pytest.mark.asyncio
    async def test_embed_batch_consistency(self, embedder, mock_sentence_transformer):
        """Test that batch embedding is consistent."""
        import numpy as np

        mock_embeddings = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ])
        mock_sentence_transformer.encode.return_value = mock_embeddings

        texts = ["Text 1", "Text 2"]

        # First batch
        results1 = await embedder.embed_batch(texts)

        # Reset mock for second call
        mock_sentence_transformer.encode.return_value = mock_embeddings

        # Second batch with same texts
        results2 = await embedder.embed_batch(texts)

        # Results should be consistent
        for r1, r2 in zip(results1, results2):
            assert r1.text == r2.text
            assert r1.embedding == r2.embedding
            assert r1.model == r2.model

    @pytest.mark.asyncio
    async def test_embed_unicode_text(self, embedder, mock_sentence_transformer):
        """Test embedding Unicode text from various languages."""
        import numpy as np

        mock_array = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_array

        unicode_texts = [
            "Hello world",  # English
            "你好世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "こんにちは世界",  # Japanese
        ]

        for text in unicode_texts:
            result = await embedder.embed(text)
            assert result.text == text
            assert len(result.embedding) > 0
