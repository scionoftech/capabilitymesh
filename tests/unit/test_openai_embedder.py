"""Tests for OpenAIEmbedder."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capabilitymesh.embeddings.base import EmbeddingResult


class TestOpenAIEmbedderImportError:
    """Test OpenAIEmbedder when openai library is not available."""

    def test_import_error_when_openai_not_available(self):
        """Test that ImportError is raised when openai is not installed."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", False):
            from capabilitymesh.embeddings.openai import OpenAIEmbedder

            with pytest.raises(ImportError) as exc_info:
                OpenAIEmbedder(api_key="test-key")

            assert "openai is required" in str(exc_info.value)
            assert "pip install openai" in str(exc_info.value)


class TestOpenAIEmbedderInitialization:
    """Test OpenAIEmbedder initialization."""

    def test_api_key_required_error(self):
        """Test that ValueError is raised when API key is not provided."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI"):
                with patch.dict(os.environ, {}, clear=True):
                    from capabilitymesh.embeddings.openai import OpenAIEmbedder

                    with pytest.raises(ValueError) as exc_info:
                        OpenAIEmbedder()

                    assert "API key is required" in str(exc_info.value)
                    assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_api_key_from_parameter(self):
        """Test initialization with API key from parameter."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI") as mock_client:
                from capabilitymesh.embeddings.openai import OpenAIEmbedder

                embedder = OpenAIEmbedder(api_key="test-key-123")

                assert embedder.api_key == "test-key-123"
                mock_client.assert_called_once_with(api_key="test-key-123")

    def test_api_key_from_environment(self):
        """Test initialization with API key from environment variable."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI") as mock_client:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
                    from capabilitymesh.embeddings.openai import OpenAIEmbedder

                    embedder = OpenAIEmbedder()

                    assert embedder.api_key == "env-key-456"
                    mock_client.assert_called_once_with(api_key="env-key-456")

    def test_parameter_overrides_environment(self):
        """Test that parameter API key overrides environment variable."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI") as mock_client:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                    from capabilitymesh.embeddings.openai import OpenAIEmbedder

                    embedder = OpenAIEmbedder(api_key="param-key")

                    assert embedder.api_key == "param-key"
                    mock_client.assert_called_once_with(api_key="param-key")

    def test_unsupported_model_error(self):
        """Test that ValueError is raised for unsupported model."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI"):
                from capabilitymesh.embeddings.openai import OpenAIEmbedder

                with pytest.raises(ValueError) as exc_info:
                    OpenAIEmbedder(model_name="invalid-model", api_key="test-key")

                assert "Unsupported model" in str(exc_info.value)
                assert "invalid-model" in str(exc_info.value)

    def test_default_model(self):
        """Test that default model is text-embedding-3-small."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI"):
                from capabilitymesh.embeddings.openai import OpenAIEmbedder

                embedder = OpenAIEmbedder(api_key="test-key")

                assert embedder.model_name == "text-embedding-3-small"
                assert embedder._dimension == 1536

    def test_supported_models_dimensions(self):
        """Test all supported models have correct dimensions."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch("capabilitymesh.embeddings.openai.AsyncOpenAI"):
                from capabilitymesh.embeddings.openai import OpenAIEmbedder

                models = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536,
                }

                for model_name, expected_dim in models.items():
                    embedder = OpenAIEmbedder(model_name=model_name, api_key="test-key")
                    assert embedder.get_dimension() == expected_dim


class TestOpenAIEmbedderWithMock:
    """Test OpenAIEmbedder with mocked OpenAI client."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock AsyncOpenAI client."""
        mock_client = MagicMock()

        # Mock embeddings.create response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.model_dump.return_value = {"total_tokens": 10}

        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        return mock_client

    @pytest.fixture
    def embedder(self, mock_openai_client):
        """Create an OpenAIEmbedder with mocked client."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch(
                "capabilitymesh.embeddings.openai.AsyncOpenAI",
                return_value=mock_openai_client
            ):
                from capabilitymesh.embeddings.openai import OpenAIEmbedder
                return OpenAIEmbedder(api_key="test-key")

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder, mock_openai_client):
        """Test embedding a single text."""
        result = await embedder.embed("Hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "Hello world"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "text-embedding-3-small"
        assert result.metadata["dimension"] == 1536
        assert "usage" in result.metadata

        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="Hello world",
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder, mock_openai_client):
        """Test embedding multiple texts in batch."""
        # Setup mock for batch response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9]),
        ]
        mock_response.usage.model_dump.return_value = {"total_tokens": 30}
        mock_openai_client.embeddings.create.return_value = mock_response

        texts = ["First text", "Second text", "Third text"]
        results = await embedder.embed_batch(texts)

        assert len(results) == 3

        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        for i, (text, result) in enumerate(zip(texts, results)):
            assert isinstance(result, EmbeddingResult)
            assert result.text == text
            assert result.embedding == expected_embeddings[i]
            assert result.model == "text-embedding-3-small"
            assert result.metadata["dimension"] == 1536

        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_large(self, embedder, mock_openai_client):
        """Test embedding more than 2048 texts (chunking)."""
        # Create a large batch (3000 texts)
        texts = [f"Text {i}" for i in range(3000)]

        # Setup mock to return appropriate responses
        def create_mock_response(count):
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(count)]
            mock_response.usage.model_dump.return_value = {"total_tokens": count * 10}
            return mock_response

        # Mock will be called twice: first with 2048, then with 952
        mock_openai_client.embeddings.create.side_effect = [
            create_mock_response(2048),
            create_mock_response(952),
        ]

        results = await embedder.embed_batch(texts)

        assert len(results) == 3000
        assert mock_openai_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedder, mock_openai_client):
        """Test embedding empty text."""
        result = await embedder.embed("")

        assert result.text == ""
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, embedder, mock_openai_client):
        """Test embedding empty list."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_response.usage.model_dump.return_value = {"total_tokens": 0}
        mock_openai_client.embeddings.create.return_value = mock_response

        results = await embedder.embed_batch([])

        assert results == []

    def test_get_dimension(self, embedder):
        """Test getting embedding dimension."""
        assert embedder.get_dimension() == 1536

    def test_get_model_name(self, embedder):
        """Test getting model name."""
        assert embedder.get_model_name() == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_special_characters(self, embedder, mock_openai_client):
        """Test embedding text with special characters."""
        special_text = "Hello! @#$%^&*() 你好 émoji 🚀"
        result = await embedder.embed(special_text)

        assert result.text == special_text
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_models(self, mock_openai_client):
        """Test using different embedding models."""
        with patch("capabilitymesh.embeddings.openai.OPENAI_AVAILABLE", True):
            with patch(
                "capabilitymesh.embeddings.openai.AsyncOpenAI",
                return_value=mock_openai_client
            ):
                from capabilitymesh.embeddings.openai import OpenAIEmbedder

                models = [
                    ("text-embedding-3-small", 1536),
                    ("text-embedding-3-large", 3072),
                    ("text-embedding-ada-002", 1536),
                ]

                for model_name, expected_dim in models:
                    embedder = OpenAIEmbedder(model_name=model_name, api_key="test-key")
                    assert embedder.model_name == model_name
                    assert embedder.get_dimension() == expected_dim

    @pytest.mark.asyncio
    async def test_api_error_propagation(self, embedder, mock_openai_client):
        """Test that API errors are properly propagated."""
        # Mock an API error
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            await embedder.embed("Test text")

        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_usage_metadata_included(self, embedder, mock_openai_client):
        """Test that usage metadata is included in results."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.model_dump.return_value = {
            "total_tokens": 50,
            "prompt_tokens": 50,
        }
        mock_openai_client.embeddings.create.return_value = mock_response

        result = await embedder.embed("Test text")

        assert "usage" in result.metadata
        assert result.metadata["usage"]["total_tokens"] == 50
        assert result.metadata["usage"]["prompt_tokens"] == 50

    @pytest.mark.asyncio
    async def test_embed_batch_exactly_2048(self, embedder, mock_openai_client):
        """Test embedding exactly 2048 texts (boundary case)."""
        texts = [f"Text {i}" for i in range(2048)]

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(2048)]
        mock_response.usage.model_dump.return_value = {"total_tokens": 20480}
        mock_openai_client.embeddings.create.return_value = mock_response

        results = await embedder.embed_batch(texts)

        assert len(results) == 2048
        # Should be called once (not chunked)
        assert mock_openai_client.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch_2049(self, embedder, mock_openai_client):
        """Test embedding 2049 texts (just over the limit, should chunk)."""
        texts = [f"Text {i}" for i in range(2049)]

        def create_mock_response(count):
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(count)]
            mock_response.usage.model_dump.return_value = {"total_tokens": count * 10}
            return mock_response

        mock_openai_client.embeddings.create.side_effect = [
            create_mock_response(2048),
            create_mock_response(1),
        ]

        results = await embedder.embed_batch(texts)

        assert len(results) == 2049
        # Should be called twice (chunked)
        assert mock_openai_client.embeddings.create.call_count == 2
