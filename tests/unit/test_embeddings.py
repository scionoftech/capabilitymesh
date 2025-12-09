"""Unit tests for embedders."""

import pytest

from capabilitymesh.embeddings import (
    KeywordEmbedder,
    Embedder,
    auto_select_embedder,
)


@pytest.fixture
def keyword_embedder():
    """Create a keyword embedder."""
    return KeywordEmbedder(dimension=128)


@pytest.mark.asyncio
async def test_keyword_embedder_single_text(keyword_embedder):
    """Test embedding a single text."""
    result = await keyword_embedder.embed("hello world")

    assert result.text == "hello world"
    assert len(result.embedding) == 128
    assert result.model == "keyword-tfidf"
    assert isinstance(result.embedding, list)
    assert all(isinstance(x, float) for x in result.embedding)


@pytest.mark.asyncio
async def test_keyword_embedder_batch(keyword_embedder):
    """Test embedding multiple texts in batch."""
    texts = ["hello world", "goodbye world", "machine learning"]

    results = await keyword_embedder.embed_batch(texts)

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.text == texts[i]
        assert len(result.embedding) == 128


@pytest.mark.asyncio
async def test_keyword_embedder_similarity(keyword_embedder):
    """Test similarity calculation."""
    # Embed related texts
    results = await keyword_embedder.embed_batch([
        "machine learning algorithms",
        "deep learning neural networks",
        "cooking recipes dinner",
    ])

    # Calculate similarities
    sim_ml_dl = keyword_embedder.cosine_similarity(
        results[0].embedding, results[1].embedding
    )
    sim_ml_cooking = keyword_embedder.cosine_similarity(
        results[0].embedding, results[2].embedding
    )

    # ML and DL should be more similar than ML and cooking
    assert sim_ml_dl > sim_ml_cooking
    assert 0 <= sim_ml_dl <= 1
    assert 0 <= sim_ml_cooking <= 1


@pytest.mark.asyncio
async def test_keyword_embedder_identical_texts(keyword_embedder):
    """Test that identical texts have similarity 1.0."""
    result1 = await keyword_embedder.embed("test text")
    result2 = await keyword_embedder.embed("test text")

    similarity = keyword_embedder.cosine_similarity(
        result1.embedding, result2.embedding
    )

    assert similarity == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_keyword_embedder_empty_text(keyword_embedder):
    """Test embedding empty text."""
    result = await keyword_embedder.embed("")

    assert len(result.embedding) == 128
    # Empty text should have zero vector
    assert all(x == 0.0 for x in result.embedding)


@pytest.mark.asyncio
async def test_keyword_embedder_dimension(keyword_embedder):
    """Test get_dimension method."""
    assert keyword_embedder.get_dimension() == 128


@pytest.mark.asyncio
async def test_keyword_embedder_model_name(keyword_embedder):
    """Test get_model_name method."""
    assert keyword_embedder.get_model_name() == "keyword-tfidf"


@pytest.mark.asyncio
async def test_keyword_embedder_idf_updates():
    """Test that IDF scores update correctly."""
    embedder = KeywordEmbedder(dimension=64)

    # First embedding - IDF scores are built
    result1 = await embedder.embed("machine learning")
    nonzero1 = sum(1 for x in result1.embedding if x != 0)

    # Second embedding - IDF scores updated
    result2 = await embedder.embed("deep learning")
    nonzero2 = sum(1 for x in result2.embedding if x != 0)

    # Both should have non-zero embeddings
    assert nonzero1 > 0
    assert nonzero2 > 0


@pytest.mark.asyncio
async def test_keyword_embedder_consistency():
    """Test that embeddings are consistent across calls."""
    embedder = KeywordEmbedder(dimension=128)

    # Build IDF with batch
    await embedder.embed_batch(["hello world", "goodbye world"])

    # Embed same text twice
    result1 = await embedder.embed("hello")
    result2 = await embedder.embed("hello")

    # Should be identical
    assert result1.embedding == result2.embedding


@pytest.mark.asyncio
async def test_auto_select_embedder():
    """Test auto-selection of embedder."""
    embedder = auto_select_embedder()

    # Should return some embedder
    assert isinstance(embedder, Embedder)

    # Should be able to embed
    result = await embedder.embed("test text")
    assert len(result.embedding) > 0


@pytest.mark.asyncio
async def test_cosine_similarity_normalized():
    """Test cosine similarity with normalized vectors."""
    embedder = KeywordEmbedder()

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 0.0, 0.0]

    # Orthogonal vectors
    sim_12 = embedder.cosine_similarity(vec1, vec2)
    assert sim_12 == pytest.approx(0.0, abs=0.01)

    # Identical vectors
    sim_13 = embedder.cosine_similarity(vec1, vec3)
    assert sim_13 == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_cosine_similarity_error():
    """Test cosine similarity with mismatched dimensions."""
    embedder = KeywordEmbedder()

    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="same dimension"):
        embedder.cosine_similarity(vec1, vec2)


@pytest.mark.asyncio
async def test_keyword_embedder_special_characters(keyword_embedder):
    """Test embedding text with special characters."""
    result = await keyword_embedder.embed("hello! world? #test @user")

    # Should handle special characters gracefully
    assert len(result.embedding) == 128
    nonzero = sum(1 for x in result.embedding if x != 0)
    assert nonzero > 0


@pytest.mark.asyncio
async def test_keyword_embedder_case_insensitive(keyword_embedder):
    """Test that embedder is case insensitive."""
    result1 = await keyword_embedder.embed("Machine Learning")
    result2 = await keyword_embedder.embed("machine learning")

    # Should be very similar (if not identical)
    similarity = keyword_embedder.cosine_similarity(
        result1.embedding, result2.embedding
    )
    assert similarity > 0.99
