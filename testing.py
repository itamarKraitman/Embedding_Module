import pytest
from unittest.mock import patch
from embeddings_module import EmbeddingsModule
import torch

@pytest.fixture
def embeddings_module():
    return EmbeddingsModule(splitting_strategy=0, chunk_size=100, overlap=50)

# Unit Tests

def test_initialization(embeddings_module):
    assert embeddings_module.splitting_strategy == 0
    assert embeddings_module.chunk_size == 100
    assert embeddings_module.overlap == 50
    assert embeddings_module.device in [torch.device("cuda"), torch.device("cpu")]
    assert embeddings_module.tokenizer is not None
    assert embeddings_module.model is not None


def test_split_fixed_overlap(embeddings_module):
    text = "word " * 200  # A text of 200 words
    chunks = embeddings_module._split_fixed_overlap(text)
    assert len(chunks) == 4
    assert len(chunks[0].split()) == 100

def test_split_by_sentences(embeddings_module):
    text = "This is a sentence. Another sentence. And another one."
    chunks = embeddings_module._split_by_sentences(text)
    assert len(chunks) == 3
    assert chunks[0] == "This is a sentence"


def test_split_by_paragraphs(embeddings_module):
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = embeddings_module._split_by_paragraphs(text)
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph one."


def test_split_text_invalid_strategy():
    with pytest.raises(ValueError):
        EmbeddingsModule(splitting_strategy="invalid_strategy")._split_text("Sample text")


def test_embed_text(embeddings_module):
    text = "This is a test text."
    embedding = embeddings_module._embed_text(text)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 1  # Embedding should be a 1D vector


# Integration Tests
@patch("embeddings_module.EmbeddingsModule._read_pdf")
def test_process_document_pdf(mock_read_pdf, embeddings_module):
    mock_read_pdf.return_value = "Sample text from PDF."
    embeddings_module.process_document("sample.pdf", file_format="pdf")
    assert len(embeddings_module.chunk_embeddings) > 0


@patch("embeddings_module.EmbeddingsModule._read_docx")
def test_process_document_docx(mock_read_docx, embeddings_module):
    mock_read_docx.return_value = "Sample text from DOCX."
    embeddings_module.process_document("sample.docx", file_format="docx")
    assert len(embeddings_module.chunk_embeddings) > 0


def test_process_document_invalid_format(embeddings_module):
    with pytest.raises(ValueError):
        embeddings_module.process_document("sample.txt", file_format="txt")


@patch("embeddings_module.EmbeddingsModule._embed_text")
def test_retrieve(mock_embed_text, embeddings_module):
    # Mock embeddings
    mock_embed_text.side_effect = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]), torch.tensor([0.15, 0.25])]
    embeddings_module.chunk_embeddings = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]

    results = embeddings_module.retrieve("Test prompt", top_k=1)
    assert len(results) == 1
    assert results[0][0] == 0  # Most similar index


# Edge Case Tests
def test_empty_document(embeddings_module):
    empty_text = ""
    with patch("embeddings_module.EmbeddingsModule._read_pdf", return_value=empty_text):
        embeddings_module.process_document("empty.pdf", file_format="pdf")
        assert len(embeddings_module.chunk_embeddings) == 0


def test_large_document(embeddings_module):
    large_text = "word " * 10000  # Simulate a very large document
    with patch("embeddings_module.EmbeddingsModule._read_pdf", return_value=large_text):
        embeddings_module.process_document("large.pdf", file_format="pdf")
        assert len(embeddings_module.chunk_embeddings) > 0


# Performance Tests
def test_retrieve_performance(embeddings_module):
    num_chunks = 1000
    embedding_dim = 128
    embeddings_module.chunk_embeddings = [torch.rand(embedding_dim) for _ in range(num_chunks)]
    query = "Performance test query."

    with patch("embeddings_module.EmbeddingsModule._embed_text", return_value=torch.rand(embedding_dim)):
        results = embeddings_module.retrieve(query, top_k=10)
        assert len(results) == 10
