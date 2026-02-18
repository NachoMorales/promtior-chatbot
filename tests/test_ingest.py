import os

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH
from app.ingest import get_embeddings, split_documents

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def test_pdf_exists():
    assert PDF_PATH.exists(), f"PDF not found at {PDF_PATH}"


def test_split_documents_creates_chunks():
    docs = [Document(page_content="word " * 500, metadata={"source": "test"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1


def test_split_documents_respects_config():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = [Document(page_content="word " * 500, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + 50


@requires_openai
def test_embeddings_model_loads():
    embeddings = get_embeddings()
    result = embeddings.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) > 0
