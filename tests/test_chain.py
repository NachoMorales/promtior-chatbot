import os

import pytest
from langchain_core.documents import Document

from app.ingest import create_vectorstore, get_embeddings

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@requires_openai
def test_vectorstore_creation():
    embeddings = get_embeddings()
    chunks = [
        Document(page_content="Promtior was founded in May 2023.", metadata={"source": "test"}),
        Document(page_content="Promtior offers consulting services.", metadata={"source": "test"}),
    ]
    vectorstore = create_vectorstore(chunks, embeddings)
    results = vectorstore.similarity_search("When was Promtior founded?", k=1)
    assert len(results) == 1
    assert "2023" in results[0].page_content


@requires_openai
def test_retriever_returns_relevant_docs():
    embeddings = get_embeddings()
    chunks = [
        Document(page_content="Promtior was founded in May 2023.", metadata={"source": "test"}),
        Document(
            page_content="Promtior offers technology and organizational consulting.",
            metadata={"source": "test"},
        ),
        Document(
            page_content="The weather is sunny today.",
            metadata={"source": "irrelevant"},
        ),
    ]
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("What services does Promtior offer?")
    assert len(results) == 2
    contents = " ".join(r.page_content for r in results)
    assert "consulting" in contents
