"""Debug script to inspect the RAG pipeline step by step."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.config import RETRIEVER_K, VECTORSTORE_PATH  # noqa: E402
from app.ingest import get_vectorstore  # noqa: E402


def inspect_vectorstore():
    """Show all documents stored in the vectorstore."""
    print("=" * 80)
    print("STEP 1: VECTORSTORE CONTENTS")
    print(f"Cache path: {VECTORSTORE_PATH}")
    print(f"Cache exists: {VECTORSTORE_PATH.exists()}")
    print("=" * 80)

    vectorstore = get_vectorstore()
    all_docs = vectorstore.docstore._dict

    print(f"\nTotal chunks stored: {len(all_docs)}\n")

    for i, (doc_id, doc) in enumerate(all_docs.items()):
        source = doc.metadata.get("source", "unknown")
        print(f"--- Chunk {i + 1} (id: {doc_id[:8]}...) ---")
        print(f"Source: {source}")
        print(f"Content ({len(doc.page_content)} chars):")
        print(doc.page_content[:300])
        if len(doc.page_content) > 300:
            print("...")
        print()


def simulate_query(question: str):
    """Show exactly what the LLM receives for a given question."""
    print("=" * 80)
    print("STEP 2: SIMULATING QUERY")
    print(f"Question: {question}")
    print("=" * 80)

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    print(f"\nRetrieving top {RETRIEVER_K} chunks...\n")
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        print(f"--- Retrieved Chunk {i + 1} (score ranked) ---")
        print(f"Source: {source}")
        print(f"Content: {doc.page_content[:200]}...")
        print()

    print("=" * 80)
    print("STEP 3: FULL PROMPT SENT TO LLM")
    print("=" * 80)
    prompt = (
        "You are a helpful assistant for Promtior, a technology and "
        "organizational consulting company.\n"
        "Answer the question based only on the following context. "
        "If the context does not contain enough information, "
        "say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    print(prompt)


if __name__ == "__main__":
    inspect_vectorstore()

    questions = sys.argv[1:] or [
        "What services does Promtior offer?",
        "When was the company founded?",
    ]

    for q in questions:
        print("\n\n")
        simulate_query(q)
