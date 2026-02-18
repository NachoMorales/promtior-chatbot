import re

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
    PROMTIOR_URLS,
    VECTORSTORE_PATH,
)


def _clean_web_content(text: str) -> str:
    """Remove navigation noise from scraped web pages."""
    # Remove common navigation/footer patterns
    noise_patterns = [
        r"top of page",
        r"bottom of page",
        r"ServicesCase StudiesAbout UsCareersContact UsBlog",
        r"Privacy Policy",
        r"\n{3,}",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "\n", text, flags=re.IGNORECASE)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents():
    """Load documents from the Promtior website and PDF presentation."""
    web_loader = WebBaseLoader(PROMTIOR_URLS)
    web_docs = web_loader.load()

    for doc in web_docs:
        doc.page_content = _clean_web_content(doc.page_content)

    pdf_loader = PyPDFLoader(str(PDF_PATH))
    pdf_docs = pdf_loader.load()

    # Pages 2-3 contain info about Promtior (company history, clients).
    # Pages 0-1 are the test cover/intro, pages 4+ are test instructions.
    pdf_docs = [doc for doc in pdf_docs if doc.metadata.get("page", 0) in (2, 3)]

    return web_docs + pdf_docs


def split_documents(documents):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """Get the OpenAI embeddings model."""
    return OpenAIEmbeddings()


def create_vectorstore(chunks, embeddings):
    """Create a FAISS vectorstore from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def get_vectorstore():
    """Load the FAISS vectorstore from cache, or build and cache it."""
    embeddings = get_embeddings()

    if VECTORSTORE_PATH.exists():
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    return vectorstore
