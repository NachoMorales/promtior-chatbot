from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import OPENAI_MODEL, RETRIEVER_K
from app.ingest import get_vectorstore

PROMPT_TEMPLATE = """\
You are a helpful assistant for Promtior, a technology and \
organizational consulting company.
Answer the question based only on the following context. \
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    """Build and return the RAG chain."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=OPENAI_MODEL)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
