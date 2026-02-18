# Promtior RAG Chatbot - Project Overview

## Approach

The challenge was to build a chatbot that can answer questions about Promtior using the RAG (Retrieval Augmented Generation) architecture. My approach was to keep the solution simple, well-structured, and production-ready while leveraging LangChain's LCEL (LangChain Expression Language) for a clean, composable chain.

## Implementation Logic

The solution follows a standard RAG pipeline:

1. **Data Ingestion**: Documents are loaded from two sources — multiple pages of the Promtior website (home, services, use cases) via web scraping, and the provided PDF presentation. Web content is cleaned to remove navigation noise, and PDF pages containing only test instructions are filtered out. Documents are then split into manageable chunks using a recursive character text splitter.

2. **Embedding & Storage**: Each chunk is converted into a vector embedding using OpenAI's embedding model and stored in a FAISS vector store, persisted to disk to avoid recomputing embeddings on restart.

3. **Retrieval**: When a user asks a question, the query is embedded and the most relevant document chunks are retrieved using MMR (Maximum Marginal Relevance), which balances relevance with diversity to pull context from different sources.

4. **Generation**: The retrieved context and the user's question are passed to OpenAI GPT-5 Nano with a prompt that instructs the model to answer based only on the provided context.

5. **Serving**: The chain is exposed as an API endpoint via LangServe (built on FastAPI), which also provides a built-in playground for testing. A simple HTML/JS frontend provides the chat interface.

## Key Technical Decisions

- **OpenAI GPT-5 Nano**: Chosen for being the latest cost-effective model from OpenAI, with strong performance, low latency, and very low cost ($0.05/1M input tokens) — ideal for a Q&A chatbot.
- **OpenAI Embeddings**: Consistent provider for both LLM and embeddings simplifies the architecture and API key management.
- **FAISS**: Lightweight, fast vector store with disk persistence — ideal for a small document corpus without needing a managed database. The vectorstore is cached to disk to avoid recomputing embeddings on each restart.
- **MMR Retrieval**: Maximum Marginal Relevance balances relevance and diversity, ensuring the retrieved chunks come from different sources (website pages, PDF) rather than repeating similar content.
- **LCEL**: LangChain's Expression Language for building the chain, providing a clean, composable, and type-safe pipeline.

## Challenges and Solutions

1. **Web scraping quality**: The Promtior website content extracted via `WebBaseLoader` includes navigation elements and other noise. A cleaning step removes common patterns (navbar, footer, "top of page"), and the PDF is filtered to exclude test instruction pages. Additionally, MMR retrieval ensures diverse chunk selection across sources.

2. **No Python background**: This project was implemented in Python (my primary stack is TypeScript/Node.js) to align with the LangChain/LangServe ecosystem and demonstrate ability to quickly adapt to new languages and frameworks.

## Component Diagram

See `component-diagram.png` in this directory.