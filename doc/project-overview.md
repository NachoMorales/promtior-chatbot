# Promtior RAG Chatbot - Project Overview

## Approach

My goal was to build a simple but solid RAG chatbot that actually gives useful answers about Promtior. I focused on getting the fundamentals right: clean data ingestion, good retrieval, and a straightforward serving layer. I used LangChain's LCEL to keep the chain declarative and easy to follow.

## Implementation Logic

The pipeline works in two phases:

**Data ingestion (runs once at startup):**
I load content from three pages of the Promtior website (home, services, use cases) and the PDF presentation. The raw web content comes with a lot of noise (navbar, footer, "top of page" elements), so I added a cleaning step that strips those out before splitting. For the PDF, I filter out the pages that are just test instructions (only the pages with actual company info get indexed). Everything gets split into chunks of 1000 characters with 200 overlap, embedded with OpenAI's embedding model, and stored in a FAISS vector store that persists to disk so I don't have to recompute embeddings on every restart.

**Query flow (on each user question):**
When a user asks something, the question gets embedded and I run an MMR (Maximum Marginal Relevance) search against the vector store. I chose MMR over basic similarity search because with a small corpus like this, plain similarity tends to return very similar chunks from the same source. MMR forces diversity, so the LLM gets context from different pages instead of four chunks that say roughly the same thing. The top 6 chunks plus the question go into a prompt that tells GPT-5 Nano to answer only based on the provided context. The response comes back through LangServe's API endpoint and the frontend displays it.

## Key Technical Decisions

- **OpenAI GPT-5 Nano**: Latest cost-effective model from OpenAI. It's perfect for a Q&A chatbot, fast, cheap, and good enough for this use case.
- **OpenAI Embeddings**: Using the same provider for LLM and embeddings keeps things simple.
- **FAISS with disk cache**: Lightweight vector store that doesn't need a separate database service. The disk persistence avoids redundant API calls to recompute embeddings.
- **MMR Retrieval**: Balances relevance and diversity. This was key to getting good answers, without it, the retriever kept returning similar PDF chunks instead of the actual service descriptions from the website.
- **LCEL**: LangChain's Expression Language lets me define the entire chain in a few lines. It's composable and easy to debug.

## CI/CD & Deployment

I set up a full CI/CD pipeline with GitHub Actions: on every push to `main`, it runs linting (ruff), tests (pytest), and if everything passes, deploys automatically to an AWS EC2 instance via SSH, pulling the latest code and rebuilding the Docker container. This way the deployed app is always in sync with the repo.

## Challenges and Solutions

1. **Web scraping quality**: The biggest challenge was getting clean, useful data from the website. `WebBaseLoader` grabs everything including navigation elements, which pollutes the vector store and hurts retrieval quality. I solved this with a regex-based cleaning step and by switching from similarity search to MMR. I also realized I needed to scrape multiple pages (not just the homepage) to actually capture the service descriptions.

2. **PDF noise**: The presentation PDF mixes company information with test instructions. Including everything would confuse the retriever. Filtering to only the relevant pages fixed this.

3. **Working in Python**: My primary stack is TypeScript/Node.js, so this was my first real project in Python. I chose Python intentionally because the LangChain/LangServe ecosystem is Python-first and I wanted to demonstrate that I can pick up new languages quickly. The fundamentals (clean architecture, testing, CI/CD) translate across languages.

## Component Diagram

See `component-diagram.png` in this directory.
