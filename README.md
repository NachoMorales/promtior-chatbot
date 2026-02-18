# Promtior RAG Chatbot

A chatbot assistant that uses the RAG (Retrieval Augmented Generation) architecture to answer questions about Promtior, built with LangChain and LangServe.

## Tech Stack

- **LLM**: OpenAI GPT-5 Nano
- **Embeddings**: OpenAI Embeddings
- **Vector Store**: FAISS (with disk persistence)
- **Retrieval**: MMR (Maximum Marginal Relevance)
- **Framework**: LangChain LCEL + LangServe (FastAPI)
- **Deploy**: AWS EC2 + Docker
- **CI/CD**: GitHub Actions (lint → test → deploy)

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/NachoMorales/promtior-chatbot.git
cd promtior-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
uvicorn app.server:app --reload
```

- **Chat UI**: http://localhost:8000
- **LangServe Playground**: http://localhost:8000/chat/playground
- **API Endpoint**: POST http://localhost:8000/chat/invoke

### Run with Docker

```bash
docker build -t promtior-chatbot .
docker run -p 8000:8000 --env-file .env promtior-chatbot
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
ruff check .

# Run tests
pytest tests/ -v
```

## CI/CD

On every push to `main`, GitHub Actions runs:

1. **Lint** — `ruff check .`
2. **Test** — `pytest tests/ -v`
3. **Deploy** — SSH to EC2, pull latest code, rebuild and restart Docker container

## Documentation

See the [doc/](doc/) folder for the project overview and component diagram.
