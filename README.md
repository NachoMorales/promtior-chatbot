# Promtior RAG Chatbot

A chatbot assistant that uses the RAG (Retrieval Augmented Generation) architecture to answer questions about Promtior, built with LangChain and LangServe.

## Tech Stack

- **LLM**: OpenAI GPT-5 Nano
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: FAISS
- **Framework**: LangChain + LangServe (FastAPI)
- **Deploy**: AWS EC2 + Docker
- **CI/CD**: GitHub Actions (lint + test + docker build)

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/promtior-chatbot-2.git
cd promtior-chatbot-2

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

## Documentation

See the [doc/](doc/) folder for the project overview and component diagram.
