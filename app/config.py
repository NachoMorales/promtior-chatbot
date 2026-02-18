import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 6

PROMTIOR_URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
]
PDF_PATH = DATA_DIR / "promtior-presentation.pdf"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"
