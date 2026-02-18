from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langserve import add_routes

from app.chain import get_chain
from app.config import STATIC_DIR

app = FastAPI(title="Promtior Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


chain = get_chain()
add_routes(app, chain, path="/chat")

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
