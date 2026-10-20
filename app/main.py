"""Marhba FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embed import get_model
from app.rag import answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedding model at startup so the first /ask request
    # doesn't pay the model-load cost (or trigger a download timeout).
    print("Warming up embedding model...")
    get_model()
    print("Embedding model ready.")
    yield


app = FastAPI(
    title="Marhba",
    description="Algerian travel regulation RAG",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    refused: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    return await answer(req.question)