"""
Servidor simples para o summarizer: carrega o modelo uma vez e atende requisições.
Uso: python -m uvicorn tests.summarizer_server:app --host 0.0.0.0 --port 8003
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = "Sachin21112004/distilbart-news-summarizer"

tokenizer = None
model = None


def load_model():
    global tokenizer, model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("Model loaded successfully")


def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    print("Shutting down summarizer server")


app = FastAPI(
    title="Summarizer API",
    version="1.0.0",
    description="Summarize text using DistilBART (model loaded once at startup)",
    lifespan=lifespan,
)


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(150, ge=10, le=300, description="Max summary length")
    min_length: int = Field(40, ge=5, le=150, description="Min summary length")


class SummarizeResponse(BaseModel):
    summary: str


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    summary = summarize(req.text, max_length=req.max_length, min_length=req.min_length)
    return SummarizeResponse(summary=summary)


@app.get("/health")
async def health():
    return {
        "status": "ok" if (tokenizer is not None and model is not None) else "loading",
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn
    # Run from project root: python -m uvicorn tests.summarizer_server:app --port 8003
    uvicorn.run(
        "tests.summarizer_server:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
    )
