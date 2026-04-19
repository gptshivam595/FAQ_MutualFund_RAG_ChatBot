import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import smart_answer

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App init ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for the HDFC Mutual Fund RAG chatbot.",
    version="1.0.0"
)

# ── CORS middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response schema ────────────────────────────────────────────────
class AskResponse(BaseModel):
    answer: str
    status: str

# ── Health check ───────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Main endpoint ──────────────────────────────────────────────────
@app.get("/ask", response_model=AskResponse)
def ask(
    query: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask the RAG bot."
    )
):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty or whitespace."
        )

    logger.info(f"Received query: {query}")

    try:
        answer = smart_answer(query)
        logger.info("Answer generated successfully.")
        return AskResponse(answer=answer, status="success")

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again."
        )
