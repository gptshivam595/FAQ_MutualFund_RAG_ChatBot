from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import AskRequest, AskResponse, SuggestionResponse
from app.services.faq_service import ask_question, get_suggestions


router = APIRouter(prefix="/api", tags=["faq"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return ask_question(request.query.strip())


@router.get("/suggestions", response_model=SuggestionResponse)
def suggestions() -> SuggestionResponse:
    return get_suggestions()
