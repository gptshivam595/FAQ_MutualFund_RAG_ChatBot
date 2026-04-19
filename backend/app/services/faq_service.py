from __future__ import annotations

from functools import lru_cache

from app.models.schemas import AskResponse, ChatResponse, SuggestionResponse
from app.services.knowledge_base import FAQBot
from app.services.openai_support import (
    DEFAULT_SUGGESTIONS,
    generate_question_suggestions,
    improve_unavailable_answer,
)


@lru_cache(maxsize=1)
def get_knowledge_base() -> FAQBot:
    return FAQBot()


def _map_status(response: ChatResponse) -> str:
    if response.status == "refusal":
        return "refused"
    if response.status == "needs_scope":
        return "needs_scope"
    return "success"


def ask_question(query: str) -> AskResponse:
    response = get_knowledge_base().answer(query)
    answer = response.answer
    suggested_questions = DEFAULT_SUGGESTIONS

    if response.status != "answer":
        answer, suggested_questions = improve_unavailable_answer(query, response.answer)

    return AskResponse(
        status=_map_status(response),  # type: ignore[arg-type]
        answer=answer,
        source_label=response.source_label,
        source_url=response.source_url,
        last_updated=response.last_updated_from_sources,
        suggested_questions=suggested_questions,
    )


def get_suggestions() -> SuggestionResponse:
    return SuggestionResponse(questions=generate_question_suggestions())
