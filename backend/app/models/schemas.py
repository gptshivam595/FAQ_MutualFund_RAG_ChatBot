from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Facts-only mutual fund question.")


class ChatResponse(BaseModel):
    status: Literal["answer", "refusal", "needs_scope"]
    answer: str
    source_label: str
    source_url: str
    last_updated_from_sources: str


class AskResponse(BaseModel):
    status: Literal["success", "refused", "needs_scope"]
    answer: str
    source_label: str
    source_url: str
    last_updated: str
    suggested_questions: list[str] = []


class SuggestionResponse(BaseModel):
    questions: list[str]


class HealthResponse(BaseModel):
    status: str
