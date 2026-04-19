from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
DEFAULT_SUGGESTIONS = [
    "Expense ratio of HDFC Flexi Cap Fund?",
    "How do I download a capital gains statement?",
    "What is the lock-in for HDFC ELSS Tax Saver?",
]


def _load_env_files() -> None:
    for env_path in (BACKEND_DIR / ".env", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_files()


def get_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def is_configured() -> bool:
    return bool(get_api_key())


def _chat_completion(messages: list[dict[str, str]], temperature: float = 0.3) -> str | None:
    api_key = get_api_key()
    if not api_key:
        return None

    payload = {
        "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }

    request = Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    choices = body.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def generate_question_suggestions() -> list[str]:
    system_prompt = (
        "You generate concise factual mutual fund example questions for a facts-only assistant. "
        "Only use the allowed scope: HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver, "
        "account statement help, capital gains statement, CAS, riskometer, benchmark, expense ratio, exit load, lock-in, and minimum SIP. "
        "Do not generate advice, performance, return, allocation, or PII-related questions. "
        "Return JSON with a single key named questions containing exactly 3 strings."
    )
    user_prompt = (
        "Generate 3 short example questions suitable for quick-action buttons in the chat UI. "
        "Keep each question under 60 characters."
    )
    content = _chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    if not content:
        return DEFAULT_SUGGESTIONS

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return DEFAULT_SUGGESTIONS

    questions = parsed.get("questions")
    if not isinstance(questions, list):
        return DEFAULT_SUGGESTIONS

    cleaned = [str(item).strip() for item in questions if str(item).strip()]
    return cleaned[:3] or DEFAULT_SUGGESTIONS


def improve_unavailable_answer(query: str, base_answer: str) -> tuple[str, list[str]]:
    system_prompt = (
        "You improve fallback answers for a mutual fund facts-only assistant. "
        "You must stay within these boundaries: no investment advice, no performance claims, no fund comparisons, and no PII handling. "
        "Do not invent facts. If the corpus does not support the answer, say so clearly and redirect the user to ask an in-scope factual question. "
        "Return JSON with keys answer and questions. "
        "The answer must be at most 2 sentences. "
        "The questions array must contain exactly 3 short in-scope factual questions."
    )
    user_prompt = (
        f"User query: {query}\n"
        f"Current fallback answer: {base_answer}\n"
        "Scope: HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver, and official statement/CAS help."
    )
    content = _chat_completion(
        [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    if not content:
        return base_answer, DEFAULT_SUGGESTIONS

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return base_answer, DEFAULT_SUGGESTIONS

    answer = str(parsed.get("answer") or "").strip() or base_answer
    questions = parsed.get("questions")
    if not isinstance(questions, list):
        return answer, DEFAULT_SUGGESTIONS

    cleaned = [str(item).strip() for item in questions if str(item).strip()]
    return answer, (cleaned[:3] or DEFAULT_SUGGESTIONS)
