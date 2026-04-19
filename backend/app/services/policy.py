from __future__ import annotations

import re


ADVICE_KEYWORDS = {
    "should i buy",
    "should i invest",
    "best fund",
    "which fund is better",
    "which one should i choose",
    "buy or sell",
    "portfolio",
    "allocate",
    "returns better",
    "compare returns",
    "high return",
}

PERFORMANCE_KEYWORDS = {
    "return",
    "returns",
    "performance",
    "cagr",
    "xirr",
    "profit",
    "gain",
    "alpha",
}

PII_PATTERNS = [
    re.compile(r"\b[a-z]{5}[0-9]{4}[a-z]\b", re.IGNORECASE),
    re.compile(r"\b\d{12}\b"),
    re.compile(r"\b\d{10}\b"),
    re.compile(r"\b\d{6,16}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
]


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def contains_pii(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in {"pan", "aadhaar", "otp", "folio", "phone", "email"}):
        return True

    return any(pattern.search(text) for pattern in PII_PATTERNS)


def is_advice_or_performance_question(text: str) -> bool:
    normalized = normalize(text)
    return any(keyword in normalized for keyword in ADVICE_KEYWORDS | PERFORMANCE_KEYWORDS)
