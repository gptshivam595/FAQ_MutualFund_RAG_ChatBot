from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from app.models.schemas import ChatResponse
from app.services.policy import contains_pii, normalize


BASE_DIR = Path(__file__).resolve().parents[2]
CORPUS_FILE = BASE_DIR / "data" / "corpus.json"


class FAQBot:
    def __init__(self, corpus_path: Path | None = None) -> None:
        corpus_file = corpus_path or CORPUS_FILE
        self.data = json.loads(corpus_file.read_text(encoding="utf-8"))
        self.pack_last_refreshed = self.data["pack_last_refreshed"]
        self.scheme_items = self.data["schemes"]
        self.generic = self.data["generic"]
        self.documents = self.data["documents"]
        self.source_index = [
            {
                "label": document["title"],
                "url": document["url"],
                "type": document["type"],
                "scheme": document.get("scheme"),
            }
            for document in self.documents
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        return normalize(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _contains_pii(self, query: str) -> bool:
        patterns = [
            r"\b[a-z]{5}\d{4}[a-z]\b",
            r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
            r"\b(?:\+91[ -]?)?[6-9]\d{9}\b",
            r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
            r"\b(?:otp|one time password)\b",
        ]
        if contains_pii(query):
            return True
        if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns):
            return True
        return bool(re.findall(r"\b\d{9,18}\b", query))

    def _detect_scheme(self, query: str) -> dict[str, Any] | None:
        normalized_query = self._normalize(query)
        best_item = None
        best_score = 0

        for item in self.scheme_items:
            for alias in item["aliases"]:
                alias_norm = self._normalize(alias)
                if alias_norm in normalized_query and len(alias_norm) > best_score:
                    best_score = len(alias_norm)
                    best_item = item

        if best_item:
            return best_item

        if "elss" in normalized_query or "tax saver" in normalized_query:
            return self._get_scheme_by_name("HDFC ELSS Tax Saver")
        if "large cap" in normalized_query and "hdfc" in normalized_query:
            return self._get_scheme_by_name("HDFC Large Cap Fund")
        if "flexi cap" in normalized_query and "hdfc" in normalized_query:
            return self._get_scheme_by_name("HDFC Flexi Cap Fund")

        return None

    def _get_scheme_by_name(self, name: str) -> dict[str, Any] | None:
        for item in self.scheme_items:
            if item["name"] == name:
                return item
        return None

    def _detect_field(self, query: str) -> str | None:
        normalized_query = self._normalize(query)
        field_rules = [
            ("expense_ratio", ["expense ratio", "ter", "total expense ratio"]),
            ("exit_load", ["exit load", "load structure", "redemption load"]),
            ("min_sip", ["minimum sip", "min sip", "sip amount", "minimum investment", "minimum application amount"]),
            ("lock_in", ["lock in", "lock-in", "lockin"]),
            ("riskometer", ["riskometer", "risk level"]),
            ("benchmark", ["benchmark", "benchmark index", "index"]),
            ("capital_gains_statement", ["capital gains statement", "capital gain statement", "realized gains", "realised gains"]),
            ("download_cas", ["download cas", "e-cas", "ecas"]),
            ("cas_definition", ["what is cas", "what is a cas", "consolidated account statement"]),
            ("cas_timing", ["when is cas", "when will i receive cas", "when do i receive cas", "cas sent", "cas dispatch"]),
            ("statement_types", ["pan-based", "folio-based", "statement types"]),
            ("account_statement", ["account statement", "download statement", "statement"]),
        ]
        for field_name, keywords in field_rules:
            if any(keyword in normalized_query for keyword in keywords):
                return field_name
        return None

    def _is_advice_query(self, query: str) -> bool:
        normalized_query = self._normalize(query)
        advice_markers = [
            "should i",
            "good investment",
            "is it good",
            "buy or sell",
            "better fund",
            "best fund",
            "which fund",
            "recommend",
            "suitable",
            "safe",
            "worth it",
            "allocate",
            "portfolio",
            "switch from",
        ]
        return any(marker in normalized_query for marker in advice_markers)

    def _is_performance_query(self, query: str) -> bool:
        normalized_query = self._normalize(query)
        performance_markers = [
            "return",
            "returns",
            "performance",
            "nav growth",
            "rank",
            "outperform",
            "compare",
            "cagr",
            "xirr",
            "1 year",
            "3 year",
            "5 year",
            "10 year",
        ]
        if any(marker in normalized_query for marker in performance_markers):
            factual_fields = {
                "expense_ratio",
                "exit_load",
                "min_sip",
                "lock_in",
                "riskometer",
                "benchmark",
                "account_statement",
                "capital_gains_statement",
                "cas_definition",
                "cas_timing",
                "download_cas",
                "statement_types",
            }
            return self._detect_field(query) not in factual_fields
        return False

    def _needs_scheme_but_missing(self, field: str | None, scheme: dict[str, Any] | None) -> bool:
        return field in {"expense_ratio", "exit_load", "min_sip", "lock_in", "riskometer", "benchmark"} and scheme is None

    def _fallback_retrieve(self, query: str, limit: int = 1) -> list[dict[str, Any]]:
        query_tokens = set(self._tokenize(query))
        results: list[tuple[float, dict[str, Any]]] = []

        for document in self.documents:
            haystack = " ".join(
                [
                    document.get("title", ""),
                    document.get("content", ""),
                    " ".join(document.get("tags", [])),
                    str(document.get("scheme", "")),
                ]
            )
            document_tokens = set(self._tokenize(haystack))
            if not document_tokens:
                continue

            overlap = len(query_tokens & document_tokens)
            score = overlap / math.sqrt(max(len(query_tokens), 1) * max(len(document_tokens), 1))

            if document.get("scheme") and str(document["scheme"]).lower() in self._normalize(query):
                score += 0.75

            for tag in document.get("tags", []):
                if self._normalize(tag) in self._normalize(query):
                    score += 0.35

            if score > 0:
                results.append((score, document))

        results.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in results[:limit]]

    def _format_response(
        self,
        *,
        answer: str,
        source_title: str,
        source_url: str,
        last_updated: str,
        kind: str,
    ) -> ChatResponse:
        status = "answer"
        if kind in {"refusal", "safety"}:
            status = "refusal"
        if kind == "scope":
            status = "needs_scope"

        return ChatResponse(
            status=status,  # type: ignore[arg-type]
            answer=answer,
            source_label=source_title,
            source_url=source_url,
            last_updated_from_sources=last_updated,
        )

    def answer(self, query: str) -> ChatResponse:
        raw_query = query.strip()
        if not raw_query:
            scope = self.generic["scope"]
            return self._format_response(
                answer=scope["answer"],
                source_title=scope["source_title"],
                source_url=scope["source_url"],
                last_updated=scope["last_updated"],
                kind="scope",
            )

        if self._contains_pii(raw_query):
            services = self.generic["services"]
            return self._format_response(
                answer="Please do not enter PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers here; use HDFC Mutual Fund's official service flows instead.",
                source_title=services["source_title"],
                source_url=services["source_url"],
                last_updated=services["last_updated"],
                kind="safety",
            )

        if self._is_advice_query(raw_query):
            education = self.generic["education"]
            return self._format_response(
                answer="I can only answer facts from official public pages and cannot tell you whether to buy, sell, switch, or choose a scheme.",
                source_title=education["source_title"],
                source_url=education["source_url"],
                last_updated=education["last_updated"],
                kind="refusal",
            )

        scheme = self._detect_scheme(raw_query)
        field = self._detect_field(raw_query)

        if self._is_performance_query(raw_query):
            if scheme:
                return self._format_response(
                    answer=f"I do not compute or compare returns in this prototype; please use the official factsheet for {scheme['name']}.",
                    source_title=f"{scheme['name']} official factsheet",
                    source_url=scheme["factsheet_url"],
                    last_updated="2026-03",
                    kind="refusal",
                )
            factsheet_hub = self.generic["factsheet_hub"]
            return self._format_response(
                answer="I do not compute or compare returns in this prototype; please use the official HDFC Mutual Fund factsheets instead.",
                source_title=factsheet_hub["source_title"],
                source_url=factsheet_hub["source_url"],
                last_updated=factsheet_hub["last_updated"],
                kind="refusal",
            )

        if self._needs_scheme_but_missing(field, scheme):
            scope = self.generic["scope"]
            return self._format_response(
                answer="Please ask about one of the in-scope schemes: HDFC Large Cap Fund, HDFC Flexi Cap Fund, or HDFC ELSS Tax Saver.",
                source_title=scope["source_title"],
                source_url=scope["source_url"],
                last_updated=scope["last_updated"],
                kind="scope",
            )

        if field in {
            "capital_gains_statement",
            "download_cas",
            "cas_definition",
            "cas_timing",
            "statement_types",
            "account_statement",
        }:
            fact = self.generic[field]
            return self._format_response(
                answer=fact["answer"],
                source_title=fact["source_title"],
                source_url=fact["source_url"],
                last_updated=fact["last_updated"],
                kind="answer",
            )

        if scheme and field and field in scheme["fields"]:
            fact = scheme["fields"][field]
            return self._format_response(
                answer=fact["answer"],
                source_title=fact["source_title"],
                source_url=fact["source_url"],
                last_updated=fact["last_updated"],
                kind="answer",
            )

        if scheme and not field:
            return self._format_response(
                answer=f"I can answer scheme facts for {scheme['name']} such as expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, and official statement-download questions.",
                source_title=f"{scheme['name']} official page",
                source_url=scheme["source_url"],
                last_updated=scheme["last_updated"],
                kind="scope",
            )

        fallback_docs = self._fallback_retrieve(raw_query)
        if fallback_docs:
            document = fallback_docs[0]
            return self._format_response(
                answer=document["content"],
                source_title=document["title"],
                source_url=document["url"],
                last_updated=document["last_updated"],
                kind="answer",
            )

        scope = self.generic["scope"]
        return self._format_response(
            answer="This prototype only answers facts for the scoped HDFC Mutual Fund schemes and statement-service questions from official public pages.",
            source_title=scope["source_title"],
            source_url=scope["source_url"],
            last_updated=scope["last_updated"],
            kind="scope",
        )


knowledge_base = FAQBot()
answer_question = knowledge_base.answer
