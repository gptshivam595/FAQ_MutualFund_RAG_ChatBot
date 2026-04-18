from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


SYSTEM_PROMPT = """You are a Mutual Fund FAQ assistant for INDMoney.

You help users with factual information about HDFC Mutual Fund schemes.

Rules:

* Answer ONLY from provided documents
* Maximum 3 sentences
* ALWAYS include a source citation (PDF name + page number)
* If answer not found, say:
  'I could not find this in official documents.'
* If user asks opinion (e.g., 'Should I invest?', 'Best fund?'):
  respond:
  'I cannot provide investment advice. Please refer to official sources like SEBI: https://www.sebi.gov.in'
* Do NOT calculate returns
* Do NOT compare funds unless explicitly present in documents
* Do NOT assume missing data

Answer format:
Answer: <short factual answer>
Source: <PDF name, page number>
Last updated from sources: Based on available documents
"""


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
}

INVESTMENT_ADVICE_PATTERNS = (
    r"\bshould i invest\b",
    r"\bwhich fund is best\b",
    r"\bbest fund\b",
    r"\bcompare returns\b",
    r"\bwhich should i choose\b",
    r"\bis it good\b",
    r"\bworth investing\b",
    r"\brecommend\b",
    r"\badvice\b",
)

PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+|(?:\u2022)\s*")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SourceCitation:
    filename: str
    page_number: int
    excerpt: str
    relevance_score: float | None = None

    @property
    def display_name(self) -> str:
        return f"{self.filename}, page {self.page_number}"


@dataclass(frozen=True)
class ChatbotResponse:
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    last_updated: str = "Based on available documents"

    @property
    def source_text(self) -> str:
        if not self.sources:
            return "N/A"
        return "; ".join(source.display_name for source in self.sources)

    @property
    def formatted(self) -> str:
        return (
            f"Answer:\n{self.answer}\n\n"
            f"Source:\n{self.source_text}\n\n"
            f"Last updated from sources:\n{self.last_updated}"
        )


@dataclass(frozen=True)
class RAGConfig:
    data_dir: Path = Path("data")
    index_dir: Path = Path("faiss_index")
    manifest_path: Path = Path("faiss_index/manifest.json")
    chunk_size: int = 500
    chunk_overlap: int = 100
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    fetch_k: int = 12
    similarity_threshold: float = 0.45
    max_answer_sentences: int = 3


class MutualFundRAGAssistant:
    """Facts-only FAQ assistant grounded in local HDFC Mutual Fund PDFs."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embeddings_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store: FAISS | None = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def answer_query(self, query: str, force_rebuild: bool = False) -> ChatbotResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            return self._build_response("Please enter a question about HDFC Mutual Fund documents.")

        if self._contains_sensitive_information(cleaned_query):
            return self._build_response("Please do not share sensitive personal information.")

        if self._is_investment_advice_request(cleaned_query):
            return self._build_response(
                "I cannot provide investment advice. Please refer to official documents or consult a financial advisor."
            )

        try:
            retrieved_documents = self.retrieve(cleaned_query, force_rebuild=force_rebuild)
        except FileNotFoundError:
            return self._build_response("I could not find this in official documents.")

        if not retrieved_documents:
            return self._build_response("I could not find this in official documents.")

        answer_text, supporting_sources = self._compose_answer(cleaned_query, retrieved_documents)
        if not answer_text:
            return self._build_response("I could not find this in official documents.")

        return self._build_response(answer_text, supporting_sources)

    def retrieve(self, query: str, force_rebuild: bool = False) -> list[Document]:
        vector_store = self._get_vector_store(force_rebuild=force_rebuild)

        threshold_hits = vector_store.similarity_search_with_relevance_scores(
            query,
            k=self.config.fetch_k,
        )

        qualified_by_id: dict[str, float] = {}
        for document, score in threshold_hits:
            if score >= self.config.similarity_threshold:
                qualified_by_id[document.metadata["chunk_id"]] = score

        if not qualified_by_id:
            return []

        mmr_candidates = vector_store.max_marginal_relevance_search(
            query,
            k=self.config.top_k,
            fetch_k=self.config.fetch_k,
        )

        selected_documents: list[Document] = []
        seen_chunk_ids: set[str] = set()
        for document in mmr_candidates:
            chunk_id = document.metadata["chunk_id"]
            if chunk_id not in qualified_by_id or chunk_id in seen_chunk_ids:
                continue
            document.metadata["relevance_score"] = qualified_by_id[chunk_id]
            selected_documents.append(document)
            seen_chunk_ids.add(chunk_id)

        if selected_documents:
            return selected_documents

        fallback_documents: list[Document] = []
        for document, score in threshold_hits:
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in seen_chunk_ids or chunk_id not in qualified_by_id:
                continue
            document.metadata["relevance_score"] = score
            fallback_documents.append(document)
            seen_chunk_ids.add(chunk_id)
            if len(fallback_documents) == self.config.top_k:
                break

        return fallback_documents

    def build_index(self, force_rebuild: bool = False) -> int:
        documents = self._load_documents()
        if not documents:
            raise FileNotFoundError(
                "No eligible PDFs were found in data/. Add the official HDFC Mutual Fund PDFs and try again."
            )

        should_rebuild = force_rebuild or self._index_is_stale()
        if not should_rebuild and self._vector_store is None:
            self._vector_store = FAISS.load_local(
                str(self.config.index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return len(documents)

        chunks = self._chunk_documents(documents)
        self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        self._vector_store.save_local(str(self.config.index_dir))
        self.config.manifest_path.write_text(
            json.dumps(self._build_manifest(), indent=2),
            encoding="utf-8",
        )
        return len(documents)

    def data_status(self) -> dict[str, int]:
        pdf_count = len(self._discover_pdf_paths())
        eligible_count = len(self._eligible_pdf_paths())
        indexed = int((self.config.index_dir / "index.faiss").exists())
        return {
            "pdf_count": pdf_count,
            "eligible_pdf_count": eligible_count,
            "index_ready": indexed,
        }

    def _build_response(
        self,
        answer: str,
        sources: Sequence[SourceCitation] | None = None,
    ) -> ChatbotResponse:
        return ChatbotResponse(answer=answer, sources=list(sources or []))

    def _get_vector_store(self, force_rebuild: bool = False) -> FAISS:
        if self._vector_store is not None and not force_rebuild:
            return self._vector_store

        self.build_index(force_rebuild=force_rebuild)
        if self._vector_store is None:
            raise RuntimeError("Vector store could not be initialized.")
        return self._vector_store

    def _discover_pdf_paths(self) -> list[Path]:
        return sorted(self.config.data_dir.glob("*.pdf"))

    def _eligible_pdf_paths(self) -> list[Path]:
        eligible_paths: list[Path] = []
        for pdf_path in self._discover_pdf_paths():
            normalized_name = pdf_path.name.lower()
            if any(disallowed in normalized_name for disallowed in ("presentation", "other funds", "rsf")):
                continue
            if any(
                allowed in normalized_name
                for allowed in (
                    "hdfc flexi cap",
                    "hdfc elss",
                    "hdfc top 100",
                    "hdfc large cap",
                    "hdfc mf factsheet",
                    "riskometer",
                    "investor charter",
                )
            ):
                eligible_paths.append(pdf_path)
        return eligible_paths

    def _load_documents(self) -> list[Document]:
        page_documents: list[Document] = []
        for pdf_path in self._eligible_pdf_paths():
            loader = PyPDFLoader(str(pdf_path))
            for page in loader.load():
                content = self._normalize_whitespace(page.page_content)
                if not content:
                    continue
                page_number = int(page.metadata.get("page", 0)) + 1
                page_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "filename": pdf_path.name,
                            "page_number": page_number,
                            "source": pdf_path.name,
                        },
                    )
                )
        return page_documents

    def _chunk_documents(self, documents: Sequence[Document]) -> list[Document]:
        split_documents = self._text_splitter.split_documents(list(documents))
        chunked_documents: list[Document] = []
        for index, document in enumerate(split_documents):
            metadata = dict(document.metadata)
            metadata["chunk_id"] = (
                f"{metadata['filename']}::page-{metadata['page_number']}::chunk-{index}"
            )
            chunked_documents.append(Document(page_content=document.page_content, metadata=metadata))
        return chunked_documents

    def _build_manifest(self) -> list[dict[str, str | int]]:
        manifest: list[dict[str, str | int]] = []
        for pdf_path in self._eligible_pdf_paths():
            stat = pdf_path.stat()
            manifest.append(
                {
                    "name": pdf_path.name,
                    "size": stat.st_size,
                    "modified": int(stat.st_mtime),
                }
            )
        return manifest

    def _index_is_stale(self) -> bool:
        index_file = self.config.index_dir / "index.faiss"
        store_file = self.config.index_dir / "index.pkl"
        if not index_file.exists() or not store_file.exists() or not self.config.manifest_path.exists():
            return True

        try:
            current_manifest = self._build_manifest()
            stored_manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True

        return current_manifest != stored_manifest

    def _compose_answer(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> tuple[str, list[SourceCitation]]:
        ranked_candidates: list[tuple[float, str, SourceCitation]] = []
        query_terms = self._query_terms(query)
        seen_units: set[str] = set()

        for rank, document in enumerate(documents):
            source = SourceCitation(
                filename=document.metadata["filename"],
                page_number=int(document.metadata["page_number"]),
                excerpt=document.page_content[:280],
                relevance_score=document.metadata.get("relevance_score"),
            )

            for unit in self._extract_answer_units(document.page_content):
                normalized_unit = unit.lower()
                if normalized_unit in seen_units:
                    continue
                score = self._score_answer_unit(query_terms, unit, rank)
                if score <= 0:
                    continue
                ranked_candidates.append((score, unit, source))
                seen_units.add(normalized_unit)

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        if not ranked_candidates:
            return "", []

        answer_sentences: list[str] = []
        sources: list[SourceCitation] = []
        seen_sources: set[tuple[str, int]] = set()

        for _, sentence, source in ranked_candidates:
            answer_sentences.append(sentence.rstrip(".") + ".")
            source_key = (source.filename, source.page_number)
            if source_key not in seen_sources:
                sources.append(source)
                seen_sources.add(source_key)
            if len(answer_sentences) >= self.config.max_answer_sentences:
                break

        answer = " ".join(answer_sentences[: self.config.max_answer_sentences]).strip()
        return answer, sources

    def _extract_answer_units(self, content: str) -> list[str]:
        units: list[str] = []
        for raw_unit in SENTENCE_SPLIT_PATTERN.split(content):
            cleaned = self._normalize_whitespace(raw_unit)
            if 25 <= len(cleaned) <= 280:
                units.append(cleaned)

        if units:
            return units

        cleaned_content = self._normalize_whitespace(content)
        if cleaned_content:
            return [cleaned_content[:280]]
        return []

    def _score_answer_unit(self, query_terms: set[str], answer_unit: str, rank: int) -> float:
        if not query_terms:
            return 0.0

        tokens = set(TOKEN_PATTERN.findall(answer_unit.lower()))
        overlap = query_terms & tokens
        if not overlap:
            return 0.0

        numeric_overlap = {token for token in overlap if token.isdigit()}
        score = float(len(overlap)) + (2.0 * len(numeric_overlap))
        if len(overlap) == len(query_terms):
            score += 1.0
        score += max(0.0, 1.5 - (rank * 0.25))
        if any(symbol in answer_unit for symbol in ("%", "year", "years", "month", "months", "tri", "benchmark")):
            score += 0.5
        return score

    def _query_terms(self, query: str) -> set[str]:
        tokens = TOKEN_PATTERN.findall(query.lower())
        return {token for token in tokens if token not in STOPWORDS}

    def _contains_sensitive_information(self, query: str) -> bool:
        return any(
            pattern.search(query)
            for pattern in (PAN_PATTERN, AADHAAR_PATTERN, PHONE_PATTERN, EMAIL_PATTERN)
        )

    def _is_investment_advice_request(self, query: str) -> bool:
        lowered_query = query.lower()
        return any(re.search(pattern, lowered_query) for pattern in INVESTMENT_ADVICE_PATTERNS)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()


def mmr_rerank_query(
    query_embedding: Iterable[float],
    document_embeddings: Sequence[Sequence[float]],
    top_k: int,
) -> list[int]:
    """Standalone MMR helper kept available for future retrieval tuning."""

    return maximal_marginal_relevance(
        np.array(query_embedding),
        [np.array(embedding) for embedding in document_embeddings],
        k=top_k,
    )
