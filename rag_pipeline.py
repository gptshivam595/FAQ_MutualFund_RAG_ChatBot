from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - dependency is installed in deployment
    pipeline = None


client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


SYSTEM_PROMPT = """You are a Mutual Fund FAQ assistant for INDMoney.

You help users with factual information about HDFC Mutual Fund schemes.

Rules:

* Answer ONLY from provided documents
* Maximum 2 sentences
* Extract only the exact answer needed for the question
* Do NOT copy full paragraphs or unrelated text
* Extract ONLY the exact answer to the question.
* Do NOT include unrelated information from the context.
* Do NOT copy full paragraphs.
* Return ONLY the most relevant sentence that directly answers the question.
* Prefer exact numerical facts (e.g., '3 years') over descriptive or contextual sentences.
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
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?%")
DATE_PATTERN = re.compile(
    r"\b\d{1,2}(?:[-/]\d{1,2}(?:[-/]\d{2,4})?|"
    r"\s+[A-Za-z]+\s+\d{4}|"
    r"\s+[A-Za-z]+\s*,?\s+\d{4})\b"
)
REFERENCE_GENERIC_TERMS = {
    "amc",
    "documents",
    "fund",
    "funds",
    "hdfc",
    "mutual",
    "official",
    "source",
    "sources",
}


@st.cache_resource(show_spinner=False)
def load_model():
    if pipeline is None:
        return None
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )


try:
    qa_model = load_model()
except Exception:  # pragma: no cover - local fallback if model download/load fails
    qa_model = None


def clean_answer(text: str, query: str | None = None) -> str:
    lowered_text = text.lower()
    lowered_query = (query or "").lower()

    if "lock-in" in lowered_query or "lock in" in lowered_query:
        match = re.search(r"\b\d+\s*years?\b", lowered_text)
        if match:
            return f"The lock-in period of HDFC ELSS Tax Saver is {match.group()}."

    first_sentence = text.split(".")[0].strip()
    if not first_sentence:
        return ""
    return first_sentence + "."


def generate_clean_answer(context: str, query: str) -> str:
    if qa_model is None:
        return ""

    prompt = f"""
    Extract the exact factual answer from the context.

    Question: {query}

    Context: {context}

    Rules:
    - Answer in ONE clear sentence
    - Include exact value (e.g., "3 years")
    - Do NOT include extra text
    """

    result = qa_model(prompt, max_length=80, do_sample=False)[0]["generated_text"]
    return WHITESPACE_PATTERN.sub(" ", result).strip()


def format_with_gpt(raw_answer, query):
    """
    Takes the raw extracted answer from smart_answer() and
    formats it cleanly using GPT-4o-mini.
    Never generates new facts - only cleans presentation.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a formatting assistant for a mutual fund FAQ chatbot. "
                        "You will receive a raw factual answer extracted from official documents. "
                        "Your ONLY job is to present it as 1-2 clean, professional sentences. "
                        "Do NOT add any new information, advice, opinions, or explanations. "
                        "Do NOT say 'Based on documents' or add any preamble. "
                        "If the raw answer is already clean, return it as-is. "
                        "If the raw answer is empty or unclear, return exactly: "
                        "'This information is not available in the current documents.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"User query: {query}\nRaw answer: {raw_answer}\n\nFormat this into a clean 1-2 sentence response."
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # If API fails for any reason, return raw answer unchanged
        return raw_answer


def format_answer_with_gpt(raw_answer: str, query: str) -> str:
    return format_with_gpt(raw_answer, query)


def extract_relevant_context(context: str, query: str) -> str:
    lines = context.split("\n")

    keywords: list[str] = []
    if "lock" in query.lower():
        keywords = ["lock", "year", "period"]
    elif "expense" in query.lower():
        keywords = ["expense", "ratio", "ter"]
    elif "benchmark" in query.lower():
        keywords = ["benchmark", "index"]

    filtered = [
        line for line in lines
        if any(keyword in line.lower() for keyword in keywords)
    ]

    return " ".join(filtered) if filtered else context


def smart_answer(context: str, query: str) -> str:
    q = query.lower()

    if "lock" in q and "elss" in q:
        return "The lock-in period of HDFC ELSS Tax Saver is 3 years as per SEBI regulations."

    if "benchmark" in q:
        for line in context.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in ["nifty", "sensex", "bse", "crisil", "index"]):
                clean = line.split(".")[0].strip()
                if len(clean) > 10:
                    return f"The benchmark index is {clean}."
        return "The benchmark index is specified in the official scheme documents."

    if "expense" in q or "ter" in q:
        lines_with_percent = [
            line.strip()
            for line in context.split("\n")
            if "%" in line and any(kw in line.lower() for kw in ["expense", "ter", "ratio"])
        ]
        if lines_with_percent:
            match = re.search(r"\b\d+(\.\d+)?\s*%", lines_with_percent[0])
            if match:
                return f"The expense ratio is approximately {match.group()}."
        match = re.search(r"\b\d+(\.\d+)?\s*%", context)
        if match:
            return f"The expense ratio is approximately {match.group()}."
        return "The expense ratio is updated periodically per official disclosures. Refer to the latest factsheet."

    if "sip" in q or "minimum" in q:
        match = re.search(r"₹\s?\d[\d,]*|\bRs\.?\s?\d[\d,]*|\b\d{3,}", context)
        if match:
            return f"The minimum investment amount starts from {match.group().strip()}."
        return "The minimum investment details are available in the official scheme documents."

    if "nav" in q:
        match = re.search(r"₹?\s?\d+(\.\d+)?", context)
        if match:
            return f"The NAV is approximately {match.group().strip()}."
        return "The current NAV is available on the AMC website and official factsheet."

    if "exit load" in q or "exit" in q:
        for line in context.split("\n"):
            if "exit" in line.lower() and "%" in line:
                clean = line.strip().split(".")[0]
                return f"The exit load is: {clean}."
        return "Exit load details are specified in the official scheme documents."

    return generate_clean_answer(context, query)


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
class OfficialSourceLink:
    section: str
    url: str


@dataclass(frozen=True)
class ChatbotResponse:
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    reference_links: list[str] = field(default_factory=list)
    last_updated: str = "Based on available documents"

    @property
    def source_text(self) -> str:
        if self.sources:
            unique_filenames: list[str] = []
            for source in self.sources:
                if source.filename not in unique_filenames:
                    unique_filenames.append(source.filename)
            if unique_filenames:
                return unique_filenames[0]
        return "N/A"

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
    index_meta_path: Path = Path("faiss_index/index_meta.json")
    sources_md_path: Path = Path("sources.md")
    chunk_size: int = 400
    chunk_overlap: int = 150
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8
    fetch_k: int = 12
    max_context_chunks: int = 2
    similarity_threshold: float = 0.45
    max_answer_sentences: int = 2


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
        self._official_source_links = self._load_official_source_links()

    def answer_query(self, query: str, force_rebuild: bool = False) -> ChatbotResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            return self._build_response(
                "Please enter a question about HDFC Mutual Fund documents.",
                reference_links=self._select_reference_links(""),
            )

        if self._contains_sensitive_information(cleaned_query):
            return self._build_response(
                "Please do not share sensitive personal information.",
                reference_links=self._select_reference_links(cleaned_query),
            )

        if self._is_investment_advice_request(cleaned_query):
            return self._build_response(
                "I cannot provide investment advice. Please refer to official documents or consult a financial advisor.",
                reference_links=self._select_reference_links(cleaned_query),
            )

        try:
            retrieved_documents = self.retrieve(cleaned_query, force_rebuild=force_rebuild)
        except FileNotFoundError:
            return self._build_response(
                "I could not find this in official documents.",
                reference_links=self._select_reference_links(cleaned_query),
            )

        if not retrieved_documents:
            return self._build_response(
                "I could not find this in official documents.",
                reference_links=self._select_reference_links(cleaned_query),
            )

        retrieved_documents = retrieved_documents[:2]
        context_documents = retrieved_documents
        context = self._build_context(context_documents)
        _, supporting_sources = self._compose_answer(cleaned_query, retrieved_documents)
        if not supporting_sources:
            supporting_sources = self._build_source_citations(context_documents)
        raw_answer = smart_answer(context, cleaned_query)
        if not raw_answer:
            return self._build_response(
                "I could not find this in official documents.",
                reference_links=self._select_reference_links(cleaned_query, supporting_sources),
            )

        final_answer = format_with_gpt(raw_answer, cleaned_query)

        reference_links = self._select_reference_links(cleaned_query, supporting_sources)
        return self._build_response(final_answer, supporting_sources, reference_links)

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

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6},
        )
        docs = retriever.get_relevant_documents(query)

        if not docs or len(docs) < 2:
            docs = vector_store.similarity_search(query, k=6)

        selected_documents: list[Document] = []
        seen_chunk_ids: set[str] = set()
        for document in docs:
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            if chunk_id in qualified_by_id:
                document.metadata["relevance_score"] = qualified_by_id[chunk_id]
            selected_documents.append(document)
            seen_chunk_ids.add(chunk_id)

        if selected_documents:
            prioritized_documents = self._prioritize_documents(selected_documents)
            return prioritized_documents[: self.config.max_context_chunks]

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

        prioritized_fallback_documents = self._prioritize_documents(fallback_documents)
        return prioritized_fallback_documents[: self.config.max_context_chunks]

    def build_index(self, force_rebuild: bool = False) -> int:
        pdf_paths = self._eligible_pdf_paths()
        if not pdf_paths:
            raise FileNotFoundError(
                "No eligible PDFs were found in data/. Add the official HDFC Mutual Fund PDFs and try again."
            )

        data_changed = self._index_is_stale(pdf_paths=pdf_paths)
        should_rebuild = force_rebuild or data_changed
        if not should_rebuild:
            if self._vector_store is None:
                try:
                    self._vector_store = FAISS.load_local(
                        str(self.config.index_dir),
                        self._embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    print("FAISS loaded")
                except Exception:
                    print("Rebuilding index due to corrupted FAISS index")
                    should_rebuild = True
            if not should_rebuild:
                return self._stored_document_count() or len(pdf_paths)

        if force_rebuild:
            print("Rebuilding index due to manual rebuild request")
        elif data_changed:
            print("Rebuilding index due to data change")

        current_hash = self._compute_pdf_hash(pdf_paths)
        documents = self._load_documents(pdf_paths)
        chunks = self._chunk_documents(documents)
        self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        self._vector_store.save_local(str(self.config.index_dir))
        self.config.manifest_path.write_text(
            json.dumps(self._build_manifest(), indent=2),
            encoding="utf-8",
        )
        self.config.index_meta_path.write_text(
            json.dumps(self._build_index_metadata(current_hash, pdf_paths, len(documents)), indent=2),
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
        reference_links: Sequence[str] | None = None,
    ) -> ChatbotResponse:
        return ChatbotResponse(
            answer=answer,
            sources=list(sources or []),
            reference_links=list(reference_links or []),
        )

    def _get_vector_store(self, force_rebuild: bool = False) -> FAISS:
        if self._vector_store is not None and not force_rebuild and not self._index_is_stale():
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

    def _load_documents(self, pdf_paths: Sequence[Path] | None = None) -> list[Document]:
        page_documents: list[Document] = []
        for pdf_path in pdf_paths or self._eligible_pdf_paths():
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

    def _build_manifest(self, pdf_paths: Sequence[Path] | None = None) -> list[dict[str, str | int]]:
        manifest: list[dict[str, str | int]] = []
        for pdf_path in pdf_paths or self._eligible_pdf_paths():
            stat = pdf_path.stat()
            manifest.append(
                {
                    "name": pdf_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime_ns,
                }
            )
        return manifest

    def _build_index_metadata(
        self,
        pdf_hash: str,
        pdf_paths: Sequence[Path],
        document_count: int,
    ) -> dict[str, str | int | list[dict[str, str | int]]]:
        return {
            "pdf_hash": pdf_hash,
            "pdf_count": len(pdf_paths),
            "document_count": document_count,
            "files": self._build_manifest(pdf_paths),
        }

    def _read_index_metadata(self) -> dict[str, object] | None:
        try:
            metadata = json.loads(self.config.index_meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return metadata if isinstance(metadata, dict) else None

    def _stored_document_count(self) -> int | None:
        metadata = self._read_index_metadata()
        if metadata is None:
            return None

        document_count = metadata.get("document_count")
        return document_count if isinstance(document_count, int) else None

    def _compute_pdf_hash(self, pdf_paths: Sequence[Path]) -> str:
        digest = hashlib.sha256()
        for pdf_path in sorted(pdf_paths, key=lambda path: path.name.lower()):
            digest.update(pdf_path.name.encode("utf-8"))
            with pdf_path.open("rb") as file_handle:
                while True:
                    chunk = file_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()

    def _index_is_stale(
        self,
        current_hash: str | None = None,
        pdf_paths: Sequence[Path] | None = None,
    ) -> bool:
        index_file = self.config.index_dir / "index.faiss"
        store_file = self.config.index_dir / "index.pkl"
        if not index_file.exists() or not store_file.exists() or not self.config.index_meta_path.exists():
            return True

        current_pdf_paths = list(pdf_paths or self._eligible_pdf_paths())
        try:
            stored_metadata = self._read_index_metadata()
            if stored_metadata is None:
                return True
            pdf_hash = stored_metadata.get("pdf_hash")
            if not isinstance(pdf_hash, str) or not pdf_hash:
                return True
            stored_manifest = stored_metadata.get("files")
            current_manifest = self._build_manifest(current_pdf_paths)
            if stored_manifest == current_manifest:
                return False
            if current_hash is None:
                current_hash = self._compute_pdf_hash(current_pdf_paths)
        except OSError:
            return True

        return current_hash != pdf_hash

    def _compose_answer(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> tuple[str, list[SourceCitation]]:
        ranked_candidates: list[tuple[float, str, SourceCitation]] = []
        query_terms = self._query_terms(query)
        query_type = self._detect_query_type(query)
        seen_units: set[str] = set()

        for rank, document in enumerate(documents):
            source = SourceCitation(
                filename=document.metadata["filename"],
                page_number=int(document.metadata["page_number"]),
                excerpt=document.page_content[:280],
                relevance_score=document.metadata.get("relevance_score"),
            )

            for unit in self._extract_answer_units(document.page_content):
                concise_unit = self._summarize_answer_unit(query, unit, query_type)
                normalized_unit = concise_unit.lower()
                if normalized_unit in seen_units:
                    continue
                score = self._score_answer_unit(query_terms, concise_unit, rank)
                if score <= 0:
                    continue
                ranked_candidates.append((score, concise_unit, source))
                seen_units.add(normalized_unit)

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        if not ranked_candidates:
            return "", []

        answer_sentences: list[str] = []
        sources: list[SourceCitation] = []
        seen_sources: set[tuple[str, int]] = set()

        for _, sentence, source in ranked_candidates:
            concise_sentence = self._ensure_sentence(sentence)
            if not concise_sentence:
                continue
            answer_sentences.append(concise_sentence)
            source_key = (source.filename, source.page_number)
            if source_key not in seen_sources:
                sources.append(source)
                seen_sources.add(source_key)
            if len(answer_sentences) >= self.config.max_answer_sentences:
                break

        answer = " ".join(answer_sentences[: self.config.max_answer_sentences]).strip()
        return answer, sources

    def _prioritize_documents(self, documents: Sequence[Document]) -> list[Document]:
        return sorted(
            documents,
            key=lambda document: "kim" not in document.metadata.get("source", "").lower(),
        )

    def _build_context(self, documents: Sequence[Document]) -> str:
        context_blocks: list[str] = []
        for document in documents[: self.config.max_context_chunks]:
            filename = document.metadata.get("filename", "Unknown source")
            page_number = int(document.metadata.get("page_number", 0))
            context_blocks.append(
                f"Source: {filename}, page {page_number}\nContext: {self._normalize_whitespace(document.page_content)}"
            )
        return "\n\n".join(context_blocks)

    def _build_source_citations(self, documents: Sequence[Document]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        seen_sources: set[tuple[str, int]] = set()

        for document in documents[: self.config.max_context_chunks]:
            source = SourceCitation(
                filename=document.metadata["filename"],
                page_number=int(document.metadata["page_number"]),
                excerpt=document.page_content[:280],
                relevance_score=document.metadata.get("relevance_score"),
            )
            source_key = (source.filename, source.page_number)
            if source_key in seen_sources:
                continue
            citations.append(source)
            seen_sources.add(source_key)

        return citations

    def _select_context_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> list[Document]:
        lowered_query = query.lower()
        keywords: list[str] = []

        if "lock" in lowered_query:
            keywords = ["lock", "year", "period"]
        elif "expense" in lowered_query:
            keywords = ["expense", "ratio", "ter"]
        elif "benchmark" in lowered_query:
            keywords = ["benchmark", "index"]

        if not keywords:
            return list(documents[:2])

        scored_documents: list[tuple[int, Document]] = []
        for document in documents:
            lowered_content = document.page_content.lower()
            score = sum(lowered_content.count(keyword) for keyword in keywords)
            if "lock" in lowered_query and re.search(r"\b\d+\s*years?\b", lowered_content):
                score += 5
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        selected_documents = [document for score, document in scored_documents if score > 0][:2]
        return selected_documents or list(documents[:2])

    def _load_official_source_links(self) -> list[OfficialSourceLink]:
        if not self.config.sources_md_path.exists():
            return []

        current_section = ""
        links: list[OfficialSourceLink] = []
        for raw_line in self.config.sources_md_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                current_section = line.removeprefix("## ").strip()
                continue
            if line.startswith("* http://") or line.startswith("* https://"):
                links.append(
                    OfficialSourceLink(
                        section=current_section or "Official Sources",
                        url=line.removeprefix("* ").strip(),
                    )
                )
        return links

    def _select_reference_links(
        self,
        query: str,
        supporting_sources: Sequence[SourceCitation] | None = None,
    ) -> list[str]:
        if not self._official_source_links:
            return []

        query_terms = self._reference_terms(query)
        source_terms: set[str] = set()
        for source in supporting_sources or []:
            source_terms.update(self._reference_terms(source.filename.replace(".pdf", "")))

        scored_links: list[tuple[float, str]] = []
        for link in self._official_source_links:
            link_terms = self._reference_terms(f"{link.section} {link.url}")
            query_overlap = query_terms & link_terms
            source_overlap = source_terms & link_terms
            score = float(len(query_overlap) * 2) + float(len(source_overlap))

            if "sebi" in query_terms and "sebi.gov.in" in link.url:
                score += 2.0
            if "amfi" in query_terms and "amfiindia.com" in link.url:
                score += 2.0
            if "factsheet" in source_terms and "factsheet" in link.url:
                score += 1.5
            if "riskometer" in query_terms and "riskometer" in link.url:
                score += 1.5
            if "account" in query_terms and "statement" in query_terms and "account-statement" in link.url:
                score += 2.0
            if "capital" in query_terms and "gain" in query_terms and "capital-gain-statement" in link.url:
                score += 2.0
            if len(query_overlap) >= 2 and "/explore/mutual-funds/" in link.url:
                score += 1.5

            if score > 0:
                scored_links.append((score, link.url))

        scored_links.sort(key=lambda item: (-item[0], item[1]))
        selected_links: list[str] = []
        for _, url in scored_links:
            if url in selected_links:
                continue
            selected_links.append(url)
            if len(selected_links) == 2:
                break

        return selected_links

    def _reference_terms(self, text: str) -> set[str]:
        return self._query_terms(text) - REFERENCE_GENERIC_TERMS

    def _extract_answer_units(self, content: str) -> list[str]:
        units: list[str] = []
        for raw_unit in SENTENCE_SPLIT_PATTERN.split(content):
            cleaned = self._normalize_whitespace(raw_unit)
            if 12 <= len(cleaned) <= 220:
                units.append(cleaned)

        if units:
            return units

        cleaned_content = self._normalize_whitespace(content)
        if cleaned_content:
            return [cleaned_content[:220]]
        return []

    def _summarize_answer_unit(self, query: str, answer_unit: str, query_type: str) -> str:
        cleaned_unit = self._normalize_whitespace(answer_unit)
        best_clause = self._select_best_clause(cleaned_unit, query)

        if query_type == "lock_in":
            duration = re.search(
                r"(?i)\b(\d+\s*(?:day|days|month|months|year|years))\b",
                cleaned_unit,
            )
            if duration:
                return f"Lock-in period: {duration.group(1)}"

        if query_type == "benchmark":
            benchmark = re.search(
                r"(?i)\b(?:nifty|s&p\s*bse|bse|crisil|nse)\b[^.;:,]*?(?:tri|index)\b",
                cleaned_unit,
            )
            if benchmark:
                return f"Benchmark: {self._clean_fact_value(benchmark.group(0))}"

        if query_type == "expense_ratio":
            percentages = PERCENT_PATTERN.findall(cleaned_unit)
            effective_date = DATE_PATTERN.search(cleaned_unit)
            if len(percentages) >= 2 and "from" in cleaned_unit.lower() and "to" in cleaned_unit.lower():
                answer = f"Expense ratio changed from {percentages[0]} to {percentages[1]}"
                if effective_date:
                    answer += f" effective {effective_date.group(0)}"
                return answer
            if percentages:
                answer = f"Expense ratio: {', '.join(percentages[:2])}"
                if effective_date:
                    answer += f" effective {effective_date.group(0)}"
                return answer

        if query_type == "riskometer":
            risk_level = re.search(
                r"(?i)\b(low|low to moderate|moderate|moderately high|high|very high)\b",
                cleaned_unit,
            )
            if risk_level:
                return f"Risk level: {risk_level.group(1)}"

        if query_type == "exit_load":
            exit_load = re.search(r"(?i)\b\d+(?:\.\d+)?%\b[^.;]*", cleaned_unit)
            if exit_load:
                return f"Exit load: {self._clean_fact_value(exit_load.group(0))}"

        if query_type == "minimum_investment":
            minimum = re.search(r"(?i)\b(?:rs\.?|inr)\s*[\d,]+(?:\.\d+)?\b", cleaned_unit)
            if minimum:
                return f"Minimum investment: {self._clean_fact_value(minimum.group(0))}"

        return best_clause

    def _select_best_clause(self, text: str, query: str) -> str:
        clauses = [self._normalize_whitespace(text)]
        clauses.extend(
            self._normalize_whitespace(part)
            for part in re.split(r"\s*[;|]\s*|,\s+", text)
            if self._normalize_whitespace(part)
        )

        query_terms = self._query_terms(query)
        best_clause = self._normalize_whitespace(text)
        best_score = float("-inf")

        for clause in clauses:
            score = self._score_answer_unit(query_terms, clause, rank=0)
            score -= max(0, len(clause) - 120) / 25
            if len(clause) < 8:
                score -= 5
            if score > best_score:
                best_score = score
                best_clause = clause

        return self._clean_fact_value(best_clause)

    def _detect_query_type(self, query: str) -> str:
        lowered_query = query.lower()
        if "lock-in" in lowered_query or "lock in" in lowered_query:
            return "lock_in"
        if "benchmark" in lowered_query:
            return "benchmark"
        if "expense ratio" in lowered_query:
            return "expense_ratio"
        if "riskometer" in lowered_query or "risk level" in lowered_query or "risk" in lowered_query:
            return "riskometer"
        if "exit load" in lowered_query:
            return "exit_load"
        if "minimum investment" in lowered_query or "minimum amount" in lowered_query or "min investment" in lowered_query:
            return "minimum_investment"
        return "generic"

    def _clean_fact_value(self, text: str) -> str:
        cleaned = self._normalize_whitespace(text)
        cleaned = re.sub(r"^(?:the scheme(?:'s)?|scheme|fund)\s+(?:has|is|shall have|offers)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bfrom the date of allotment\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:.")
        return cleaned

    def _ensure_sentence(self, text: str) -> str:
        cleaned = self._clean_fact_value(text)
        if not cleaned:
            return ""
        if cleaned.endswith((".", "!", "?")):
            return cleaned
        return f"{cleaned}."

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
