# INDMoney Mutual Fund FAQ Assistant

A production-oriented, facts-only RAG assistant for HDFC Mutual Fund FAQs built with Python, LangChain, FAISS, HuggingFace embeddings, and Streamlit.

## What it does

- Loads official PDFs from `data/` only.
- Extracts and chunks document text with `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- Builds a local FAISS index with `all-MiniLM-L6-v2` embeddings.
- Retrieves answers with MMR-style diversification plus a similarity threshold fallback.
- Refuses investment advice and blocks sensitive personal information.
- Returns short answers with source citations in the required format.

## Supported document scope

The ingestion layer is intentionally restrictive and only indexes official HDFC Mutual Fund documents related to:

- HDFC Flexi Cap Fund
- HDFC ELSS Tax Saver
- HDFC Top 100 Fund
- HDFC Large Cap Fund
- HDFC MF factsheets
- Investor Charter
- Riskometer disclosures
- Expense ratio notices

Out-of-scope PDFs such as presentations or unrelated fund documents are skipped automatically.

## Project files

- `app.py`: Streamlit UI
- `rag_pipeline.py`: document ingestion, FAISS indexing, retrieval, refusals, and answer formatting
- `requirements.txt`: Python dependencies

## Setup

1. Use Python 3.11 or 3.12 for best compatibility with `faiss-cpu` and `sentence-transformers`.
2. Create and activate a virtual environment.
3. Place the official PDFs inside `data/`.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
streamlit run app.py
```

## Example questions

- `What is lock-in period of HDFC ELSS?`
- `What is benchmark of HDFC Flexi Cap Fund?`
- `What is expense ratio change in HDFC Large Cap Fund?`

## Safety behavior

- If the user shares a PAN, Aadhaar number, phone number, or email address, the app responds with: `Please do not share sensitive personal information.`
- If the user asks for advice such as `Should I invest?` or `Which fund is best?`, the app responds with: `I cannot provide investment advice. Please refer to official documents or consult a financial advisor.`
- If the answer is not grounded in the indexed PDFs, the app responds with: `I could not find this in official documents.`

## Retrieval notes

- Chunk size: `500`
- Chunk overlap: `100`
- Retriever behavior: similarity threshold filtering followed by MMR selection
- Returned context size: `top_k = 4`

## Operational notes

- The FAISS index is stored locally in `faiss_index/`.
- Rebuild the index from the Streamlit sidebar whenever PDFs are added or updated.
- The app never uses blogs, APIs, or internet data for answering user questions.
