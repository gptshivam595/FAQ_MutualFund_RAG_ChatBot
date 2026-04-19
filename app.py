from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_pipeline import MutualFundRAGAssistant, load_vectorstore


st.set_page_config(
    page_title="INDMoney \u2013 Mutual Fund FAQ Assistant",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_assistant() -> MutualFundRAGAssistant:
    return MutualFundRAGAssistant()


def queue_example(question: str) -> None:
    st.session_state.prefilled_question = question


def render_sidebar(assistant: MutualFundRAGAssistant) -> None:
    status = assistant.data_status()

    with st.sidebar:
        st.subheader("Document status")
        st.write(f"PDFs in `data/`: {status['pdf_count']}")
        st.write(f"Eligible official PDFs: {status['eligible_pdf_count']}")
        st.write(f"FAISS index ready: {'Yes' if status['index_ready'] else 'No'}")

        if st.button("Rebuild index", use_container_width=True):
            try:
                with st.spinner("Rebuilding the local FAISS index..."):
                    load_vectorstore.clear()
                    st.rerun()
            except FileNotFoundError as exc:
                st.warning(str(exc))
            else:
                st.success("Index rebuilt from local PDFs.")

        st.caption("Only local official PDFs are used. No internet data, no blogs, and no investment advice.")


def render_examples() -> None:
    st.caption("Example queries")
    example_columns = st.columns(3)
    examples = (
        "What is lock-in period of HDFC ELSS?",
        "What is benchmark of HDFC Flexi Cap Fund?",
        "What is expense ratio change in HDFC Large Cap Fund?",
    )
    for column, example in zip(example_columns, examples, strict=True):
        with column:
            if st.button(example, use_container_width=True):
                queue_example(example)


def render_response(answer: str, source_documents) -> None:
    response_card = st.container(border=True)
    with response_card:
        st.markdown("**Answer**")
        st.write(answer)

        st.markdown("**Source**")
        urls = [
            doc.metadata.get("url") or
            doc.metadata.get("source_url") or
            doc.metadata.get("link")
            for doc in source_documents
        ]
        urls = [u for u in urls if u and u.startswith("http")]
        urls = list(dict.fromkeys(urls))

        if urls:
            for url in urls:
                st.markdown(url)
        else:
            st.markdown("Source URL not available")

        dates = [
            doc.metadata.get("date") or
            doc.metadata.get("last_updated") or
            doc.metadata.get("document_date") or
            doc.metadata.get("created_at")
            for doc in source_documents
        ]
        dates = [d for d in dates if d]
        last_updated = dates[0] if dates else "Date not available"

        st.markdown(f"**Last Updated**\n\n{last_updated}")


def main() -> None:
    assistant = get_assistant()
    try:
        assistant.build_index()
    except FileNotFoundError:
        pass
    render_sidebar(assistant)

    st.title("INDMoney \u2013 Mutual Fund FAQ Assistant")
    st.caption("Facts only. No investment advice.")
    render_examples()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prefilled_question" not in st.session_state:
        st.session_state.prefilled_question = ""

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant":
                render_response(
                    answer=item["answer"],
                    source_documents=item.get("source_documents", []),
                )
            else:
                st.write(item["content"])

    question = st.chat_input(
        "Ask a factual question about HDFC Mutual Fund documents",
    )

    if st.session_state.prefilled_question and not question:
        question = st.session_state.prefilled_question
        st.session_state.prefilled_question = ""

    if not question:
        data_path = Path("data")
        if not any(data_path.glob("*.pdf")):
            st.info("Add the official HDFC Mutual Fund PDFs to `data/` to begin.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching official documents..."):
            response = assistant.answer_query(question)
        source_documents = response.sources
        render_response(
            answer=response.answer,
            source_documents=source_documents,
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "answer": response.answer,
            "source_documents": source_documents,
        }
    )


if __name__ == "__main__":
    main()
