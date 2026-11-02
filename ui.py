import streamlit as st

from support_rag.config import Settings
from support_rag.observer import read_run
from support_rag.service import Service

st.set_page_config(page_title="NimbusDesk Support", page_icon="📚", layout="wide")


@st.cache_resource
def service():
    return Service(Settings())


st.title("NimbusDesk Support Lab")
st.caption("Answers from software manuals · source citations · model comparison · graph inspection")
backend = service()
aliases = list(backend.settings.models)
st.sidebar.subheader("Knowledge base")
st.sidebar.write(f"Embeddings: {backend.settings.embedding_provider}")
if backend.settings.embedding_provider == "demo":
    st.sidebar.warning("Demo: lexical hash embeddings and extractive answers. No live AI evaluation.")
if st.sidebar.button("Index sample manuals"):
    try:
        st.sidebar.success(backend.store.ingest())
    except Exception:
        st.sidebar.error("Ingestion failed. Check configuration and manual files.")


def show_answer(result):
    st.caption(f"{result['model']} · {result['status']} · {result['latency_ms']} ms")
    st.write(result["answer"])
    if result.get("error"):
        st.error(result["error"])
    st.code(result["run_id"], language=None)
    with st.expander("Source passages"):
        for source in result["sources"]:
            st.markdown(f"**{source['source']} / {source['section']}**")
            st.caption(f"{source['id']} · similarity {source['similarity']}")
            st.text(source["text"])
    with st.expander("Workflow events and usage"):
        st.json(result["events"])
        st.json(result["usage"])


ask_tab, compare_tab, inspect_tab = st.tabs(["Ask Support", "Compare Models", "Inspect Run"])
with ask_tab:
    with st.form("ask"):
        question = st.text_input("Support question", "How do I fix AUTH-104?")
        model = st.selectbox("Model", aliases)
        submitted = st.form_submit_button("Ask")
    if submitted:
        try:
            st.session_state["answer"] = backend.ask(question, model).model_dump()
        except Exception:
            st.error("Request failed. Check your input and configuration.")
    if "answer" in st.session_state:
        show_answer(st.session_state["answer"])
with compare_tab:
    with st.form("compare"):
        compare_question = st.text_input("Question for all models", "How do I fix AUTH-104?")
        selected = st.multiselect("Choose up to three models", aliases, default=aliases[:3])
        compare_submitted = st.form_submit_button("Compare")
    if compare_submitted:
        try:
            st.session_state["comparison"] = backend.compare(compare_question, selected)
        except Exception:
            st.error("Comparison failed. Choose one to three models and index the manuals first.")
    if "comparison" in st.session_state:
        comparison = st.session_state["comparison"]
        st.caption(
            f"One shared retrieval: {comparison['retrieval_ms']} ms. Per-model time excludes this step."
        )
        for column, result in zip(st.columns(len(comparison["results"])), comparison["results"]):
            with column:
                show_answer(result)
        st.download_button(
            "Download comparison JSON",
            __import__("json").dumps(comparison, indent=2),
            file_name="comparison.json",
            mime="application/json",
        )
with inspect_tab:
    st.markdown(
        "**Workflow:** retrieve → assess evidence → generate → validate citations; otherwise abstain."
    )
    run_id = st.text_input("Run identifier")
    if st.button("Load run"):
        try:
            st.json(read_run(backend.settings, run_id))
        except (ValueError, FileNotFoundError):
            st.warning("Run not found. Paste an identifier from a completed answer.")
    st.caption("Disk traces omit answers and passage text by default. Studio instructions are in the README.")
