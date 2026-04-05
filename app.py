"""
app.py
------
Streamlit front-end for the UK Data Science Job RAG application.

Run with:
    streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st

from rag.ingest import load_job_descriptions, chunk_documents
from rag.vectorstore import get_embeddings, get_or_build_vectorstore
from rag.chain import retrieve_relevant_jobs, stream_gap_analysis, stream_qa

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UK DS Job Gap Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — API key and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 UK DS Gap Analyser")
    st.markdown(
        "Paste your CV and discover which data science skills you need to land "
        "your target role in the UK job market."
    )
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Your API key is never stored. It is used only for this session.",
    )

    st.divider()
    st.subheader("Retrieval settings")

    num_results = st.slider(
        "Job descriptions to retrieve",
        min_value=2,
        max_value=8,
        value=5,
        help=(
            "Number of JDs retrieved via Max Marginal Relevance. "
            "Higher values give Claude more context but use more tokens."
        ),
    )

    lambda_mult = st.slider(
        "MMR diversity (λ)",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help=(
            "λ=1.0 → pure similarity (all results close to query). "
            "λ=0.0 → pure diversity (results maximally spread). "
            "0.5–0.7 is a good balanced range."
        ),
    )

    st.divider()
    st.subheader("Index")

    rebuild = st.button(
        "Rebuild vector index",
        help="Re-embed all job descriptions. Use after adding new JDs to data/jobs/.",
    )

    st.divider()
    st.caption(
        "Powered by **Claude Opus 4.6** · "
        "Embeddings by **all-MiniLM-L6-v2** · "
        "Vector search by **FAISS**"
    )

# ---------------------------------------------------------------------------
# Initialise the RAG pipeline (cached in session state)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def load_pipeline(force_rebuild: bool = False):
    """Load embeddings + build/load FAISS index.  Cached across reruns."""
    # Ensure we run from the ds-job-rag/ directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "jobs"
    index_dir = script_dir / "faiss_index"

    docs = load_job_descriptions(str(data_dir))
    chunks = chunk_documents(docs)

    embeddings = get_embeddings(device="cpu")
    vectorstore = get_or_build_vectorstore(
        chunks,
        embeddings=embeddings,
        persist_dir=str(index_dir),
        force_rebuild=force_rebuild,
    )
    return vectorstore, docs


if rebuild:
    # Clear the cache and force a rebuild
    st.cache_resource.clear()

try:
    vectorstore, all_docs = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_gap, tab_qa, tab_browse = st.tabs(
    ["📋 Gap Analysis", "💬 Ask About Roles", "🗂 Browse Job Descriptions"]
)

# ---- Tab 1: Gap Analysis ---------------------------------------------------
with tab_gap:
    st.header("CV Gap Analysis")
    st.markdown(
        "Paste your CV below. The app will find the most relevant UK data science "
        "roles, then ask Claude to identify which skills you already have and which "
        "you need to develop."
    )

    cv_text = st.text_area(
        "Your CV (plain text)",
        height=300,
        placeholder=(
            "Paste your CV here…\n\n"
            "Include: education, work experience, skills, tools, frameworks, "
            "projects, certifications, publications."
        ),
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyse_btn = st.button("Analyse gaps", type="primary", use_container_width=True)

    if analyse_btn:
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif len(cv_text.strip()) < 50:
            st.warning("Please paste more of your CV — at least a few sentences.")
        else:
            with st.spinner("Searching job descriptions…"):
                retrieved = retrieve_relevant_jobs(
                    cv_text,
                    vectorstore,
                    k=num_results,
                    fetch_k=max(num_results * 3, 20),
                    lambda_mult=lambda_mult,
                )

            st.subheader("Matched job descriptions")
            with st.expander(
                f"Retrieved {len(retrieved)} JD chunks (click to inspect)", expanded=False
            ):
                for doc in retrieved:
                    st.markdown(
                        f"**{doc.metadata.get('role', 'Unknown')}** "
                        f"· `{doc.metadata.get('source', '')}`"
                    )
                    st.text(doc.page_content[:300] + "…")
                    st.divider()

            st.subheader("Gap analysis")
            result_container = st.empty()
            full_response = ""

            try:
                start = time.time()
                for chunk in stream_gap_analysis(
                    cv_text, retrieved, api_key=api_key
                ):
                    full_response += chunk
                    result_container.markdown(full_response + "▌")
                result_container.markdown(full_response)
                elapsed = time.time() - start
                st.caption(f"Generated in {elapsed:.1f}s")
            except Exception as e:
                st.error(f"Error calling Claude API: {e}")

# ---- Tab 2: Q&A ------------------------------------------------------------
with tab_qa:
    st.header("Ask About UK Data Science Roles")
    st.markdown(
        "Ask any question about the skills, salaries, tools, or hiring practices "
        "in the UK data science job market. The app retrieves relevant JDs and "
        "grounds Claude's answer in them."
    )

    sample_questions = [
        "What Python libraries are most commonly required across all roles?",
        "Which roles require cloud platform experience, and which cloud providers?",
        "What is the typical salary range for senior data scientists in London?",
        "What soft skills appear most frequently in the job descriptions?",
        "Which role is the best entry point for someone transitioning from academia?",
    ]

    st.markdown("**Example questions:**")
    cols = st.columns(len(sample_questions))
    selected_q = ""
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        if col.button(q[:40] + "…", key=f"sample_{i}", help=q):
            selected_q = q

    question = st.text_input(
        "Your question",
        value=selected_q,
        placeholder="e.g. What MLOps skills are most in demand?",
    )

    ask_btn = st.button("Ask", type="primary")

    if ask_btn:
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant job descriptions…"):
                retrieved = retrieve_relevant_jobs(
                    question,
                    vectorstore,
                    k=num_results,
                    fetch_k=max(num_results * 3, 20),
                    lambda_mult=lambda_mult,
                )

            result_container = st.empty()
            full_response = ""

            try:
                for chunk in stream_qa(question, retrieved, api_key=api_key):
                    full_response += chunk
                    result_container.markdown(full_response + "▌")
                result_container.markdown(full_response)
            except Exception as e:
                st.error(f"Error calling Claude API: {e}")

# ---- Tab 3: Browse JDs -----------------------------------------------------
with tab_browse:
    st.header("Browse Job Descriptions")
    st.markdown(
        f"The index contains **{len(all_docs)} job descriptions**. "
        "Click any card to read the full text."
    )

    for doc in all_docs:
        role = doc.metadata.get("role", "Unknown Role")
        source = doc.metadata.get("source", "")

        # Extract first line as a rough "company + title" header
        first_line = doc.page_content.split("\n")[0].strip()

        with st.expander(f"📄 {role}  ·  `{source}`"):
            st.markdown(doc.page_content)
