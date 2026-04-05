"""
chain.py
--------
Retrieval and generation logic: given a CV string, retrieve the most
relevant job descriptions from FAISS and stream a structured gap analysis
from Claude Opus 4.6.

Retrieval strategy
~~~~~~~~~~~~~~~~~~
We use Max Marginal Relevance (MMR) retrieval rather than plain top-k
similarity search.  MMR jointly maximises relevance to the query and
diversity among the returned documents:

    score(d) = λ · sim(q, d) − (1 − λ) · max_{r ∈ R} sim(r, d)

where R is the set of already-selected results.  λ = 0.6 here, which
weights relevance slightly higher than diversity.  The practical effect is
that if two job descriptions are near-identical (e.g. both "Senior Data
Scientist at a bank"), MMR will pick one and substitute a more different
role in the k-th slot — giving Claude broader context about the skills
landscape.

Gap analysis prompt design
~~~~~~~~~~~~~~~~~~~~~~~~~~
The prompt asks Claude for a *structured* response with four sections:
  1. Matched skills  — what the CV already covers
  2. Missing skills  — what the JDs require that is absent from the CV
  3. Priority gaps   — the top 3 gaps ranked by market frequency
  4. Learning path   — concrete next steps with estimated effort

Claude is instructed to cite which job descriptions it is drawing from,
so the user can verify the grounding.  Adaptive thinking is enabled so
Claude can reason through multi-step skill inference (e.g., "PySpark" in
a JD versus "distributed computing" mentioned loosely in the CV).
"""

import os
from typing import Generator, List

import anthropic
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior technical recruiter and career coach specialising
in data science and machine learning roles in the UK job market.  You have
deep knowledge of the technical skills, tools, and domain expertise that UK
employers require at different seniority levels.

When analysing a candidate's CV against job descriptions, you:
- Identify both explicit skill matches (e.g., "Python" in both) and implicit
  ones (e.g., "statistical modelling" covering "logistic regression")
- Distinguish between hard skills (tools, languages, frameworks) and soft
  skills (communication, stakeholder management)
- Are realistic about the severity of gaps — not every gap is a blocker
- Provide actionable, specific recommendations with concrete resources
- Cite which job descriptions you are drawing from when identifying gaps"""


def _build_gap_analysis_prompt(cv_text: str, job_docs: List[Document]) -> str:
    """Construct the user-turn prompt for the gap analysis."""
    jd_block = "\n\n---\n\n".join(
        f"**JD {i+1}: {doc.metadata.get('role', 'Unknown Role')}**\n{doc.page_content}"
        for i, doc in enumerate(job_docs)
    )

    return f"""I am going to give you my CV and {len(job_docs)} UK data science job descriptions
that are relevant to my target roles.  Please produce a thorough gap analysis.

## My CV
{cv_text}

## Relevant UK Job Descriptions
{jd_block}

---

Please structure your response as follows:

### 1. Skills Already Present in Your CV
List the technical and soft skills from the job descriptions that your CV already demonstrates.
Group them by category (e.g., Programming Languages, ML Frameworks, Statistics, Cloud/MLOps, etc.).

### 2. Skill Gaps
List skills required by the job descriptions that are absent or underrepresented in your CV.
For each gap, note:
- Which JD(s) require it
- Whether it is a **critical blocker**, **significant gap**, or **nice-to-have**

### 3. Top 3 Priority Gaps
Identify the three most impactful gaps to close first, ranked by:
  (a) frequency across the retrieved JDs
  (b) typical recruiter weight in shortlisting decisions

### 4. Recommended Learning Path
For each of the top 3 gaps, suggest:
- A specific resource (course, project type, certification)
- Realistic time estimate to reach a demonstrable level
- How to evidence the skill on a CV or GitHub

Be honest and specific.  Where the CV is genuinely strong, say so."""


def _build_qa_prompt(question: str, job_docs: List[Document]) -> str:
    """Construct the user-turn prompt for a free-form question."""
    jd_block = "\n\n---\n\n".join(
        f"**{doc.metadata.get('role', 'Unknown Role')}** (source: {doc.metadata.get('source', '')})\n{doc.page_content}"
        for doc in job_docs
    )

    return f"""Using the following UK data science job descriptions as your knowledge base,
please answer this question:

**Question:** {question}

## Job Descriptions
{jd_block}

Please ground your answer in the job descriptions above and cite specific roles
where relevant."""


# ---------------------------------------------------------------------------
# Main API functions
# ---------------------------------------------------------------------------

def retrieve_relevant_jobs(
    query: str,
    vectorstore: FAISS,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
) -> List[Document]:
    """
    Retrieve the top-k most relevant (and diverse) job description chunks
    using Max Marginal Relevance.

    Parameters
    ----------
    query      : The CV text or question string to embed and match against.
    vectorstore: FAISS index built from job description chunks.
    k          : Final number of documents to return.
    fetch_k    : Candidate pool size before MMR re-ranking.  Larger values
                 give MMR more to choose from; 20 is a good default for a
                 small corpus where the full index fits in RAM.
    lambda_mult: MMR balance parameter.  0.0 = pure diversity, 1.0 = pure
                 similarity, 0.6 = moderate relevance bias.
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def stream_gap_analysis(
    cv_text: str,
    job_docs: List[Document],
    api_key: str,
    model: str = "claude-opus-4-6",
) -> Generator[str, None, None]:
    """
    Stream a gap analysis from Claude Opus 4.6.

    Yields incremental text chunks as they arrive from the API so the
    Streamlit UI can display them in real time without waiting for the
    full response.

    Adaptive thinking is enabled so Claude can reason carefully through
    skill inference before writing the final answer.
    """
    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_gap_analysis_prompt(cv_text, job_docs)

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            # Only yield text deltas (not thinking blocks)
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                yield event.delta.text


def stream_qa(
    question: str,
    job_docs: List[Document],
    api_key: str,
    model: str = "claude-opus-4-6",
) -> Generator[str, None, None]:
    """
    Stream an answer to a free-form question about the job market.
    """
    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_qa_prompt(question, job_docs)

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                yield event.delta.text
