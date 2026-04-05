# UK Data Science Job Gap Analyser

A Retrieval-Augmented Generation (RAG) application that compares your CV against a curated collection of UK data science job descriptions and returns a structured gap analysis — what you already have, what you're missing, and a concrete learning path.

---

## Architecture

```mermaid
flowchart TD
    subgraph Offline["Offline — build once"]
        A[("📁 data/jobs/\n*.txt files")] -->|load_job_descriptions| B["LangChain Documents\n(8 JDs with metadata)"]
        B -->|chunk_documents\n500-char chunks, 60-char overlap| C["Chunked Documents\n~40–50 chunks"]
        C -->|all-MiniLM-L6-v2\nsentence-transformers| D["Dense vectors\n384-dim, L2-normalised"]
        D -->|FAISS IndexFlatIP\n(cosine via dot-product)| E[("💾 faiss_index/\nindex.faiss + index.pkl")]
    end

    subgraph Online["Online — per query"]
        F["User pastes CV\nor types a question"] -->|embed with all-MiniLM-L6-v2| G["Query vector\n384-dim"]
        G -->|MMR search\nk=5, λ=0.6, fetch_k=20| E
        E -->|top-k diverse chunks| H["Retrieved JD chunks\n(with metadata)"]
        H --> I["Prompt constructor\n(CV + JDs + structured template)"]
        F --> I
        I -->|streaming API call\nadaptive thinking| J["Claude Opus 4.6"]
        J -->|text_delta stream| K["Streamlit UI\nreal-time markdown render"]
    end
```

---

## How it Works

### 1. Ingestion (`rag/ingest.py`)

Eight UK job descriptions are stored as plain-text files in `data/jobs/`. Each file is loaded as a LangChain `Document` and split using `RecursiveCharacterTextSplitter` into overlapping 500-character chunks.

**Why chunk at all?** The embedding model (`all-MiniLM-L6-v2`) has a 256-token context window. An 800-word job description is ~1,000 tokens — well beyond that window. The model truncates any input over its limit, silently degrading the embedding. Chunking at ~90–100 tokens (≈500 characters) keeps each chunk comfortably within the window, ensuring high-fidelity embeddings for every section of every JD.

The 60-character overlap (~10 tokens) prevents skill phrases from being split across chunk boundaries.

### 2. Embeddings — the maths

`sentence-transformers/all-MiniLM-L6-v2` is a 6-layer, 22M-parameter transformer fine-tuned with a **multiple negatives ranking (MNR) loss** on 1 billion sentence pairs. MNR loss trains the model by treating every other sentence in a mini-batch as a negative example:

```
L = -log( exp(sim(q, p⁺)/τ) / Σⱼ exp(sim(q, pⱼ)/τ) )
```

where `sim` is cosine similarity, `τ` is a temperature scalar, `p⁺` is the positive pair, and `pⱼ` ranges over all pairs in the batch. This pushes semantically similar sentences close together in the 384-dimensional embedding space and unrelated sentences far apart.

Critically, we **L2-normalise** all vectors before storage (via `normalize_embeddings=True` in the LangChain wrapper). This maps every vector onto the unit hypersphere, making cosine similarity equivalent to the dot product: `cos(u,v) = u·v` when ‖u‖=‖v‖=1. FAISS's `IndexFlatIP` (inner product index) then gives exact cosine search in O(n) time.

### 3. FAISS — index type choice

| Index type | Search time | Memory | Use case |
|---|---|---|---|
| `IndexFlatIP` | O(n) — exact | n × 384 × 4B | < 100K vectors |
| `IndexIVFFlat` | O(n/nlist) — approximate | same + IVF overhead | 100K – 10M |
| `HNSW` | O(log n) — approximate | ~2× | > 1M, latency-critical |

With ~50 chunks, `IndexFlatIP` is the right choice — it is exact, requires no training, and brute-force over 50 × 384 floats completes in microseconds.

### 4. Max Marginal Relevance retrieval

Plain top-k cosine search returns the k vectors closest to the query — but if two job descriptions are highly similar (e.g., two "Data Scientist at a bank" roles), both slots may be occupied by near-duplicate documents. **MMR** avoids this by iteratively selecting the document that maximises:

```
MMR(d) = λ · sim(q, d) − (1 − λ) · max_{r ∈ R} sim(r, d)
```

`R` is the set of already-selected documents. At each iteration, a document scores highly if it is both relevant to the query *and* different from what has already been selected. With `λ=0.6`, relevance is weighted 1.5× more than diversity — a sensible default that ensures the retrieved set covers the query while avoiding echo-chamber selection.

The `fetch_k=20` parameter sets the initial candidate pool from which MMR draws. A larger pool gives MMR more diversity to choose from; `fetch_k = 3k` to `5k` is a good heuristic.

### 5. Generation — Claude Opus 4.6

Retrieved chunks are assembled into a structured prompt asking Claude to:

1. List skills already present in the CV
2. Identify gaps against the JDs, annotated with severity (blocker / significant / nice-to-have)
3. Rank the top 3 priority gaps by market frequency
4. Suggest a concrete learning path for each

**Adaptive thinking** is enabled. Claude decides internally how much chain-of-thought reasoning to invest. This is important for skill inference: a CV may mention "distributed computing" without naming Spark, or list "scikit-learn" without explicitly claiming "ML modelling". Extended thinking lets Claude work through these implications before generating the visible response.

**Streaming** (`client.messages.stream`) yields `text_delta` events incrementally. Streamlit's `st.empty()` container is updated on each chunk, so users see the response build word-by-word rather than waiting for a single large payload.

---

## Project Structure

```
ds-job-rag/
├── app.py                          # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                   # Document loading + chunking
│   ├── vectorstore.py              # FAISS index build/load
│   └── chain.py                    # MMR retrieval + Claude streaming
│
└── data/
    └── jobs/
        ├── 01_senior_data_scientist_fintech.txt
        ├── 02_ml_engineer_tech.txt
        ├── 03_data_scientist_nhs_healthcare.txt
        ├── 04_nlp_engineer.txt
        ├── 05_computer_vision_engineer.txt
        ├── 06_junior_data_scientist_ecommerce.txt
        ├── 07_mlops_engineer.txt
        └── 08_research_scientist_ai.txt
```

---

## Setup

```bash
# 1. Clone / navigate to the project directory
cd ds-job-rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
# Alternatively, enter the key in the sidebar when the app is running.

# 5. Run
streamlit run app.py
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace and builds the FAISS index. Both are cached locally; subsequent starts are fast.

---

## Extending the corpus

1. Add new `.txt` files to `data/jobs/`.
2. In the sidebar, click **"Rebuild vector index"** to re-embed everything.
3. The new roles are immediately available for retrieval.

No schema changes, no database migrations — the only artefact is the `faiss_index/` directory, which is rebuilt from the source text files.

---

## Design decisions and trade-offs

| Decision | Choice | Alternative considered | Reason |
|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` (local) | OpenAI `text-embedding-3-small` | No API cost, no data egress, fast on CPU |
| LLM | Claude Opus 4.6 | GPT-4o | Stronger instruction-following for structured analysis; native adaptive thinking |
| Vector store | FAISS (local) | Pinecone / Chroma | Zero infra cost; `faiss_index/` dir is self-contained |
| Retrieval | MMR | Plain cosine top-k | Prevents near-duplicate JDs consuming all k slots |
| Chunking | 500-char / 60-char overlap | Full-document (no chunking) | Keeps chunks inside `all-MiniLM-L6-v2`'s 256-token window |
| Streaming | Yes | Batch request | UX: users see output building rather than waiting 10–15s |
| Thinking | Adaptive | Fixed budget | Claude decides effort level; avoids over- or under-thinking |

---

## Limitations

- **Corpus size**: 8 JDs is illustrative. A production system would scrape hundreds of live postings from Reed, LinkedIn, or the Guardian Jobs API, embedding and indexing them nightly.
- **Embedding model**: `all-MiniLM-L6-v2` was trained on general web text, not specifically on job market language. A model fine-tuned on job descriptions (e.g., via contrastive training on skill-to-JD pairs) would improve retrieval precision.
- **CV parsing**: The app accepts raw text. Real CVs in Word/PDF format would need extraction (e.g., `pdfplumber`, `python-docx`) before passing to the pipeline.
- **Hallucination risk**: LLMs can infer skill requirements not in the retrieved JDs. The prompt asks Claude to cite specific JDs for each gap, which partially mitigates this, but the analysis should be treated as a starting point for human review.
