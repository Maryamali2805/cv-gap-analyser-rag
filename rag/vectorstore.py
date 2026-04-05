"""
vectorstore.py
--------------
Builds and persists a FAISS vector index from job-description chunks.

Why FAISS + all-MiniLM-L6-v2?
------------------------------
FAISS (Facebook AI Similarity Search) stores dense float32 vectors and
answers k-nearest-neighbour queries in sub-millisecond time for corpora
of this size.  The default index type used here — IndexFlatL2 (exact
L2 search, no approximation) — is correct for a corpus of <10,000 vectors
where speed is not a bottleneck.  For larger corpora you would switch to
IndexIVFFlat (inverted-file approximate search) or HNSW (graph-based
approximate search), trading a small recall loss for O(log n) query time.

The embedding model `all-MiniLM-L6-v2` maps sentences to 384-dimensional
vectors trained with a contrastive objective (multiple negatives ranking
loss) on 1B sentence pairs.  Because cosine similarity is used at query
time, vectors are L2-normalised before storage — which makes cosine
similarity equivalent to dot-product similarity, allowing FAISS's
IndexFlatIP to serve as the effective backend.

LangChain's FAISS wrapper normalises vectors automatically when
`distance_strategy = DistanceStrategy.COSINE` is set (it calls
`faiss.normalize_L2` on all embeddings before adding them to
IndexFlatIP).
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_DIR = "faiss_index"


def get_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance backed by all-MiniLM-L6-v2.

    The model is downloaded to HuggingFace's cache (~90 MB) on first run.
    Subsequent runs load from disk instantly.

    *device* can be "cuda" if a GPU is available — embeddings compute ~10x
    faster on GPU, which matters when re-indexing large corpora.
    """
    return HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # cosine sim via dot product
    )


def build_vectorstore(
    docs: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_dir: str = _INDEX_DIR,
) -> FAISS:
    """
    Embed *docs* and create a FAISS index.  Saves to *persist_dir* so the
    index survives across Streamlit reruns.

    Returns the FAISS vectorstore object.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_dir: str = _INDEX_DIR,
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from *persist_dir*.
    Returns None if the index does not exist yet.
    """
    index_path = Path(persist_dir)
    if not (index_path / "index.faiss").exists():
        return None

    if embeddings is None:
        embeddings = get_embeddings()

    return FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # we built this index ourselves
    )


def get_or_build_vectorstore(
    docs: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_dir: str = _INDEX_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Load the existing FAISS index if it exists; otherwise build and save it.

    *force_rebuild=True* is useful when you add new job descriptions and
    want to re-embed everything from scratch.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if not force_rebuild:
        existing = load_vectorstore(embeddings, persist_dir)
        if existing is not None:
            return existing

    return build_vectorstore(docs, embeddings, persist_dir)
