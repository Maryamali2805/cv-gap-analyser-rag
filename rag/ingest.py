"""
ingest.py
---------
Loads job-description text files from disk and splits them into chunks
suitable for embedding.

Design notes
~~~~~~~~~~~~
Each job description is ~600–900 words. We keep each file as a single
Document so retrieval returns a whole JD (preserving context) and also
create overlapping chunks of ~400 tokens for finer-grained similarity
matching.  The two representations are merged: the per-file document goes
into the index for "what does this role look like" queries, while chunks
let us also surface specific requirement sections.

In practice, with only ~8 JDs and sub-second embedding, we don't *need*
chunking for speed — but it demonstrates the pipeline for a corpus that
would otherwise overflow a single context window.
"""

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_job_descriptions(data_dir: str = "data/jobs") -> List[Document]:
    """
    Read every *.txt file in *data_dir* and return one Document per file.

    Metadata stored per document
    ----------------------------
    source : str  — filename (e.g. "01_senior_data_scientist_fintech.txt")
    role   : str  — human-readable title derived from the filename stem
    """
    docs: List[Document] = []
    base = Path(data_dir)

    if not base.exists():
        raise FileNotFoundError(
            f"Job descriptions directory not found: {base.resolve()}\n"
            "Run the app from the ds-job-rag/ root, or check the path."
        )

    for f in sorted(base.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        role = f.stem.split("_", 1)[1].replace("_", " ").title() if "_" in f.stem else f.stem
        docs.append(
            Document(
                page_content=text,
                metadata={"source": f.name, "role": role},
            )
        )

    if not docs:
        raise ValueError(f"No .txt files found in {base.resolve()}")

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 60,
) -> List[Document]:
    """
    Split documents into overlapping chunks for finer similarity matching.

    chunk_size / chunk_overlap tuning
    ----------------------------------
    all-MiniLM-L6-v2 has a 256-token context window (it truncates beyond
    that).  At ~0.75 tokens/word, 500 characters ≈ 90–100 tokens — safely
    under the model limit while still giving each chunk enough semantic
    content for stable embedding geometry.  The 60-char overlap (~10 tokens)
    ensures section boundaries don't split a skill phrase across chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Propagate metadata from parent documents to their chunks
    for chunk in chunks:
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("role", "unknown")

    return chunks
