from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.settings import SETTINGS
from rag.store import get_embeddings, get_vectorstore

def ingest_pdfs(pdf_paths: Iterable[str | Path]) -> dict:
    pdf_paths = [Path(p) for p in pdf_paths]
    for p in pdf_paths:
        if not p.exists() or p.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF no encontrado o inválido: {p}")

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.CHUNK_SIZE,
        chunk_overlap=SETTINGS.CHUNK_OVERLAP,
    )

    all_chunks = []
    file_count = 0
    page_count = 0

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()  # docs por página con metadata (source, page)
        file_count += 1
        page_count += len(docs)

        # Normaliza metadata útil
        for d in docs:
            d.metadata["filename"] = pdf_path.name

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if all_chunks:
        vectorstore.add_documents(all_chunks)

    return {
        "files": file_count,
        "pages": page_count,
        "chunks": len(all_chunks),
    }
