# rag/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.settings import SETTINGS
from rag.store import get_embeddings, get_vectorstore


def _ocr_pdf_page(pdf_path: Path, page_number_1based: int) -> str:
    """
    Convierte 1 página del PDF a imagen y aplica OCR.
    Requiere: poppler (pdf2image) + tesseract instalado.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=SETTINGS.OCR_DPI,
        first_page=page_number_1based,
        last_page=page_number_1based,
        fmt="png",
        thread_count=1,
    )
    if not images:
        return ""

    img = images[0]
    text = pytesseract.image_to_string(img, lang=SETTINGS.OCR_LANG)
    return text or ""


def _load_pdf_pages_with_ocr(pdf_path: Path) -> List[Document]:
    """
    - Intenta extraer texto con pdfplumber (rápido).
    - Si una página está vacía o casi vacía -> OCR por página.
    """
    docs: List[Document] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            used_ocr = False

            if SETTINGS.OCR_ENABLED and len(text) < SETTINGS.OCR_MIN_TEXT_CHARS:
                try:
                    text = _ocr_pdf_page(pdf_path, i).strip()
                    used_ocr = True
                except Exception:
                    # Si OCR falla en esa página, seguimos
                    text = ""

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": pdf_path.name,
                        "source": str(pdf_path),
                        "page": i,
                        "ocr": used_ocr,
                    },
                )
            )

    return docs


def ingest_pdfs(pdf_paths: Iterable[Path]) -> Dict[str, int]:
    """
    Ingesta PDFs (con OCR fallback), chunking y upsert a Chroma.
    Retorna stats para UI.
    """
    pdf_paths = list(pdf_paths)
    all_docs: List[Document] = []

    for p in pdf_paths:
        p = Path(p)
        if not p.exists():
            continue
        all_docs.extend(_load_pdf_pages_with_ocr(p))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.CHUNK_SIZE,
        chunk_overlap=SETTINGS.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)

    if chunks:
        vs.add_documents(chunks)

    return {
        "files": len(pdf_paths),
        "pages": len(all_docs),     # páginas con texto (incluye OCR)
        "chunks": len(chunks),
    }