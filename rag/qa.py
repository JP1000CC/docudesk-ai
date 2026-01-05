# rag/qa.py
from __future__ import annotations

from typing import Generator, List, Tuple, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.settings import SETTINGS
from rag.store import get_embeddings, get_vectorstore

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente de preguntas y respuestas sobre documentos.\n"
     "Responde SOLO usando el CONTEXTO.\n"
     "Si el contexto no contiene la respuesta, di claramente: 'No lo sé con los documentos cargados.'\n"
     "Responde en español, directo y concreto."),
    ("human",
     "PREGUNTA:\n{question}\n\n"
     "CONTEXTO:\n{context}\n")
])


def _page_to_1based(page_val) -> Optional[int]:
    """
    En tu ingest guardas page 1-based (i empieza en 1), así que:
    - si es int, lo devolvemos tal cual
    """
    if isinstance(page_val, int):
        return page_val
    return None


def _doc_source_meta(d) -> dict:
    filename = d.metadata.get("filename") or d.metadata.get("source", "unknown")
    page = _page_to_1based(d.metadata.get("page"))
    ocr = bool(d.metadata.get("ocr", False))
    return {"filename": filename, "page": page, "ocr": ocr}


def _format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = _doc_source_meta(d)
        ocr_tag = " OCR" if meta["ocr"] else ""
        page_txt = f" p.{meta['page']}" if meta["page"] else ""
        parts.append(f"[{i}] ({meta['filename']}{page_txt}{ocr_tag})\n{d.page_content}")
    return "\n\n".join(parts).strip()


def retrieve(question: str, k: int | None = None):
    """Recupera docs sin score (modo retriever)."""
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": k or SETTINGS.TOP_K})
    docs = retriever.invoke(question)
    return docs


def retrieve_with_score(question: str, k: int | None = None):
    """
    Recupera docs CON score cuando Chroma lo soporta.
    En Chroma normalmente el score es distancia (menor = mejor).
    """
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    k_ = k or SETTINGS.TOP_K

    try:
        results = vs.similarity_search_with_score(question, k=k_)
        # results: List[Tuple[Document, score]]
        return results
    except Exception:
        docs = retrieve(question, k=k_)
        return [(d, None) for d in docs]


def answer(question: str, k: int | None = None, strict: bool = True) -> Tuple[str, List[dict]]:
    """
    strict=True:
      - Si no hay docs o el contexto queda vacío -> devuelve "No lo sé..." sin llamar al LLM.
    """
    results = retrieve_with_score(question, k=k)
    docs = [d for d, _s in results]
    context = _format_context(docs)

    if strict and (not docs or not context):
        return "No lo sé con los documentos cargados.", []

    llm = ChatOllama(model=SETTINGS.OLLAMA_MODEL, temperature=0)
    msg = PROMPT.format_messages(question=question, context=context)
    out = llm.invoke(msg).content

    sources: List[dict] = []
    for d, score in results:
        meta = _doc_source_meta(d)
        sources.append({
            "filename": meta["filename"],
            "page": meta["page"],     # 1-based
            "ocr": meta["ocr"],
            "score": score,           # puede ser None si no disponible
        })

    return out, sources


def stream_answer(question: str, k: int | None = None, strict: bool = True) -> Tuple[Generator[str, None, None], List[dict]]:
    results = retrieve_with_score(question, k=k)
    docs = [d for d, _s in results]
    context = _format_context(docs)

    if strict and (not docs or not context):
        def gen_empty():
            yield "No lo sé con los documentos cargados."
        return gen_empty(), []

    llm = ChatOllama(model=SETTINGS.OLLAMA_MODEL, temperature=0)
    msg = PROMPT.format_messages(question=question, context=context)

    def gen():
        for chunk in llm.stream(msg):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    sources: List[dict] = []
    for d, score in results:
        meta = _doc_source_meta(d)
        sources.append({
            "filename": meta["filename"],
            "page": meta["page"],     # 1-based
            "ocr": meta["ocr"],
            "score": score,
        })

    return gen(), sources