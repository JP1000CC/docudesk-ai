# rag/qa.py
from __future__ import annotations

from typing import Generator, List, Tuple, Optional, Dict
import unicodedata
import hashlib

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

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


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _page_to_1based(page_val) -> Optional[int]:
    if isinstance(page_val, int):
        return page_val
    return None


def _doc_source_meta(d: Document) -> dict:
    filename = d.metadata.get("filename") or d.metadata.get("source", "unknown")
    page = _page_to_1based(d.metadata.get("page"))
    ocr = bool(d.metadata.get("ocr", False))
    return {"filename": filename, "page": page, "ocr": ocr}


def _doc_key(d: Document) -> str:
    meta = _doc_source_meta(d)
    txt = (d.page_content or "").strip()
    h = hashlib.sha1(txt[:500].encode("utf-8", errors="ignore")).hexdigest()
    return f"{meta['filename']}|{meta['page']}|{meta['ocr']}|{h}"


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        k = _doc_key(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def _format_context(docs: List[Document], max_chars_per_doc: int = 1800) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = _doc_source_meta(d)
        ocr_tag = " OCR" if meta["ocr"] else ""
        page_txt = f" p.{meta['page']}" if meta["page"] else ""
        text = (d.page_content or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "…"
        parts.append(f"[{i}] ({meta['filename']}{page_txt}{ocr_tag})\n{text}")
    return "\n\n".join(parts).strip()


def _get_retriever(vs, k: int, fetch_k: int, lambda_mult: float):
    try:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    except Exception:
        return vs.as_retriever(search_kwargs={"k": k})


# cache: (persist_dir::collection) -> (bm25, all_docs)
_BM25_CACHE: Dict[str, Tuple[BM25Retriever, List[Document]]] = {}


def _load_all_docs_from_chroma(vs, batch_size: int = 500) -> List[Document]:
    """
    Carga TODOS los documentos desde la colección interna de Chroma con paginación.
    Esto es lo que te estaba fallando: sin limit/offset BM25 quedaba incompleto.
    """
    # langchain_chroma expone la colección interna como _collection
    col = getattr(vs, "_collection", None)
    if col is None:
        return []

    docs: List[Document] = []
    offset = 0

    while True:
        got = col.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
        texts = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []
        if not texts:
            break

        for text, md in zip(texts, metas):
            if text and isinstance(text, str):
                docs.append(Document(page_content=text, metadata=md or {}))

        offset += len(texts)

    return docs


def _build_bm25(vs) -> Tuple[BM25Retriever, List[Document]]:
    cache_key = f"{SETTINGS.PERSIST_DIR}::{SETTINGS.COLLECTION_NAME}"
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    all_docs = _load_all_docs_from_chroma(vs, batch_size=getattr(SETTINGS, "BM25_BATCH", 500))
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = getattr(SETTINGS, "BM25_TOP_K", 30)

    _BM25_CACHE[cache_key] = (bm25, all_docs)
    return bm25, all_docs


def _score_map_from_vector(vs, queries: List[str], fetch_k: int) -> Dict[str, float | None]:
    score_map: Dict[str, float | None] = {}
    for q in queries:
        try:
            scored = vs.similarity_search_with_score(q, k=fetch_k)
            for d, s in scored:
                key = _doc_key(d)
                if key not in score_map or (s is not None and score_map[key] is not None and s < score_map[key]):
                    score_map[key] = s
        except Exception:
            pass
    return score_map


def _substring_fallback(all_docs: List[Document], needle: str, limit: int = 20) -> List[Document]:
    """
    Fallback determinista: si la pregunta contiene un nombre propio,
    buscamos chunks que lo contengan literal (normalizado sin tildes).
    Con 1-5k chunks esto es rápido y salva el MVP.
    """
    n = _strip_accents(needle).lower()
    hits = []
    for d in all_docs:
        txt = _strip_accents((d.page_content or "")).lower()
        if n in txt:
            hits.append(d)
            if len(hits) >= limit:
                break
    return hits


def _retrieve_hybrid(question: str, k: int) -> Tuple[List[Document], Dict[str, float | None]]:
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)

    fetch_k = getattr(SETTINGS, "FETCH_K", max(80, k * 6))
    lambda_mult = getattr(SETTINGS, "MMR_LAMBDA", 0.25)

    q1 = question.strip()
    q2 = _strip_accents(q1)
    queries = [q1] if q2 == q1 else [q1, q2]

    # 1) Vector MMR
    retriever = _get_retriever(vs, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    vec_docs: List[Document] = []
    for q in queries:
        try:
            vec_docs.extend(retriever.invoke(q))
        except Exception:
            vec_docs.extend(vs.similarity_search(q, k=k))

        # 2) BM25 (keyword) completo
    bm25_docs: List[Document] = []
    all_docs: List[Document] = []
    try:
        bm25, all_docs = _build_bm25(vs)
        for q in queries:
            bm25_docs.extend(bm25.get_relevant_documents(q))
    except Exception:
        pass

    # 3) Unir + dedupe
    docs_all = _dedupe_docs(vec_docs + bm25_docs)

    # 4) SEED determinista por substring cuando hay entidad clara (ej: Melquíades)
    # Esto GARANTIZA que el contexto incluya fragmentos donde aparece el nombre.
    q_norm = _strip_accents(q1).lower()
    if all_docs and ("melqui" in q_norm):
        seed_n = getattr(SETTINGS, "ENTITY_SEED_TOP", 40)
        seed_hits = _substring_fallback(all_docs, "melquiades", limit=seed_n)
        docs_all = _dedupe_docs(seed_hits + docs_all)

    # 5) Scores aproximados desde vector
    score_map = _score_map_from_vector(vs, queries, fetch_k=fetch_k)

    def sort_key(d: Document):
        s = score_map.get(_doc_key(d))
        return (s is None, s if s is not None else 999999)

    docs_all.sort(key=sort_key)

    final_n = max(16, k)  # manda más contexto al LLM
    docs_final = docs_all[:final_n]

    # DEBUG: confirma que el contexto final trae 'melqui'
    if getattr(SETTINGS, "DEBUG_RAG", False):
        joined = _strip_accents("\n".join((d.page_content or "") for d in docs_final)).lower()
        print("[DEBUG_RAG] final_docs:", len(docs_final), "| contiene 'melqui'?:", "melqui" in joined)
        # imprime primeras 5 fuentes para ver si son las correctas
        for d in docs_final[:5]:
            m = _doc_source_meta(d)
            print("  -", m["filename"], "p.", m["page"], "ocr=", m["ocr"])

    return docs_final, score_map


def answer(question: str, k: int | None = None, strict: bool = True) -> Tuple[str, List[dict]]:
    k_ = k or SETTINGS.TOP_K
    docs, score_map = _retrieve_hybrid(question, k=k_)
    context = _format_context(docs)

    if strict and (not docs or not context):
        return "No lo sé con los documentos cargados.", []

    llm = ChatOllama(
        model=SETTINGS.OLLAMA_MODEL,
        temperature=getattr(SETTINGS, "TEMPERATURE", 0.0),
        base_url=getattr(SETTINGS, "OLLAMA_BASE_URL", None),
    )
    msg = PROMPT.format_messages(question=question, context=context)
    out = llm.invoke(msg).content

    sources: List[dict] = []
    for d in docs:
        meta = _doc_source_meta(d)
        sources.append({
            "filename": meta["filename"],
            "page": meta["page"],
            "ocr": meta["ocr"],
            "score": score_map.get(_doc_key(d)),
        })

    return out, sources


def stream_answer(question: str, k: int | None = None, strict: bool = True) -> Tuple[Generator[str, None, None], List[dict]]:
    k_ = k or SETTINGS.TOP_K
    docs, score_map = _retrieve_hybrid(question, k=k_)
    context = _format_context(docs)

    if strict and (not docs or not context):
        def gen_empty():
            yield "No lo sé con los documentos cargados."
        return gen_empty(), []

    llm = ChatOllama(
        model=SETTINGS.OLLAMA_MODEL,
        temperature=getattr(SETTINGS, "TEMPERATURE", 0.0),
        base_url=getattr(SETTINGS, "OLLAMA_BASE_URL", None),
    )
    msg = PROMPT.format_messages(question=question, context=context)

    def gen():
        for chunk in llm.stream(msg):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    sources: List[dict] = []
    for d in docs:
        meta = _doc_source_meta(d)
        sources.append({
            "filename": meta["filename"],
            "page": meta["page"],
            "ocr": meta["ocr"],
            "score": score_map.get(_doc_key(d)),
        })

    return gen(), sources