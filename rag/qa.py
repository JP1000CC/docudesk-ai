from __future__ import annotations

from typing import Generator, List, Tuple

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

def _format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("filename") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_txt = f"p.{page + 1}" if isinstance(page, int) else ""
        parts.append(f"[{i}] ({src} {page_txt})\n{d.page_content}")
    return "\n\n".join(parts)

def retrieve(question: str, k: int | None = None):
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": k or SETTINGS.TOP_K})
    docs = retriever.invoke(question)
    return docs

def answer(question: str, k: int | None = None) -> Tuple[str, List[dict]]:
    docs = retrieve(question, k=k)
    context = _format_context(docs)

    llm = ChatOllama(model=SETTINGS.OLLAMA_MODEL, temperature=0)
    msg = PROMPT.format_messages(question=question, context=context)
    out = llm.invoke(msg).content

    sources = []
    for d in docs:
        sources.append({
            "filename": d.metadata.get("filename") or d.metadata.get("source", "unknown"),
            "page": (d.metadata.get("page", None) + 1) if isinstance(d.metadata.get("page", None), int) else None
        })

    return out, sources

def stream_answer(question: str, k: int | None = None) -> Tuple[Generator[str, None, None], List[dict]]:
    docs = retrieve(question, k=k)
    context = _format_context(docs)

    llm = ChatOllama(model=SETTINGS.OLLAMA_MODEL, temperature=0)
    msg = PROMPT.format_messages(question=question, context=context)

    def gen():
        for chunk in llm.stream(msg):
            # chunk suele ser AIMessageChunk
            text = getattr(chunk, "content", None)
            if text:
                yield text

    sources = []
    for d in docs:
        sources.append({
            "filename": d.metadata.get("filename") or d.metadata.get("source", "unknown"),
            "page": (d.metadata.get("page", None) + 1) if isinstance(d.metadata.get("page", None), int) else None
        })

    return gen(), sources
