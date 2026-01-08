# rag/store.py
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.settings import SETTINGS


def get_embeddings():
    """
    Embeddings 100% locales con Ollama.

    Asegúrate de tener el modelo:
      ollama pull mxbai-embed-large
    """
    embed_model = getattr(SETTINGS, "OLLAMA_EMBED_MODEL", "mxbai-embed-large")
    base_url = getattr(SETTINGS, "OLLAMA_BASE_URL", None)

    if base_url:
        return OllamaEmbeddings(model=embed_model, base_url=base_url)

    return OllamaEmbeddings(model=embed_model)


def get_vectorstore(embeddings):
    persist_dir = Path(SETTINGS.PERSIST_DIR).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    # test write permission (falla rápido con mensaje claro)
    test_file = persist_dir / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"No tengo permisos de escritura en {persist_dir}. Error: {e}"
        )

    return Chroma(
        collection_name=SETTINGS.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )