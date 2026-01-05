from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag.settings import SETTINGS

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=SETTINGS.EMBEDDING_MODEL)

def get_vectorstore(embeddings):
    persist_dir = Path(SETTINGS.PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # test write permission (falla rápido con mensaje claro)
    test_file = persist_dir / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(f"No tengo permisos de escritura en {persist_dir}. Error: {e}")

    return Chroma(
        collection_name=SETTINGS.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
