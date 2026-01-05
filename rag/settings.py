from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    COLLECTION_NAME: str = "docudesk"
    PERSIST_DIR: str = "storage/chroma"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Embeddings (local)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM (Ollama local)
    OLLAMA_MODEL: str = "llama3.1"
    TOP_K: int = 4

SETTINGS = Settings()
