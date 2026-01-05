from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    COLLECTION_NAME: str = "docudesk"
    PERSIST_DIR: str = "/tmp/docudesk-ai/chroma"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_MODEL: str = "llama3.1"
    TOP_K: int = 4

SETTINGS = Settings()
