# rag/settings.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    COLLECTION_NAME: str = "docudesk"
    PERSIST_DIR: str = "/private/tmp/docudesk-ai/chroma"

    CHUNK_SIZE: int = 650
    CHUNK_OVERLAP: int = 100

    OLLAMA_MODEL: str = "qwen2.5:14b-instruct"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "mxbai-embed-large"

    TOP_K: int = 12
    FETCH_K: int = 120
    MMR_LAMBDA: float = 0.25
    BM25_TOP_K: int = 40
    BM25_BATCH: int = 500

    TEMPERATURE: float = 0.0

    OCR_ENABLED: bool = True
    OCR_LANG: str = "spa+eng"
    OCR_DPI: int = 250
    OCR_MIN_TEXT_CHARS: int = 40

    DEBUG_RAG: bool = True

SETTINGS = Settings()