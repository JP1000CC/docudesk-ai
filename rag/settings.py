# rag/settings.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    COLLECTION_NAME: str = "docudesk"
    PERSIST_DIR: str = "/private/tmp/docudesk-ai/chroma"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    OLLAMA_MODEL: str = "llama3.2:3b"

    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    TOP_K: int = 4

    OCR_ENABLED: bool = True
    OCR_LANG: str = "spa+eng"
    OCR_DPI: int = 250
    OCR_MIN_TEXT_CHARS: int = 40

SETTINGS = Settings()