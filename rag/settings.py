from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    COLLECTION_NAME: str = "docudesk"
    PERSIST_DIR: str = "/private/tmp/docudesk-ai/chroma"  # o tu ruta final

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_MODEL: str = "llama3.1"
    TOP_K: int = 4

    # OCR (MVP+)
    OCR_ENABLED: bool = True
    OCR_LANG: str = "spa+eng"     # si no tienes spa, pon "eng"
    OCR_DPI: int = 250            # 200-300 recomendado
    OCR_MIN_TEXT_CHARS: int = 40  # si una página trae menos que esto -> OCR

SETTINGS = Settings()