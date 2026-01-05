from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.settings import SETTINGS

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=SETTINGS.EMBEDDING_MODEL)

def get_vectorstore(embeddings):
    return Chroma(
        collection_name=SETTINGS.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=SETTINGS.PERSIST_DIR,
    )
