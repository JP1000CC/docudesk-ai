# app.py
import shutil
from pathlib import Path

import streamlit as st

from rag.ingest import ingest_pdfs
from rag.qa import answer, stream_answer
from rag.settings import SETTINGS

st.set_page_config(page_title="DocuDesk AI (MVP)", layout="wide")

st.title("📄 DocuDesk AI — MVP")
st.caption("Sube PDFs → se indexan → preguntas y respuestas (RAG)")

DATA_DIR = Path("data/pdfs")
PERSIST_DIR = Path(SETTINGS.PERSIST_DIR)

DATA_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.subheader("Configuración")
    st.write(f"**Colección:** {SETTINGS.COLLECTION_NAME}")
    st.write(f"**Chroma dir:** {SETTINGS.PERSIST_DIR}")
    st.write(f"**Embeddings:** {SETTINGS.EMBEDDING_MODEL}")
    st.write(f"**Ollama model:** {SETTINGS.OLLAMA_MODEL}")

    st.divider()

    st.write("Tip: si no responde, asegúrate que Ollama está corriendo y el modelo existe.")

    st.divider()
    st.subheader("Mantenimiento")

    # Limpia el índice persistido de Chroma (para que no “mezcle” PDFs viejos)
    if st.button("🧹 Limpiar índice (Chroma)"):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        st.success("Índice borrado. Ahora ingesta PDFs de nuevo.")
        st.stop()

    # Borra PDFs guardados localmente
    if st.button("🗑️ Borrar PDFs subidos"):
        if DATA_DIR.exists():
            for f in DATA_DIR.glob("*.pdf"):
                try:
                    f.unlink()
                except Exception:
                    pass
        st.success("PDFs borrados.")
        st.stop()

uploaded = st.file_uploader(
    "Sube uno o varios PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("📥 Ingestar PDFs", disabled=not uploaded):
        try:
            saved_paths = []
            for f in uploaded:
                out_path = DATA_DIR / f.name
                out_path.write_bytes(f.getvalue())
                saved_paths.append(out_path)

            stats = ingest_pdfs(saved_paths)
            st.success(
                f"Listo: {stats['files']} PDF(s), {stats['pages']} páginas, "
                f"{stats['chunks']} chunks indexados."
            )
        except Exception as e:
            st.error(f"Error ingestando: {e}")

with col2:
    question = st.text_input("Pregunta", placeholder="Ej: ¿Qué dice el documento sobre X?")
    mode = st.radio("Modo respuesta", ["Normal", "En vivo (stream)"], horizontal=True)

    if st.button("🤖 Preguntar", disabled=not question.strip()):
        try:
            st.write("**Respuesta:**")

            if mode == "En vivo (stream)":
                gen, sources = stream_answer(question)
                st.write_stream(gen)
            else:
                out, sources = answer(question)
                st.write(out)

            # Dedup de fuentes (por filename + page)
            st.write("**Fuentes usadas:**")
            seen = set()
            for s in sources:
                key = (s.get("filename"), s.get("page"))
                if key in seen:
                    continue
                seen.add(key)

                if s.get("page"):
                    st.write(f"- {s['filename']} (p. {s['page']})")
                else:
                    st.write(f"- {s['filename']}")

        except Exception as e:
            st.error(f"Error: {e}")