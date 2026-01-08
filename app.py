# app.py
# DocuDesk AI — MVP (PDF -> Chroma -> RAG con Ollama)
#
# Incluye:
# - Indicador de índice (chunks + archivos únicos)
# - Limpieza de PDFs e índice (con aviso cuando hace falta reiniciar Streamlit por cache en memoria)
# - Reset total (PDFs + índice + limpia estado UI)
# - Preguntar habilitado correctamente (keys estables + motivo si está deshabilitado)
# - Deduplicación de fuentes
# - UI más intuitiva

import shutil
from pathlib import Path
from datetime import datetime

import streamlit as st

from rag.ingest import ingest_pdfs
from rag.qa import answer, stream_answer
from rag.settings import SETTINGS
from rag.store import get_embeddings, get_vectorstore

# -----------------------------
# Config UI
# -----------------------------
st.set_page_config(page_title="DocuDesk AI (MVP)", layout="wide")

st.title("📄 DocuDesk AI — MVP")
st.caption("Sube PDFs → se indexan → preguntas y respuestas (RAG)")

DATA_DIR = Path("data/pdfs")
PERSIST_DIR = Path(SETTINGS.PERSIST_DIR).expanduser().resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_rmtree(path: Path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def safe_mkdir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def list_pdfs_in_data_dir() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"), key=lambda p: p.name.lower())


def disk_has_chroma_files() -> bool:
    """Señal simple de si hay algo persistido en disco."""
    try:
        if not PERSIST_DIR.exists():
            return False
        return any(PERSIST_DIR.iterdir())
    except Exception:
        return False


def get_index_stats(max_meta_pull: int = 5000):
    """
    Devuelve:
      - chunks: cantidad de chunks/embeddings en la colección
      - files: cantidad de archivos únicos (si es razonable calcularlo)
    Nota: Chroma puede quedar "cacheado" en memoria hasta reiniciar Streamlit.
    """
    try:
        embeddings = get_embeddings()
        vs = get_vectorstore(embeddings)

        chunks = None
        files = None

        # count de la colección (chunks)
        try:
            chunks = vs._collection.count()
        except Exception:
            chunks = None

        # intenta contar archivos únicos sin explotar memoria
        if isinstance(chunks, int) and chunks > 0 and chunks <= max_meta_pull:
            try:
                data = vs._collection.get(include=["metadatas"])
                metadatas = data.get("metadatas") or []
                uniq = set()
                for m in metadatas:
                    if not m:
                        continue
                    fname = m.get("filename") or m.get("source")
                    if fname:
                        uniq.add(fname)
                files = len(uniq)
            except Exception:
                files = None

        return chunks, files
    except Exception:
        return None, None


def mark_restart_required(reason: str):
    st.session_state["restart_required"] = True
    st.session_state["restart_reason"] = reason
    st.session_state["restart_marked_at"] = now_str()


def clear_restart_required():
    st.session_state.pop("restart_required", None)
    st.session_state.pop("restart_reason", None)
    st.session_state.pop("restart_marked_at", None)


# -----------------------------
# Session defaults
# -----------------------------
st.session_state.setdefault("last_ingest", None)
st.session_state.setdefault("restart_required", False)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Configuración")
    st.write(f"**Colección:** {SETTINGS.COLLECTION_NAME}")
    st.write(f"**Chroma dir:** `{PERSIST_DIR}`")
    st.write(f"**Embeddings:** {SETTINGS.OLLAMA_EMBED_MODEL} (Ollama)")
    st.write(f"**Ollama model:** {SETTINGS.OLLAMA_MODEL}")

    st.divider()

    st.subheader("Estado del índice")
    chunks, files = get_index_stats()
    on_disk = disk_has_chroma_files()

    if chunks is None:
        st.warning("No pude leer el estado del índice (aún).")
    else:
        if files is None:
            st.info(f"Chunks indexados: **{chunks}**")
        else:
            st.info(f"Chunks indexados: **{chunks}**\n\nArchivos únicos: **{files}**")

    st.caption(f"Persistencia en disco: **{'sí' if on_disk else 'no'}**")

    if st.button("🔄 Actualizar indicador", key="btn_refresh_indicator"):
        st.rerun()

    if st.session_state.get("restart_required"):
        st.warning(
            "Para limpiar el cache de Chroma en memoria necesitas reiniciar Streamlit.\n\n"
            f"Motivo: {st.session_state.get('restart_reason', '—')}\n\n"
            "Comando:\n"
            "`pkill -f streamlit && streamlit run app.py`"
        )
        st.caption(f"Marcado: {st.session_state.get('restart_marked_at', '')}")

    st.divider()
    st.write("Tip: si no responde, asegúrate que Ollama está corriendo y el modelo existe.")

    st.divider()
    st.subheader("Mantenimiento")

    pdfs = list_pdfs_in_data_dir()
    st.caption(f"PDFs en `data/pdfs`: **{len(pdfs)}**")

    colA, colB = st.columns(2)

    with colA:
        if st.button("🗑️ Borrar PDFs", key="btn_delete_pdfs", use_container_width=True):
            for f in pdfs:
                try:
                    f.unlink()
                except Exception:
                    pass
            st.session_state["last_ingest"] = None
            st.success("PDFs borrados.")
            st.rerun()

    with colB:
        if st.button("🧹 Limpiar índice", key="btn_clear_index", use_container_width=True):
            # Borra en disco (pero puede quedar cache en memoria)
            safe_rmtree(PERSIST_DIR)
            safe_mkdir(PERSIST_DIR)

            st.session_state["last_ingest"] = None
            mark_restart_required(
                "Se borró el índice en disco. Si el contador aún muestra chunks, es cache en memoria."
            )
            st.success("Índice borrado en disco.")
            st.rerun()

    if st.button("🧨 Reset TOTAL (PDFs + índice)", key="btn_reset_total", use_container_width=True):
        # Borra PDFs
        for f in list_pdfs_in_data_dir():
            try:
                f.unlink()
            except Exception:
                pass

        # Borra índice persistido
        safe_rmtree(PERSIST_DIR)
        safe_mkdir(PERSIST_DIR)

        # Limpia sesión UI completa
        st.session_state.clear()

        # Re-poner flag de reinicio (porque clear() lo borró)
        st.session_state["restart_required"] = True
        st.session_state["restart_reason"] = (
            "Reset total aplicado. Para limpiar el cache de Chroma en memoria, reinicia Streamlit."
        )
        st.session_state["restart_marked_at"] = now_str()

        st.success("Reset total aplicado en disco (PDFs + índice).")
        st.stop()

# -----------------------------
# Main UI
# -----------------------------
uploaded = st.file_uploader(
    "Sube uno o varios PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="PDFs con texto funcionan mejor. PDFs escaneados (imagen) requieren OCR (no incluido en este MVP).",
)

left, right = st.columns([1, 2])

# ---- Left: Ingesta
with left:
    st.subheader("1) Ingesta")

    if uploaded:
        st.caption(f"Seleccionados: **{len(uploaded)}** archivo(s)")
    else:
        st.caption("Seleccionados: **0** archivo(s)")

    if st.button("📥 Ingestar PDFs", disabled=not uploaded, key="btn_ingest", use_container_width=True):
        try:
            saved_paths = []
            for f in uploaded:
                out_path = DATA_DIR / f.name
                out_path.write_bytes(f.getvalue())
                saved_paths.append(out_path)

            stats = ingest_pdfs(saved_paths)
            st.session_state["last_ingest"] = stats

            # Si veníamos de un "reset" marcado, ya no aplica necesariamente
            clear_restart_required()

            st.success(
                f"Listo: {stats['files']} PDF(s), {stats['pages']} páginas, {stats['chunks']} chunks indexados."
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error ingestando: {e}")

    st.divider()

    st.subheader("PDFs guardados localmente")
    pdfs_local = list_pdfs_in_data_dir()
    if not pdfs_local:
        st.caption("No hay PDFs en `data/pdfs`.")
    else:
        for p in pdfs_local[:20]:
            st.write(f"• {p.name}")
        if len(pdfs_local) > 20:
            st.caption(f"… y {len(pdfs_local) - 20} más.")

# ---- Right: Preguntas
with right:
    st.subheader("2) Preguntar (RAG)")

    question = st.text_input(
        "Pregunta",
        placeholder="Ej: ¿Qué dice el documento sobre X?",
        key="question_input",
    )

    mode = st.radio(
        "Modo respuesta",
        ["Normal", "En vivo (stream)"],
        horizontal=True,
        key="mode_radio",
    )

    chunks, _files = get_index_stats()
    has_index = isinstance(chunks, int) and chunks > 0
    has_any_pdf = len(list_pdfs_in_data_dir()) > 0

    # Permitimos preguntar si hay índice o al menos hay PDFs (aunque el PDF sea escaneado)
    can_ask = has_index or has_any_pdf

    if not can_ask:
        st.caption("Primero sube al menos un PDF para habilitar preguntas.")
    elif not has_index:
        st.warning(
            "Hay PDFs, pero el índice está en 0 chunks. "
            "Esto suele pasar cuando el PDF no tiene texto extraíble (escaneado/imagen). "
            "Este MVP no hace OCR."
        )
    else:
        st.caption(f"Índice listo: **{chunks}** chunks.")

    # Motivo de deshabilitado (para que sea obvio)
    reason = None
    if not question or not question.strip():
        reason = "Escribe una pregunta para habilitar el botón."
    elif not can_ask:
        reason = "No hay PDFs ni índice disponible."

    disabled_btn = reason is not None
    if disabled_btn:
        st.info(reason)

    ask_clicked = st.button(
        "🤖 Preguntar",
        disabled=disabled_btn,
        key="ask_button",
        use_container_width=False,
    )

    if ask_clicked:
        try:
            st.write("**Respuesta:**")

            if mode == "En vivo (stream)":
                gen, sources = stream_answer(question.strip())
                st.write_stream(gen)
            else:
                out, sources = answer(question.strip())
                st.write(out)

            st.write("**Fuentes usadas:**")
            seen = set()
            for s in sources:
                key_ = (s.get("filename"), s.get("page"))
                if key_ in seen:
                    continue
                seen.add(key_)

                ocr_tag = " (OCR)" if s.get("ocr") else ""
                score = s.get("score")
                score_txt = f" | score={score:.4f}" if isinstance(score, (int, float)) else ""

                if s.get("page"):
                    st.write(f"- {s['filename']} (p. {s['page']}){ocr_tag}{score_txt}")
                else:
                    st.write(f"- {s['filename']}{ocr_tag}{score_txt}")

        except Exception as e:
            st.error(f"Error: {e}")

# Footer: última ingesta
if st.session_state.get("last_ingest"):
    stats = st.session_state["last_ingest"]
    st.caption(
        f"Última ingesta → PDFs: {stats.get('files')}, páginas: {stats.get('pages')}, chunks: {stats.get('chunks')} "
        f"({now_str()})"
    )