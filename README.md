# DocuDesk AI (MVP)

DocuDesk AI es un MVP local para **preguntar sobre tus PDFs**: subes documentos, se **indexan en Chroma** (embeddings) y haces preguntas desde una **UI mínima en Streamlit**, usando un **modelo en Ollama**.

> Alcance MVP: RAG básico (chunking + embeddings + vector search + respuesta). No pretende ser perfecto; pretende funcionar.

## Features (MVP)
- 📄 Carga de PDFs
- 🧠 Indexación: chunking + embeddings
- 🗂️ Vector DB local: Chroma
- 💬 Chat mínimo para Q&A sobre los PDFs cargados
- 🖥️ 100% local (Ollama)

## Requisitos
- Python **3.10+**
- **Ollama** instalado
- Un modelo descargado en Ollama (ej: `llama3.1`)

## Setup & Run
```bash
# 1) Entorno
python -m venv .venv
source .venv/bin/activate

# 2) Dependencias
pip install -r requirements.txt

# 3) Modelo local (ejemplo)
ollama pull llama3.1

# 4) Ejecutar app
streamlit run app.py