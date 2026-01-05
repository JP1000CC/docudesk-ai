# DocuDesk AI (MVP)

MVP: subes PDFs, se indexan (embeddings + Chroma) y puedes preguntar con un chat mínimo en Streamlit.

## Requisitos
- Python 3.10+
- Ollama instalado y un modelo descargado (ej: llama3.1)

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1
streamlit run app.py
