# RAG Knowledge Transfer Agent (PDF Parsing)

Upload a PDF (e.g. a KT/handover doc) and ask questions about it. The agent
extracts text per page, chunks it, embeds the chunks with Gemini embeddings,
indexes them in an in-memory FAISS store, and answers questions using
retrieval-augmented generation with page-level citations.

## Architecture

```
Upload PDF -> extract text per page -> chunk -> embed (Gemini) -> FAISS index (in memory)
Ask question -> embed question -> retrieve top-k chunks -> LLM answers using only that context
```

## 1. Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env and add your real key
export $(grep -v '^#' .env | xargs)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Get a Gemini API key at https://aistudio.google.com/app/apikey

Test it:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/upload -F "file=@/path/to/handover.pdf"
# -> returns {"doc_id": "...", "pages_indexed": N, "chunks_indexed": M}

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<doc_id from above>", "question": "Who owns the deployment pipeline?"}'
```

LangServe also exposes a browser-based playground for the Q&A step at:
`http://localhost:8000/agent/playground/`
(upload the PDF via `/upload` or `/docs` first to get a `doc_id`, then paste
that `doc_id` + your question into the playground form - playground forms
can't handle file uploads directly).

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "RAG KT PDF agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3. Deploy on Render

1. New -> Web Service -> connect this GitHub repo.
2. Environment: Python 3.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `GOOGLE_API_KEY` in the Render dashboard.
   Optionally add `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` to override defaults.
6. Deploy. Health check: `https://<your-app>.onrender.com/health`

## Notes / limitations

- The FAISS index lives in server memory (a Python dict keyed by `doc_id`).
  It resets on every restart/redeploy and is not shared across multiple
  server instances/workers. Fine for a demo or single-worker deployment;
  swap in a persistent vector DB (e.g. Chroma with disk storage, Pinecone,
  pgvector) for production use.
- Answers are constrained to only use retrieved context and the model is
  instructed to say so explicitly if the answer isn't in the document,
  reducing hallucination risk.
- No API keys are hardcoded anywhere; everything reads from environment
  variables.
