"""
RAG Agent - Knowledge Transfer (KT) via PDF Parsing
=====================================================
Upload a PDF (e.g. a KT/handover document) and ask questions about it.
The agent chunks the PDF, embeds the chunks with Gemini embeddings, stores
them in an in-memory FAISS index, and answers questions using
retrieval-augmented generation.

Run locally:
    export GOOGLE_API_KEY="your-key-here"
    uvicorn app:app --host 0.0.0.0 --port 8000

Deploy on Render:
    Start command -> uvicorn app:app --host 0.0.0.0 --port $PORT
    Set env var GOOGLE_API_KEY in the Render dashboard.

Limitation: the vector store is kept in server memory (a Python dict), so it
resets on every restart/redeploy and is not shared across multiple server
instances. That's fine for a demo/single-instance deployment; a production
system would use a persistent vector database instead.
"""

import io
import logging
import os
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-kt-agent")

# --------------------------------------------------------------------------
# Constants / configuration (no hardcoded secrets)
# --------------------------------------------------------------------------
DEFAULT_CHAT_MODEL = "gemini-flash-latest"
models/gemini-embedding-2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB


class AppConfig:
    GOOGLE_API_KEY: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    CHAT_MODEL: str = os.environ.get("GEMINI_MODEL", DEFAULT_CHAT_MODEL)
    EMBEDDING_MODEL: str = os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _require_api_key() -> str:
    if not AppConfig.GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not set on the server. Set it as an "
            "environment variable before calling this endpoint.",
        )
    return AppConfig.GOOGLE_API_KEY


# --------------------------------------------------------------------------
# In-memory document store: doc_id -> FAISS vectorstore
# --------------------------------------------------------------------------
_VECTOR_STORES: Dict[str, "FAISS"] = {}
_DOC_META: Dict[str, dict] = {}


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=AppConfig.EMBEDDING_MODEL,
        google_api_key=_require_api_key(),
    )


def get_chat_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=AppConfig.CHAT_MODEL,
        google_api_key=_require_api_key(),
        temperature=0.2,
    )


def extract_text(response) -> str:
    """Normalise a LangChain AIMessage's `.content` into a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts).strip()
    return str(content)


def extract_pdf_pages(pdf_bytes: bytes) -> List[dict]:
    """Return a list of {"page": int, "text": str} for each non-empty page."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the uploaded PDF. "
            "It may be a scanned/image-only PDF.",
        )
    return pages


def build_vectorstore(pages: List[dict]):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"page": page["page"]}))

    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(
    title="RAG Knowledge Transfer Agent",
    description="Upload a PDF and ask questions about it (retrieval-augmented generation).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chat_model": AppConfig.CHAT_MODEL,
        "embedding_model": AppConfig.EMBEDDING_MODEL,
        "api_key_configured": bool(AppConfig.GOOGLE_API_KEY),
        "documents_in_memory": len(_VECTOR_STORES),
    }


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages_indexed: int
    chunks_indexed: int


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Ingest a PDF: extract text, chunk it, embed it, and store it in memory."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=422, detail="file must be a PDF.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit.")
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    pages = extract_pdf_pages(pdf_bytes)

    try:
        vectorstore = build_vectorstore(pages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build vector store")
        raise HTTPException(status_code=502, detail=f"Failed to index PDF: {exc}") from exc

    doc_id = str(uuid.uuid4())
    _VECTOR_STORES[doc_id] = vectorstore
    chunk_count = vectorstore.index.ntotal
    _DOC_META[doc_id] = {
        "filename": file.filename,
        "pages_indexed": len(pages),
        "chunks_indexed": chunk_count,
    }

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename or "unknown.pdf",
        pages_indexed=len(pages),
        chunks_indexed=chunk_count,
    )


class AskRequest(BaseModel):
    doc_id: str
    question: str


class SourceChunk(BaseModel):
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """Answer a question about a previously uploaded document."""
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")

    vectorstore = _VECTOR_STORES.get(payload.doc_id)
    if vectorstore is None:
        raise HTTPException(
            status_code=404,
            detail="doc_id not found. Upload a PDF via /upload first "
            "(note: the in-memory index resets on server restart).",
        )

    retrieved_docs = vectorstore.similarity_search(payload.question, k=TOP_K)
    if not retrieved_docs:
        return AskResponse(
            answer="I couldn't find anything relevant to that question in the document.",
            sources=[],
        )

    context_blocks = []
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        context_blocks.append(f"[page {page}]\n{doc.page_content}")
        sources.append(SourceChunk(page=page, snippet=doc.page_content[:300]))

    context = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "You are a knowledge-transfer assistant. Answer the QUESTION using "
        "ONLY the CONTEXT excerpts below, which come from an internal "
        "document. If the answer isn't in the context, say so explicitly - "
        "do not make anything up. Cite the page number(s) you used.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{payload.question}"
    )

    llm = get_chat_model()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    answer = extract_text(response)
    return AskResponse(answer=answer, sources=sources)


@app.delete("/doc/{doc_id}")
def delete_doc(doc_id: str):
    """Remove a document's index from memory."""
    if doc_id not in _VECTOR_STORES:
        raise HTTPException(status_code=404, detail="doc_id not found.")
    del _VECTOR_STORES[doc_id]
    _DOC_META.pop(doc_id, None)
    return {"status": "deleted", "doc_id": doc_id}


# --------------------------------------------------------------------------
# Browser playground
#
# IMPORTANT:
# LangServe's built-in playground accepts JSON inputs and cannot directly
# upload a PDF with a file picker. Therefore /agent/playground below is a
# small custom browser UI that uploads the PDF first, receives the generated
# doc_id, and then asks questions against that document.
#
# The native LangServe playground is still available at:
# /agent/api/playground/
# --------------------------------------------------------------------------

from langchain_core.runnables import RunnableLambda  # noqa: E402
from langserve import add_routes  # noqa: E402


class AskPlaygroundInput(BaseModel):
    doc_id: str
    question: str


def _playground_ask(payload) -> AskResponse:
    if isinstance(payload, dict):
        payload = AskPlaygroundInput(**payload)
    return ask_question(
        AskRequest(doc_id=payload.doc_id, question=payload.question)
    )


rag_qa_chain = RunnableLambda(_playground_ask).with_types(
    input_type=AskPlaygroundInput,
    output_type=AskResponse,
)

# Keep the native LangServe API/playground available without taking over
# /agent/playground.
add_routes(app, rag_qa_chain, path="/agent/api")


PLAYGROUND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Transfer Agent</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background: #f5f7fb;
            color: #172033;
        }
        .wrap {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px 50px;
        }
        .hero {
            background: white;
            border-radius: 18px;
            padding: 28px;
            box-shadow: 0 8px 30px rgba(20, 35, 70, .08);
            margin-bottom: 20px;
        }
        h1 { margin: 0 0 8px; font-size: 30px; }
        .subtitle { color: #667085; margin: 0; }
        .card {
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-top: 18px;
            box-shadow: 0 8px 30px rgba(20, 35, 70, .07);
        }
        label {
            display: block;
            font-weight: 700;
            margin-bottom: 8px;
        }
        input[type=file], textarea {
            width: 100%;
            border: 1px solid #d0d5dd;
            border-radius: 12px;
            padding: 13px;
            font-size: 15px;
            background: #fff;
        }
        textarea {
            min-height: 110px;
            resize: vertical;
        }
        button {
            border: 0;
            border-radius: 12px;
            padding: 13px 20px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 12px;
        }
        .upload { background: #172033; color: white; }
        .ask { background: #2563eb; color: white; }
        button:disabled { opacity: .55; cursor: not-allowed; }
        .status {
            margin-top: 14px;
            padding: 12px;
            border-radius: 10px;
            background: #f2f4f7;
            white-space: pre-wrap;
        }
        .success { background: #ecfdf3; color: #027a48; }
        .error { background: #fef3f2; color: #b42318; }
        .answer {
            margin-top: 20px;
            padding: 18px;
            border-radius: 12px;
            background: #f8fafc;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .sources {
            margin-top: 15px;
            font-size: 13px;
            color: #667085;
        }
        .source {
            margin-top: 8px;
            padding: 10px;
            background: #f2f4f7;
            border-radius: 8px;
        }
        .docid {
            font-family: monospace;
            word-break: break-all;
        }
        .hint {
            font-size: 13px;
            color: #667085;
            margin-top: 8px;
        }
    </style>
</head>
<body>
<div class="wrap">
    <div class="hero">
        <h1>📚 Knowledge Transfer Agent</h1>
        <p class="subtitle">
            Upload a PDF, let the agent index it, then ask questions using
            retrieval-augmented generation.
        </p>
    </div>

    <div class="card">
        <label for="pdf">1. Upload Knowledge Transfer PDF</label>
        <input id="pdf" type="file" accept=".pdf,application/pdf">
        <button id="uploadBtn" class="upload" onclick="uploadPDF()">
            Upload & Index PDF
        </button>
        <div id="uploadStatus" class="status">
            No document uploaded yet.
        </div>
    </div>

    <div class="card">
        <label for="question">2. Ask a Question</label>
        <textarea id="question"
            placeholder="Example: What are the seven steps in the PDF processing pipeline?"></textarea>
        <button id="askBtn" class="ask" onclick="askQuestion()" disabled>
            Ask Agent
        </button>
        <div class="hint">
            Upload a PDF first. The document ID is generated automatically.
        </div>

        <div id="answer"></div>
        <div id="sources"></div>
    </div>
</div>

<script>
let currentDocId = null;

function setStatus(message, type="") {
    const el = document.getElementById("uploadStatus");
    el.className = "status " + type;
    el.textContent = message;
}

async function uploadPDF() {
    const input = document.getElementById("pdf");
    const btn = document.getElementById("uploadBtn");

    if (!input.files.length) {
        setStatus("Please choose a PDF first.", "error");
        return;
    }

    const file = input.files[0];

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        setStatus("Only PDF files are supported.", "error");
        return;
    }

    btn.disabled = true;
    setStatus("Uploading and indexing the PDF... This may take a little while.");

    try {
        const form = new FormData();
        form.append("file", file);

        const response = await fetch("/upload", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        currentDocId = data.doc_id;

        setStatus(
            "PDF indexed successfully.\n\n" +
            "File: " + data.filename + "\n" +
            "Pages: " + data.pages_indexed + "\n" +
            "Chunks: " + data.chunks_indexed + "\n" +
            "Document ID: " + data.doc_id,
            "success"
        );

        document.getElementById("askBtn").disabled = false;
    } catch (err) {
        currentDocId = null;
        document.getElementById("askBtn").disabled = true;
        setStatus("Upload error:\n" + err.message, "error");
    } finally {
        btn.disabled = false;
    }
}

async function askQuestion() {
    const question = document.getElementById("question").value.trim();
    const btn = document.getElementById("askBtn");
    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");

    if (!currentDocId) {
        answer.innerHTML = '<div class="answer">Please upload a PDF first.</div>';
        return;
    }

    if (!question) {
        answer.innerHTML = '<div class="answer">Please enter a question.</div>';
        return;
    }

    btn.disabled = true;
    answer.innerHTML = '<div class="answer">Thinking...</div>';
    sources.innerHTML = "";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                doc_id: currentDocId,
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Question failed.");
        }

        answer.innerHTML =
            '<div class="answer"><strong>Answer</strong><br><br>' +
            escapeHtml(data.answer) +
            '</div>';

        if (data.sources && data.sources.length) {
            let html = '<div class="sources"><strong>Retrieved sources</strong>';
            for (const source of data.sources) {
                html +=
                    '<div class="source"><strong>Page ' +
                    escapeHtml(String(source.page)) +
                    ':</strong> ' +
                    escapeHtml(source.snippet) +
                    '</div>';
            }
            html += '</div>';
            sources.innerHTML = html;
        }
    } catch (err) {
        answer.innerHTML =
            '<div class="answer error">' +
            escapeHtml("Error: " + err.message) +
            '</div>';
    } finally {
        btn.disabled = false;
    }
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
</script>
</body>
</html>
"""


@app.get("/agent/playground", response_class=HTMLResponse, include_in_schema=False)
@app.get("/agent/playground/", response_class=HTMLResponse, include_in_schema=False)
def custom_playground():
    return HTMLResponse(content=PLAYGROUND_HTML)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
