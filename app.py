import os, io, json, re
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

# ---------- env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env")

# ---------- llm client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- fastapi ----------
app = FastAPI(title="AI Portfolio MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- simple schema ----------
class Scores(BaseModel):
    argumentation: int
    writing: int
    creativity: int

RUBRIC = """RUBRIC (0–3):
argumentation: 0=none, 1=basic, 2=clear, 3=advanced
writing: 0=errors, 1=some errors, 2=clear, 3=polished
creativity: 0=generic, 1=some originality, 2=unique voice, 3=novel insight
"""

# ---------- utilities ----------
def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 5) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages[:max_pages]:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def extract_text_generic(file: UploadFile) -> str:
    name = file.filename.lower()
    if name.endswith(".pdf"):
        # We already read once in the route; pass through again here if needed
        raise RuntimeError("PDF path should call extract_text_from_pdf directly.")
    # naive text read for .txt; DOCX/others: add parsers later
    return file.file.read().decode("utf-8", "ignore")

def redact_basic_pii(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "〈REDACTED_EMAIL〉", text)
    # very crude “Firstname Lastname” pattern; replace or enhance later
    text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "〈REDACTED_NAME〉", text)
    return text

def score_text_llm(text: str) -> Dict:
    system_msg = "You are an assistant that scores student work. Output JSON only."
    user_msg = f"""Text:
\"\"\"{text[:5000]}\"\"\"

{RUBRIC}
Task:
Return JSON: {{"argumentation":0-3, "writing":0-3, "creativity":0-3}}. No extra text."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}]
    )
    raw = resp.choices[0].message.content.strip()

    # Try parse → quick repair if needed
    try:
        data = json.loads(raw)
    except Exception:
        fixer = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role":"system","content":"Fix invalid JSON to match: keys argumentation, writing, creativity as integers 0..3. Output JSON only."},
                {"role":"user","content":raw}
            ]
        )
        data = json.loads(fixer.choices[0].message.content)

    # Validate with Pydantic (enforces 3 ints)
    Scores(**data)
    return data

# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/score")
async def score(file: UploadFile = File(...), title: str = Form("untitled")):
    raw = await file.read()

    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(raw, max_pages=5)
    else:
        # fallback for .txt; add DOCX parser later
        text = raw.decode("utf-8", "ignore")

    text = redact_basic_pii(text)
    scores = score_text_llm(text)

    return {
        "title": title,
        "filename": file.filename,
        "scores": scores,
        "tokens_estimate": len(text.split()),
        "model_version": MODEL_NAME
    }