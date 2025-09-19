import os, io, json, re
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from openai import OpenAI

class TextPayload(BaseModel):
    title: str = Field(default="untitled")
    text: str = Field(min_length=1, description="Essay text to score")

class Category(BaseModel):
    name: str = Field(min_length=2, max_length=40, description="JSON key to return (e.g., 'argumentation' or 'intellectual_curiosity')")
    description: str = Field(default="", max_length=200, description="What this category means")

class FlexPayload(BaseModel):
    title: str = "untitled"
    text: str = Field(min_length=1)
    categories: List[Category] = Field(min_items=1)
    quotes: bool = False  # set true if you want a short evidence quote per category

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

@app.post("/score-text")
def score_text(payload: TextPayload = Body(...)):
    text = redact_basic_pii(payload.text)
    scores = score_text_llm(text)
    return {
        "title": payload.title,
        "scores": scores,
        "tokens_estimate": len(text.split()),
        "model_version": MODEL_NAME
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Text Scorer (Flexible Rubric)</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Inter,Arial,sans-serif;margin:24px;max-width:960px}
h1{margin:0 0 12px}
label{font-weight:600}
textarea,input,button{font-size:16px}
textarea{width:100%;min-height:220px;padding:10px;margin:6px 0 14px}
.row{display:flex;gap:12px;align-items:center;margin:8px 0;flex-wrap:wrap}
input[type=text]{padding:8px;width:320px}
button{padding:10px 14px;cursor:pointer}
pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}
.small{font-size:12px;color:#666}
</style>
</head>
<body>
<h1>Text Scorer — Flexible Categories</h1>

<div class="row">
  <label for="title">Title</label>
  <input id="title" type="text" placeholder="untitled" />
  <label><input id="quotes" type="checkbox"/> include evidence quotes</label>
</div>

<label for="cats">Categories (one per line, format: <code>name: description</code>)</label>
<textarea id="cats">argumentation: Clear claim and support with evidence
writing: Clarity, organization, grammar, style
creativity: Originality, novel insight, unique voice</textarea>

<label for="essay">Essay text</label>
<textarea id="essay" placeholder="Paste or type your essay here..."></textarea>

<div class="row">
  <button id="sample">Insert sample</button>
  <button id="scoreBtn">Score with current categories</button>
  <span class="small">Sends JSON to <code>/score-text-flex</code> and shows the result.</span>
</div>

<h2>Result</h2>
<pre id="out">{}</pre>

<script>
const $ = (id) => document.getElementById(id);

$("sample").onclick = () => {
  $("title").value = "The Role of Failure in Innovation";
  $("essay").value =
`Innovation is often celebrated as the product of genius, but history shows it grows out of failure...
A culture that encourages learning from mistakes enables innovators to persist until the right solution emerges.`;
};

function parseCategories(text) {
  const lines = text.split(/\\r?\\n/).map(l => l.trim()).filter(Boolean);
  const cats = [];
  for (const line of lines) {
    const [nameRaw, ...rest] = line.split(":");
    const name = (nameRaw || "").trim().replace(/\\s+/g, "_").toLowerCase();
    const description = (rest.join(":") || "").trim();
    if (!name) continue;
    cats.push({ name, description });
  }
  return cats;
}

$("scoreBtn").onclick = async () => {
  const title = $("title").value || "untitled";
  const text  = $("essay").value.trim();
  const quotes = $("quotes").checked;
  const cats = parseCategories($("cats").value);

  if (!text) { $("out").textContent = "Please enter essay text."; return; }
  if (!cats.length) { $("out").textContent = "Please define at least one category."; return; }

  $("out").textContent = "Scoring...";
  try {
    const res = await fetch("/score-text-flex", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ title, text, categories: cats, quotes })
    });
    const json = await res.json();
    $("out").textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    $("out").textContent = "Error: " + (err?.message || err);
  }
};
</script>
</body>
</html>
"""

@app.post("/score-text-flex")
def score_text_flex(payload: FlexPayload = Body(...)):
    # Build a dynamic rubric from the categories the browser sends
    base_scale = "Use 0–3 where 0=insufficient, 1=emerging, 2=proficient, 3=advanced."
    cat_lines = "\n".join([f"- {c.name}: {c.description or 'Assess per the scale; stay on-topic.'}"
                           for c in payload.categories])

    sys = "You are an assistant that scores student work. Always output JSON only."
    wants_quotes = payload.quotes
    json_shape = "{ " + ", ".join(
        [f"\"{c.name}\": {{\"score\":0-3{', \"quote\":\"≤25 words\"' if wants_quotes else ''}}}" for c in payload.categories]
    ) + " }"

    user = f"""Text:
\"\"\"{redact_basic_pii(payload.text)[:6000]}\"\"\"

RUBRIC:
{base_scale}
Score these categories:
{cat_lines}

Task:
Return JSON exactly with these keys and this shape:
{json_shape}
No extra fields or prose."""

    resp = client.chat.completions.create(
        model=MODEL_NAME, temperature=0,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content.strip()

    # Basic repair if the model adds stray text
    import json
    try:
        data = json.loads(raw)
    except Exception:
        fixer = client.chat.completions.create(
            model=MODEL_NAME, temperature=0,
            messages=[
                {"role":"system","content":"Fix to valid JSON only. Keep the same keys and structure."},
                {"role":"user","content":raw}
            ]
        )
        data = json.loads(fixer.choices[0].message.content)

    # Minimal validation: ensure all categories exist with score 0..3 (and quote if requested)
    for c in payload.categories:
        if c.name not in data or not isinstance(data[c.name], dict):
            raise ValueError(f"Missing category '{c.name}' in response.")
        s = data[c.name].get("score", None)
        if not isinstance(s, int) or s not in (0,1,2,3):
            raise ValueError(f"Invalid score for '{c.name}': {s}")
        if wants_quotes:
            q = data[c.name].get("quote", "")
            if not isinstance(q, str) or len(q.split()) > 25:
                raise ValueError(f"Quote too long or missing for '{c.name}'.")

    return {
        "title": payload.title,
        "scores": data,
        "tokens_estimate": len(payload.text.split()),
        "model_version": MODEL_NAME
    }