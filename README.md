# AI Portfolio MVP (Local Starter)

A tiny FastAPI service that:
- accepts a PDF/TXT upload,
- extracts text (first 5 pages for PDFs),
- redacts basic PII,
- calls an LLM with a mini rubric,
- returns JSON scores for {argumentation, writing, creativity}.

## 0) Prereqs
- Python 3.10+
- An OpenAI API key

## 1) Setup
```bash
git clone https://github.com/GabrielSwai/College-Admissions-AI-Copilot.git ai-portfolio-mvp
cd College-Admissions-AI-Copilot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY
```

## 2) Run locally
```bash
uvicorn app:app --reload --port 8000
```
Check:
```
GET http://localhost:8000/health
```

## 3) Try it (curl)
```bash
echo "This essay presents a clear argument about climate policy..." > sample.txt
curl -s -X POST http://localhost:8000/score   -F "title=Sample Essay"   -F "file=@sample.txt" | jq
```

Example response:
```json
{
  "title": "Sample Essay",
  "filename": "sample.txt",
  "scores": { "argumentation": 2, "writing": 2, "creativity": 1 },
  "tokens_estimate": 11,
  "model_version": "gpt-4o-mini"
}
```

## 4) What to add next
- Evidence quotes (JSON with `"quote"` + `"why"`).
- PDF export (WeasyPrint/ReportLab).
- Postgres storage (SQLModel/SQLAlchemy) + auth.
- File parsing for DOCX; OCR for image-PDFs (Tesseract).
- Queue for big jobs (RQ/Celery).

## 5) Deploy later (quick notes)
- **Render/Fly.io/Heroku:** add a `Procfile` like `web: uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Set env vars (OPENAI_API_KEY, MODEL_NAME).
- Put behind an auth proxy before exposing publicly.

## 6) Safety basics
- Keep PII redaction on server-side before LLM calls.
- Log model + rubric version in responses (add to DB later).
- Add a consent checkbox in your frontend before upload.

---

👤 Built & maintained by [Gabriel Swai](https://gabrielswai.com).