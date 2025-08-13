# main.py — FastAPI backend (LLM-only replies, personalized)

import os, re, textwrap, logging, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # load backend/.env

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from zoneinfo import ZoneInfo

log = logging.getLogger("uvicorn.error")

# ---------- Config ----------
LABELS = ["Urgent", "Normal", "Low"]
MODEL_DIR = os.getenv("MODEL_DIR", "models/email-priority-clf")
API_KEY   = os.getenv("API_KEY")  # optional

# One provider only (set in .env)
LLM_PROVIDER   = (os.getenv("LLM_PROVIDER") or "").lower()  # openai|anthropic|gemini
OPENAI_MODEL   = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")
ANTHROPIC_MODEL= os.getenv("ANTHROPIC_MODEL","claude-sonnet-4-20250514")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "America/Toronto")

# ---------- Helpers ----------
LOW_HINTS    = re.compile(r"\b(no rush|nice to have|feature request|suggestion|when possible|whenever possible)\b", re.I)
URGENT_HINTS = re.compile(r"\b(asap|today|within (an|[0-9]+)\s*hour|blocked|blocker|cannot (login|access)|down|outage|breach|security|customers? (blocked|impacted|affected))\b", re.I)

def apply_urgency_overrides(text: str, model_label: str) -> str:
    t = text.lower()
    if re.search(URGENT_HINTS, t): return "Urgent"
    if re.search(LOW_HINTS, t):    return "Low"
    return model_label

def _gen_id(prefix: str) -> str:
    now = datetime.now(ZoneInfo(DEFAULT_TZ))
    rand = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
    return f"{prefix}-{now.strftime('%Y%m%d-%H%M')}-{rand}"

def next_update_str(minutes: int = 120) -> str:
    dt = datetime.now(ZoneInfo(DEFAULT_TZ)) + timedelta(minutes=int(minutes or 120))
    return dt.strftime("%b %d, %Y %I:%M %p %Z")

# ---------- LLM ----------
def _reply_prompt(subject: str, body: str, urgency: str, tone: str,
                  intent_summary: str, from_name: str, from_email: str,
                  ticket_id: str, next_update: str) -> str:
    display_name = from_name or (from_email.split("@")[0] if from_email else "")
    greeting = f"Hi {display_name}," if display_name else "Hi there,"

    return textwrap.dedent(f"""
    You are an on-call support engineer. Write a tailored reply to THIS email.
    Absolutely follow these rules:
    - Start with this exact greeting: "{greeting}"
    - Do NOT use placeholders like [Recipient], [Your Name], etc.
    - Reference this ticket ID exactly once in the first paragraph: {ticket_id}
    - Include a specific next-update time: {next_update}
    - Keep tone: {tone} (formal|neutral|friendly)
    - Match urgency: {urgency} (Urgent|Normal|Low). No incident/mitigation language for Low.
    - Be concise, concrete, and avoid boilerplate. Ask only for information that unblocks progress.
    - Do not invent facts; only use info present in the email/context below.

    Email:
    Subject: {subject}
    Body:
    {body}

    Context summary (for you, do not just echo): {intent_summary}

    Output: A ready-to-send plain-text email reply ending with "Thanks,\nSupport Team".
    """)

def llm_custom_reply(subject: str, body: str, urgency: str, tone: str,
                     intent_summary: str, from_name: str, from_email: str,
                     ticket_id: str, next_update: str) -> Optional[str]:
    prompt = _reply_prompt(subject, body, urgency, tone, intent_summary, from_name, from_email, ticket_id, next_update)
    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI()  # OPENAI_API_KEY
            try:
                r = client.responses.create(model=OPENAI_MODEL, input=prompt, temperature=0.5)
                txt = getattr(r, "output_text", None)
                if txt: return txt
            except Exception:
                chat = client.chat.completions.create(
                    model=OPENAI_MODEL, temperature=0.5,
                    messages=[{"role":"user","content":prompt}]
                )
                return chat.choices[0].message.content

        if LLM_PROVIDER == "anthropic":
            from anthropic import Anthropic
            client = Anthropic()  # ANTHROPIC_API_KEY
            msg = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=800, temperature=0.5,
                system="You write precise, empathetic support emails. No boilerplate. No hallucinations.",
                messages=[{"role":"user","content":prompt}],
            )
            return "".join([c.text for c in msg.content if getattr(c, "type", "") == "text"])

        if LLM_PROVIDER == "gemini":
            from google import genai
            client = genai.Client()  # GEMINI_API_KEY
            r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return getattr(r, "text", None)

        log.error("[llm] No provider configured (LLM_PROVIDER).")
    except Exception as e:
        log.error("[llm] error: %s", e)
    return None

# ---------- HF models (classifier + summarizer) ----------
clf_tokenizer = None
clf_model = None
zero_shot = None
summarizer = None

def load_models():
    global clf_tokenizer, clf_model, zero_shot, summarizer
    if os.path.isdir(MODEL_DIR):
        clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        clf_model.eval()
        log.info("Loaded classifier from %s", MODEL_DIR)
    else:
        zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        log.info("Using zero-shot classifier")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def classify(text: str):
    if clf_model is not None:
        inputs = clf_tokenizer(text, truncation=True, max_length=384, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(clf_model(**inputs).logits.squeeze(0), dim=-1).tolist()
        scores: Dict[str, float] = {lab: float(p) for lab, p in zip(LABELS, probs)}
        return max(scores, key=scores.get), scores
    else:
        out = zero_shot(text, candidate_labels=LABELS, multi_label=False)
        scores: Dict[str, float] = {lab: 0.0 for lab in LABELS}
        for lab, sc in zip(out["labels"], out["scores"]):
            scores[lab] = float(sc)
        return max(scores, key=scores.get), scores

def make_intent_summary(text: str) -> str:
    try:
        tok_len = len(summarizer.tokenizer.encode(text, truncation=True))
        if tok_len <= 30:
            mx = max(12, int(0.8 * tok_len)); mn = max(6, min(mx - 2, int(0.5 * mx)))
        else:
            mx = min(120, max(24, int(0.6 * tok_len))); mn = max(12, min(mx - 10, int(0.35 * mx)))
        return summarizer(text, max_length=mx, min_length=mn, do_sample=False)[0]["summary_text"].strip()
    except Exception as e:
        log.error("[summarize] error: %s", e)
        return text[:600].strip()

# ---------- FastAPI ----------
app = FastAPI(title="Email Priority Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Payload(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    tone: Optional[str] = "neutral"
    from_name: Optional[str] = ""
    from_email: Optional[str] = ""

@app.on_event("startup")
def _startup():
    load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/classify")
def api_classify(p: Payload, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = ((p.subject or "") + "\n\n" + (p.body or "")).strip()
    if not text:
        return {"error": "Empty subject/body"}

    # 1) urgency
    label, scores = classify(text)
    label = apply_urgency_overrides(text, label)

    # 2) context summary
    intent = make_intent_summary(text)

    # 3) generate ticket ID + next-update string based on urgency
    prefix = "INC" if label == "Urgent" else ("CASE" if label == "Normal" else "REQ")
    ticket_id = _gen_id(prefix)
    next_update = next_update_str(120 if label in ("Urgent","Normal") else 1440)  # 2h vs 24h

    # 4) LLM must produce the reply (no template fallback)
    reply = llm_custom_reply(p.subject or "", p.body or "", label, p.tone or "neutral",
                             intent, p.from_name or "", p.from_email or "",
                             ticket_id, next_update)
    if not reply:
        return {"error": "LLM not configured or failed; check provider, key, and model on server."}

    return {
        "urgency": label,
        "scores": scores,
        "intent": intent,
        "reply": reply,
        "reply_source": "llm",
        "ticket_id": ticket_id,
        "next_update": next_update
    }