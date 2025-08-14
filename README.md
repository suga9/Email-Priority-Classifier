# Email Priority Classifier + Reply Generator (Option #2)

## Email Priority Assistant — Project Documentation

## 1) Overview
Email Priority Assistant is an end-to-end system that:

Classifies incoming emails by urgency (Urgent / Normal / Low).

Generates a bespoke reply using an LLM (no rigid templates).

Exposes a FastAPI backend that powers:

- a Streamlit demo UI, and

- a Gmail Workspace Add-on.

The assistant combines a lightweight classifier (Hugging Face pipeline) with a short intent summary and then prompts an LLM to draft a concise, context-aware response. Greetings are personalized from sender name/email; replies include an auto ticket ID and a next-update time based on urgency.

## 2) Architecture

<img width="633" height="221" alt="Screenshot 2025-08-14 at 2 45 48 PM" src="https://github.com/user-attachments/assets/00ff6e5c-6eaf-4f48-9bab-4c245e9c752f" />


Key components

FastAPI backend (backend/main.py): LLM-only replies; optional header auth (X-API-Key).

Streamlit app (app.py): Simple UI calling the backend.

Gmail Add-on (Code.gs, appsscript.json): Reads the selected message and calls the backend.

## 3) Features
LLM-only replies (supports OpenAI / Anthropic / Gemini — choose one in env).

Lightweight urgency classifier + intent summary to ground the LLM.

Personalized greeting, ticket ID (INC/CASE/REQ), next-update time.

CORS open for demo; can be restricted for production.

CSV sample for bulk testing in Streamlit.


## 4) System Requirements
Python 3.11+

pip/venv

Optional: ngrok (for Gmail Add-on testing against local backend)

One LLM API key (OpenAI shown in examples)

## 5) Repository Layout Demo.


email-priority-classifier/
│
├─ app.py                     # Streamlit UI (calls backend API)
├─ .streamlit/
│   └─ secrets.toml           # (optional) Streamlit secrets
│
├─ backend/
│   ├─ main.py                # FastAPI (LLM-only replies)
│   ├─ .env                   # Backend env vars (NOT committed)
│   └─ requirements.txt       # (optional) backend-only deps
│
├─ email_demo.csv             # (optional) sample data for demo
├─ 
└─ .gitignore

<img width="465" height="876" alt="Screenshot 2025-08-14 at 2 41 19 PM" src="https://github.com/user-attachments/assets/d71cac0c-fd32-4c2a-93a4-d4c217cb57f7" />

## 6) Setup & Local Run
   
6.1 Create a virtual environment6) Setup & Local Run

python -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate             # Windows (PowerShell)
python -m pip install --upgrade pip

6.2 Install dependencies

pip install -U fastapi uvicorn streamlit requests python-dotenv \
  transformers torch accelerate \
  openai anthropic google-genai


6.3 Configure backend environment

Create backend/.env (use one provider; OpenAI example):
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini

# optional
DEFAULT_TZ=America/Toronto
MODEL_DIR=models/email-priority-clf

API_KEY=change-me      




6.4 Start the backend (Terminal A)

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Sanity check:
curl -s http://127.0.0.1:8000/health
# -> {"status":"ok"}

curl -s -X POST http://127.0.0.1:8000/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"subject":"Feature request: PDF export (no rush)","body":"Nice to have; whenever possible.","tone":"friendly"}' \
  | python -m json.tool
# Expect: urgency + "reply_source":"llm" and a bespoke reply

6.5 Run the Streamlit UI (Terminal B)

cd /path/to/email-priority-classifier   # project root
export BACKEND_URL="http://127.0.0.1:8000"
streamlit run app.py

Open http://localhost:8501.

If you prefer **secrets:** create .streamlit/secrets.toml with
BACKEND_URL = "http://127.0.0.1:8000".

## 7) Gmail Add-on (Local)
Apps Script cannot call http://localhost, so use an HTTPS tunnel.

7.1 Start ngrok (Terminal C)
ngrok http 8000

Copy the https URL (e.g., https://abcd1234.ngrok-free.app).

7.2 Apps Script files (create in a new Apps Script project)

{
  "timeZone": "America/Toronto",
  "exceptionLogging": "STACKDRIVER",
  "runtimeVersion": "V8",
  "oauthScopes": [
    "https://www.googleapis.com/auth/gmail.addons.execute",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/script.external_request"
  ],
  "addOns": {
    "common": {
      "name": "Email Priority Assistant",
      "logoUrl": "https://ssl.gstatic.com/docs/doclist/images/mediatype/icon_1_document_x16.png",
      "layoutProperties": { "primaryColor": "#3f51b5" }
    },
    "gmail": {
      "contextualTriggers": [
        { "unconditional": {}, "onTriggerFunction": "buildAddOn" }
      ]
    }
  }
}

Code.gs (replace BACKEND_URL with your ngrok base URL)

const BACKEND_URL = 'https://<YOUR-NGROK>.ngrok-free.app';

function buildAddOn(e) {
  const messageId = e.gmail && e.gmail.messageId;
  let subject = '', body = '', fromName = '', fromEmail = '';
  if (messageId) {
    const msg = GmailApp.getMessageById(messageId);
    subject = msg.getSubject();
    body = msg.getPlainBody ? msg.getPlainBody() : msg.getBody().replace(/<[^>]+>/g,' ');
    const m = /^(.*?)\s*<([^>]+)>$/.exec(msg.getFrom() || '');
    fromName  = m ? (m[1] || '') : '';
    fromEmail = m ? (m[2] || '') : (msg.getFrom() || '');
  }

  const action = CardService.newAction().setFunctionName('onClassify_')
    .setParameters({ subject, body, fromName, fromEmail });

  const section = CardService.newCardSection()
    .addWidget(
      CardService.newTextButton()
        .setText('Classify & Draft')
        .setOnClickAction(action)
        .setTextButtonStyle(CardService.TextButtonStyle.FILLED)
    );

  return [CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle('Email Priority Assistant'))
    .addSection(section)
    .build()];
}

function onClassify_(e) {
  const p = e.parameters || {};
  const resp = UrlFetchApp.fetch(BACKEND_URL + '/api/classify', {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify({
      subject: p.subject || '',
      body: p.body || '',
      tone: 'neutral',
      from_name: p.fromName || '',
      from_email: p.fromEmail || ''
    }),
    muteHttpExceptions: true
  });

  const data = JSON.parse(resp.getContentText());
  const section = CardService.newCardSection()
    .addWidget(CardService.newKeyValue().setTopLabel('Urgency').setContent(data.urgency || '—'))
    .addWidget(CardService.newTextParagraph().setText('<b>Intent Summary</b><br/>' + (data.intent || '—')))
    .addWidget(CardService.newTextParagraph().setText('<b>Reply Draft</b><br/>' + (data.reply || '').replace(/\n/g, '<br/>')));

  return CardService.newActionResponseBuilder()
    .setNavigation(CardService.newNavigation().pushCard(
      CardService.newCardBuilder()
        .setHeader(CardService.newCardHeader().setTitle('Results'))
        .addSection(section)
        .build()))
    .build();
}

Deploy: Apps Script → Deploy → Test deployments → Install.
Open an email in Gmail → Add-on → Classify & Draft.









