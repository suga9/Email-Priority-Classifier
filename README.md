# Email Priority Classifier + Reply Generator (Option #2)

A lightweight, cloud-hostable web app that:
- Classifies incoming emails into **Urgent / Normal / Low**
- Generates an **intent summary** of the email
- **Drafts a reply** based on urgency + content (template-based by default; optional hook for an external LLM)
- Lets you **edit** the draft and **copy/download** it
- Supports **batch CSV processing** and export


## Run Locally
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## CSV Format
Upload a CSV with columns:
- `subject` (string)
- `body` (string)
- (optional) `from`, `to`, `timestamp`

See `sample_data/sample_emails.csv`.

