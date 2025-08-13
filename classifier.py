# classifier.py (fine-tuned)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABELS = ["Urgent","Normal","Low"]
model_name = "models/email-priority-clf"
_tokenizer = AutoTokenizer.from_pretrained(model_name)
_model = AutoModelForSequenceClassification.from_pretrained(model_name)
_model.eval()

def classify_priority(text: str):
    inputs = _tokenizer(text, truncation=True, max_length=384, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1).tolist()
    scores = {lab: float(p) for lab, p in zip(LABELS, probs)}
    top = max(scores, key=scores.get)
    return top, scores