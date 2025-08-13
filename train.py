# train.py
import json, torch, numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

labels = ["Urgent","Normal","Low"]
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}
model_ckpt = "distilbert-base-uncased"

def load_jsonl(path):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def to_ds(items):
    return {"text":[(i.get("subject","") + "\n\n" + i.get("body","")).strip() for i in items],
            "label":[label2id[i["label"]] for i in items]}

train = to_ds(load_jsonl("data/train.jsonl"))
val   = to_ds(load_jsonl("data/val.jsonl"))
test  = to_ds(load_jsonl("data/test.jsonl"))

raw = DatasetDict({
    "train": load_dataset("json", data_files={"train":"data/train.jsonl"})["train"],
    "validation": load_dataset("json", data_files={"validation":"data/val.jsonl"})["validation"],
    "test": load_dataset("json", data_files={"test":"data/test.jsonl"})["test"]
}).map(lambda x: {"text": (x.get("subject","")+"\n\n"+x.get("body","")).strip(),
                  "label": label2id[x["label"]]})

tok = AutoTokenizer.from_pretrained(model_ckpt)
def tokenize(example): return tok(example["text"], truncation=True, max_length=384)
tokd = raw.map(tokenize, batched=True)
tokd = tokd.remove_columns([c for c in tokd["train"].column_names if c not in ["input_ids","attention_mask","label"]])

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# Optional class weights (boost Urgent)
class_counts = np.bincount([label2id[x["label"]] for x in load_jsonl("data/train.jsonl")], minlength=len(labels))
weights = class_counts.sum() / (len(labels) * class_counts)  # inverse freq
class_weights = torch.tensor(weights, dtype=torch.float)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], average=None)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "f1_macro": macro_f1,
        "f1_urgent": f1[0],
        "precision_urgent": pr[0],
        "recall_urgent": rc[0]
    }

args = TrainingArguments(
    output_dir="models/email-priority-clf",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_urgent",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokd["train"],
    eval_dataset=tokd["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model("models/email-priority-clf")
tok.save_pretrained("models/email-priority-clf")