#!/usr/bin/env python3
import csv, json, sys

# Usage: python to_jsonl.py input.csv output.jsonl
# CSV must have: subject, body, label

inp, outp = sys.argv[1], sys.argv[2]
with open(inp, newline='', encoding='utf-8') as f, open(outp, 'w', encoding='utf-8') as g:
    reader = csv.DictReader(f)
    for row in reader:
        obj = {
            "subject": row.get("subject","").strip(),
            "body": row.get("body","").strip(),
            "label": row.get("label","").strip()
        }
        g.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("Wrote", outp)
