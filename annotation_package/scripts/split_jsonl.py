#!/usr/bin/env python3
import json, random, sys
from collections import defaultdict

# Usage: python split_jsonl.py all.jsonl train.jsonl val.jsonl test.jsonl 0.7 0.15 0.15

inp, train_p, val_p, test_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
p_train, p_val, p_test = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])

items = [json.loads(l) for l in open(inp, encoding='utf-8')]
random.shuffle(items)

n = len(items)
n_train = int(n * p_train)
n_val = int(n * p_val)
train = items[:n_train]
val = items[n_train:n_train+n_val]
test = items[n_train+n_val:]

for path, data in [(train_p, train), (val_p, val), (test_p, test)]:
    with open(path, 'w', encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("Splits saved.")
