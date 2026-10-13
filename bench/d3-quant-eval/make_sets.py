"""Build the calibration, dev and test sets used to compare a quantized d3 with BF16.

usage: make_sets.py OUT_DIR

Source: the Hugging Face dataset tasksource/tasksource-jev-typed-decisions at a pinned
revision (the dataset was republished with a different mix on 2026-10-13).
Rows with a state over 6,000 characters are dropped, and so are choice or score
rows whose options repeat (the answer would be ambiguous).

  calib.json          128 rows of the first train shard, for AWQ or GPTQ calibration
  dev200.json         200 other rows of that shard, for choosing what stays in BF16
  dev1000.json        1,000 other rows of that shard, same purpose, less noise
  eval600.json        600 rows of the test split, for quick transformers runs
  test_requests.jsonl every usable test row as a model runtime request
                      (tools/reduced_agreement.py in src/model-runtime)

The three train sets do not overlap, and nothing is chosen on the test split.
"""

import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "tasksource/tasksource-jev-typed-decisions"
REVISION = "28c391259c9e3e0fa110abc6276df95e7abe23cb"
KEEP = ["id", "kind", "state", "question", "options", "target", "source"]


def load(name):
    path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
    return [r for r in pq.read_table(path).to_pylist() if len(r["state"]) <= 6000]


def usable(row):
    return row["kind"] == "noul" or len(set(row["options"])) == len(row["options"])


def request(row):
    kind, options, target = row["kind"], row["options"], row["target"]
    if kind == "choice":
        q = {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {o: None for o in options},
        }
    elif kind == "score":
        q = {
            "type": "score",
            "instructions": row["question"],
            "criteria": list(options),
        }
    else:
        q = {"type": "noul", "instructions": row["question"]}
    if kind == "noul":
        label = "true" if target[0] >= 0.5 else "false"
    else:
        label = options[target.index(max(target))]
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {"q": q},
        "target": {"q": label},
        "source": row["source"],
    }


def dump(rows, path):
    json.dump(
        [{k: r[k] for k in KEEP} for r in rows], open(path, "w", encoding="utf-8")
    )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    train = load("data/train-00000-of-00006.parquet")
    random.Random(1).shuffle(train)
    dump(train[:128], out / "calib.json")
    dump(train[300:500], out / "dev200.json")
    dump(train[500:1500], out / "dev1000.json")
    test = load("data/test-00000-of-00001.parquet")
    with open(out / "test_requests.jsonl", "w", encoding="utf-8") as f:
        for row in test:
            if usable(row):
                f.write(json.dumps(request(row)) + "\n")
    random.Random(0).shuffle(test)
    dump(test[:600], out / "eval600.json")
    print(f"train rows {len(train)}, test rows {len(test)}")


if __name__ == "__main__":
    main()
