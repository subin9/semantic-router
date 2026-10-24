"""Mixed calibration rows for awq.py: text rows plus Box and images-in-options requests, shuffled together.

usage: calib_mixed.py TEXT_CALIB_JSON DATA_DIR OUT_JSON [--prefix PATH]

DATA_DIR holds d3-turbo's research splits (box-v2/{train,dev,test}, compose-v1/{train,dev,test}) as
jsonl.gz rows with image paths relative to each file. 32 Box rows per family of box-v2 train (256) and
the first 64 of 11 per family of compose-v1 train are added to TEXT_CALIB_JSON, after dropping every row
that shares an image with a dev or test split, so BoxSet-dev can pick the package and BoxSet-test stays
untouched. The rows are shuffled so the first --loss-rows of awq.py hold both kinds. Image paths are
written under --prefix (DATA_DIR by default), the folder as the build container sees it.
"""

import argparse
import collections
import glob
import gzip
import json
import random
from pathlib import Path


def rows(data, split):
    for f in sorted(glob.glob(str(data / split / "*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            yield Path(f).parent, json.loads(line)


def paths(r):
    return list(r.get("images", [])) + list(r.get("assets", {}).values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("data", type=Path)
    p.add_argument("out")
    p.add_argument("--prefix")
    args = p.parse_args()
    data = args.data.resolve()
    prefix = args.prefix or str(data)

    def where(folder, path):
        return f"{prefix}/{(folder / path).resolve().relative_to(data)}"

    held = {
        (folder / path).resolve().name
        for split in ("box-v2/test", "box-v2/dev", "compose-v1/test", "compose-v1/dev")
        for folder, r in rows(data, split)
        for path in paths(r)
    }
    rnd = random.Random(0)
    out = json.load(open(args.text))
    for split, per, total in (("box-v2/train", 32, 256), ("compose-v1/train", 11, 64)):
        by = collections.defaultdict(list)
        for folder, r in rows(data, split):
            if not {(folder / path).resolve().name for path in paths(r)} & held:
                by[r["family"]].append((folder, r))
        picked = []
        for fam in sorted(by):
            picked += rnd.sample(by[fam], per)
        for folder, r in picked[:total]:
            request = {
                "state": r.get("state", ""),
                "questions": {"q": r["question"]},
                "images": [where(folder, path) for path in r.get("images", [])],
                "assets": {
                    k: where(folder, path) for k, path in r.get("assets", {}).items()
                },
            }
            out.append({"id": r["id"], "kind": "image", "request": request})
    rnd.shuffle(out)
    json.dump(out, open(args.out, "w"))
    print(
        len(out),
        dict(collections.Counter(r["kind"] for r in out)),
        "image rows in the first 96:",
        sum(r["kind"] == "image" for r in out[:96]),
        "held-out images:",
        len(held),
    )


if __name__ == "__main__":
    main()
