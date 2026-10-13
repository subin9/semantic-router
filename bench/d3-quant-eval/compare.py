"""Agreement of a quantized run with the BF16 run: usage compare.py BF16_JSON OTHER_JSON..."""

import json
import sys

base = {r["id"]: r for r in json.load(open(sys.argv[1]))["rows"]}
print(json.dumps(json.load(open(sys.argv[1]))["summary"]))
for path in sys.argv[2:]:
    run = json.load(open(path))
    same, diffs, n = 0, [], 0
    for r in run["rows"]:
        b = base.get(r["id"])
        if b is None:
            continue
        n += 1
        bv, rv = list(b["probs"].values()), list(r["probs"].values())
        same += bv.index(max(bv)) == rv.index(max(rv))
        diffs.append(max(abs(x - y) for x, y in zip(bv, rv)))
    diffs.sort()
    print(json.dumps(run["summary"]))
    print(
        f"  agreement {same}/{n} = {same / n:.4f}  max|dp| median {diffs[n // 2]:.4f}"
        f" p95 {diffs[int(n * 0.95)]:.4f} max {diffs[-1]:.4f}"
    )
