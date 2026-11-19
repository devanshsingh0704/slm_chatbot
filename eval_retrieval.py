"""Retrieval gate: does the answer's content come back in the top 3? Usage: python eval_retrieval.py <dataset>
Two scores: chunk-ID hit (only when the test set's `source` IDs belong to this dataset) and
keyword recall (share of the expected answer's content words found in the retrieved text), which
works for any dataset because it needs no chunk IDs."""
import csv
import re
import sys

from retrieval import DATASETS, Store

K = 3
RECALL_HIT = 0.6  # calibrated on the first benchmark dataset against chunk-ID hits
STOP = set("""a an and are as at be but by can do does for from has have how in is it its of on or
our that the their this to was we what when where which who why will with you your not only""".split())


def keywords(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|\d", text.lower()) if w not in STOP}


dataset = sys.argv[1]
store = Store()
questions = [q for q in csv.DictReader((DATASETS / dataset / "test_questions.csv").open()) if q["type"] not in ("refuse", "chat")]
id_hits = recall_hits = 0
misses = []
for q in questions:
    results = store.search(q["question"], dataset, K)
    got_ids = [c["id"] for _, c in results]
    expected_ids = set(filter(None, q["source"].split(";")))
    id_hits += bool(expected_ids & set(got_ids))
    want = keywords(q["expected_answer"])
    found = keywords(" ".join(c["text"] for _, c in results))
    recall = len(want & found) / len(want) if want else 1.0
    recall_hits += recall >= RECALL_HIT
    if recall < RECALL_HIT:
        misses.append(f"  MISS recall {recall:.2f}  {q['question']!r}  missing {sorted(want - found)[:6]}  got {got_ids}")

n = len(questions)
print(f"=== {dataset}: {n} questions, top {K}")
if any(q["source"] for q in questions):
    print(f"chunk-ID hit:    {id_hits}/{n} = {id_hits / n:.0%}")
print(f"keyword recall:  {recall_hits}/{n} = {recall_hits / n:.0%}  (hit = recall >= {RECALL_HIT})  "
      f"gate (>=90%): {'PASS' if recall_hits / n >= 0.9 else 'FAIL'}")
print("\n".join(misses))
