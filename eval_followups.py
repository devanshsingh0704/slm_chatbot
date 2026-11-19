"""Follow-up gate: with the earlier questions of a conversation, does the right chunk come back?
Usage: python eval_followups.py <dataset> [model]
Reads datasets/<dataset>/followup_questions.csv (previous_question with earlier questions separated by " | ",
question, source chunk ids, type followup / switch / chain). With a model, the earlier questions are asked first,
as in the chat page, and every answer is printed to read."""
import csv
import sys

from answer import retrieve, stream
from retrieval import DATASETS, Store, config

dataset, model = sys.argv[1], sys.argv[2:3]
store = Store()
rows = list(csv.DictReader((DATASETS / dataset / "followup_questions.csv").open()))
hits_by_type, misses = {}, []
for r in rows:
    previous = [q.strip() for q in r["previous_question"].split("|")]
    hits = retrieve(r["question"], previous, store, dataset)
    got = [c["id"] for _, c in hits]
    ok = bool(set(r["source"].split(";")) & set(got))
    n = hits_by_type.setdefault(r["type"], [0, 0])
    n[0] += ok
    n[1] += 1
    if not ok:
        misses.append(f"  MISS {r['question']!r} after {previous}  got {got}")
    if model:
        history = []
        for q in previous:  # ask the earlier questions the way the chat page does
            earlier = retrieve(q, [t["question"] for t in history], store, dataset)
            history.append({"question": q, "answer": "".join(stream(q, earlier, config(dataset), model[0], history)).strip()})
        reply = "".join(stream(r["question"], hits, config(dataset), model[0], history)).strip()
        print(f"{' / '.join(previous)} -> {r['question']}\n  {reply}\n")
total = sum(n[0] for n in hits_by_type.values()), sum(n[1] for n in hits_by_type.values())
print(f"=== {dataset}: " + ", ".join(f"{t} {a}/{b}" for t, (a, b) in hits_by_type.items())
      + f"  total {total[0]}/{total[1]}  gate (100%): {'PASS' if total[0] == total[1] else 'FAIL'}")
print("\n".join(misses))
