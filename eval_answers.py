"""Answer gate: answer every test question, save the answers for grading, print automatic checks.
Usage: python eval_answers.py <dataset> [ollama_model ...]
Auto-checks only catch fallbacks and missing hedges; correctness is graded by reading the saved answers."""
import csv
import json
import sys
import time

from answer import DEFAULT_LLM, ask
from retrieval import DATASETS, ROOT, Store, config

# Phrases that show the model passed on a "not verified" boundary instead of inventing a value.
HEDGES = ("not verified", "unverified", "not confirmed", "not published", "not publicly",
          "not stated", "not specified", "not disclosed", "not available", "one source")
# Casting doubt on the answer itself. On an answerable question whose expected answer has no doubt,
# this is over-hedging ("Kashyap is the CTO, but this is unverified"). A trailing "X is not published"
# caveat is fine and not counted.
DOUBTS = ("unverified", "not verified", "not confirmed", "one source")

dataset = sys.argv[1]
fallback = config(dataset)["fallback"]
questions = list(csv.DictReader((DATASETS / dataset / "test_questions.csv").open()))
store = Store()
out_dir = ROOT / "results" / dataset
out_dir.mkdir(parents=True, exist_ok=True)

for model in sys.argv[2:] or [DEFAULT_LLM]:
    ask("warm-up", store, dataset, model)  # exclude model load time from the latency numbers
    out = out_dir / f"answers_{model.replace(':', '_')}.jsonl"
    checks, times = {}, []
    with out.open("w") as f:
        for q in questions:
            t = time.perf_counter()
            text, retrieved = ask(q["question"], store, dataset, model)
            times.append(time.perf_counter() - t)
            fell_back = text.startswith(fallback.rstrip("."))  # with or without the final full stop
            ok = {"answer": not fell_back,
                  "refuse": fell_back,
                  "not_verified": not fell_back and any(h in text.lower() for h in HEDGES),
                  "chat": not fell_back}[q["type"]]  # small talk: also read it for invented company facts
            if q["type"] == "answer" and ok and any(h in text.lower() for h in DOUBTS) \
                    and not any(h in q["expected_answer"].lower() for h in DOUBTS):
                ok = False
                checks.setdefault("over-hedged", []).append(True)
            checks.setdefault(q["type"], []).append(ok)
            f.write(json.dumps({**q, "model": model, "retrieved": retrieved, "reply": text,
                                "auto_ok": ok, "seconds": round(times[-1], 1)}, ensure_ascii=False) + "\n")
            print(f"{'ok ' if ok else 'BAD'} {times[-1]:5.1f}s [{q['type']}] {q['question'][:50]!r} -> {text[:110]!r}")

    times.sort()
    print(f"\n=== {dataset} / {model}: saved {out.relative_to(ROOT)}")
    for kind, oks in checks.items():
        if kind != "over-hedged":
            print(f"  {kind:13} auto-check {sum(oks)}/{len(oks)}")
    print(f"  over-hedged   {len(checks.get('over-hedged', []))} answerable questions doubted without reason")
    print(f"  latency median {times[len(times) // 2]:.1f}s  max {times[-1]:.1f}s")
