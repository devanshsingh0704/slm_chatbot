# In-house knowledge chatbot

A small, self-hosted assistant that answers questions **only from your own documents** (retrieval-augmented
generation), with typed or spoken questions and spoken answers. Everything runs on your own machines: a local
model through [Ollama](https://ollama.com), local speech recognition and a local voice. No cloud services.

One code base, many datasets: each company or team gets a folder under `datasets/<name>/` with its reviewed
documents and a settings file. `datasets/demo/` is a small **fictional** company to try it with.

## What it does
- **Answers from the documents only**, or says a fixed fallback ("Sorry, currently I don't have that information").
  Claims the documents mark as unverified are repeated with that caveat, never stated as fact.
- **Follow-up questions** ("Who is the CEO?" → "What does he do?"): the page sends the last 3 turns.
- **Greetings and small talk** are answered directly; small talk gets a short friendly reply with no company facts.
- **Fixed answers** for questions that must be word-for-word (addresses), and **person contact requests**
  ("Connect me to Mr. Smith") answered from a people list.
- **Search that handles names and typos:** meaning-based search plus exact matching of rare words (names, codes),
  a glossary for company abbreviations, and spelling correction against the documents' own words.
- **Voice:** 🎤 speech-to-text (faster-whisper) and spoken answers read while the text appears (Piper), with
  rules so the voice says units, ranges, dates, phone numbers and emails properly.
- **Feedback:** 👍/👎 on every answer, a password-protected review page of 👎 answers, a log of every question.

## How an answer is made
1. Greetings, person-contact requests and fixed answers (`config.json`) reply instantly. Small talk gets a
   short reply from the model without documents. A greeting at the start of a question is answered first.
2. Otherwise the question (with glossary and typo fixes) is matched to the 3 closest chunks in a Chroma vector
   store (bge-small embeddings, `retrieval.py`); chunks containing a rare word of the question are added.
   A follow-up is also searched together with the previous questions.
3. The model (`qwen2.5:7b` recommended, `answer.py`) answers from those chunks only.
4. The page speaks the answer phrase by phrase while showing it (`/speak`).

## Quick start (one machine)
Needs Python 3.13 and [Ollama](https://ollama.com). A GPU makes answers fast; without one it still works, slowly.
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build.py demo              # chunks.md -> vector store
.venv/bin/python eval_retrieval.py demo     # is the right chunk found?
ollama pull qwen2.5:7b                      # or qwen2.5:3b on a CPU-only machine
mkdir -p .cache/piper && for f in onnx onnx.json; do curl -L -o .cache/piper/en_US-joe-medium.$f \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.$f; done
OLLAMA_MODEL=qwen2.5:7b .venv/bin/python chat_server.py demo   # then open http://127.0.0.1:8000
```
Without an NVIDIA GPU add `WHISPER_DEVICE=cpu` for the microphone. Browsers allow the microphone only on
`localhost` or HTTPS: for other machines set `TLS_CERT` and `TLS_KEY` (a self-signed certificate works).

## Your own dataset
1. `mkdir -p datasets/<name>/raw`, put PDF / Markdown / text / JSON files in `raw/`, copy `datasets/demo/config.json`.
2. `.venv/bin/python ingest.py <name>` drafts `datasets/<name>/chunks.md` and writes `ingest_report.md`
   (duplicates, conflicting numbers, facts only one document states). **Read and fix `chunks.md` by hand:**
   it is the source of truth. Format: `## <id> | <topic>`, a `sources:` line, then the text. Mark a doubtful
   claim right after it: `up to 3 mm (unverified: ...)`. A topic should say what people ask.
3. Settings in `config.json`:

| Key | What it is |
|---|---|
| `title`, `assistant`, `fallback` | the assistant's name, its one-line role (used in prompts), the "don't know" reply |
| `aliases` | glossary: `"CTO": "chief technical officer"` |
| `fixed_answers` | `examples` + `keywords` + exact `answer`, for answers that must not vary |
| `people`, `general_contact` | name, `match` words, `role`, optional own `contact`: replies to "connect me to …" |
| `speech_hint` | names and terms the speech recognizer should expect |
| `pronunciations` | how the voice should say a word (`"Lindqvist": "Lind kvist"`); the screen text is unchanged |

4. Write tests (below), run `build.py <name>`, then serve with `chat_server.py <name> [host] [port]`.

Not handled yet: Word/Excel files (convert to PDF), scanned documents (need OCR), per-user access control
(do not load documents some users must not see), and replacing a changed document automatically.

## Tests
| Script | Checks | Needs the model |
|---|---|---|
| `eval_retrieval.py <dataset>` | right chunk in the top 3 for `test_questions.csv` (gate ≥ 90%) | no |
| `eval_followups.py <dataset> [model]` | follow-ups, topic switches, 3-question chains from `followup_questions.csv` (gate 100%) | only with a model name |
| `eval_answers.py <dataset> [model]` | answers every test question; counts fallbacks, missing caveats, over-hedging; saves answers to read | yes |
| `test_speech.py <dataset>` | how answers are rewritten for the voice | no |

`test_questions.csv`: `question, expected_answer, source` (chunk ids, `;`-separated), `type`
(`answer`, `not_verified`, `refuse`, `chat`). On the demo dataset: retrieval 14/14, follow-ups 7/7, speech 17/17.

## Running it as a service
`deploy/` has example systemd user services (`chatbot-ollama`, `chatbot-chat`). To copy code and a dataset to
the GPU machine, rebuild the store there and restart: `DEPLOY_HOST=user@host DATASET=<name> ./deploy.sh`;
`./deploy.sh log` copies the live chat log back. The model unloads after 10 idle minutes (`OLLAMA_KEEP_ALIVE`),
so a shared GPU is freed between uses. Limits in `chat_server.py`: 75 questions in progress at once (others get
"Server busy"), 120 s answer timeout.

## Files
| File | What it does |
|---|---|
| `ingest.py` | raw files → draft chunks + review report |
| `build.py` | `chunks.md` → vector store (only changed chunks are re-embedded) |
| `retrieval.py` | search, rare-word matching, glossary, typo correction |
| `answer.py` | prompts, greetings, small talk, fixed answers, person contacts, model call |
| `chat_server.py` | web page, `/ask`, `/speak`, `/transcribe`, `/review`, HTTPS, the voice rules |
| `static/` | chat page and review page |
| `deploy.sh`, `deploy/` | deploy to a GPU machine; example services |
| `requirements.txt` | exact package versions |

## Licence
Code: see `LICENSE`. Models and voices are downloaded separately under their own licences (Qwen2.5,
BAAI/bge-small-en-v1.5, faster-whisper, Piper "joe" voice: CC0). Check them for your use.
