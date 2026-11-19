"""Chat page for one dataset. Usage: python chat_server.py <dataset> [host] [port]   (default 127.0.0.1 8000)
Backend: OLLAMA_URL / OLLAMA_MODEL env vars, same as answer.py. Speech input: WHISPER_MODEL / WHISPER_DEVICE.
HTTPS (browsers only allow the microphone on https or localhost): set TLS_CERT and TLS_KEY.
Spoken answers: Piper voice file in PIPER_VOICE (default .cache/piper/en_US-joe-medium.onnx), runs on CPU.
Every question, answer, and thumbs-up/down is appended to datasets/<dataset>/chat_log.jsonl."""
import base64
import calendar
import hmac
import html
import io
import json
import ssl
import os
import re
import sys
import threading
import time
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from answer import DEFAULT_LLM, fixed_answer, greeting, opening, person_contact, retrieve, small_talk, stream
from retrieval import DATASETS, ROOT, Store, config

DATASET = sys.argv[1]
CFG = config(DATASET)
LOG = DATASETS / DATASET / "chat_log.jsonl"
PAGE = (ROOT / "static" / "chat.html").read_text().replace("{{TITLE}}", html.escape(CFG["title"])).encode()
REVIEW = (ROOT / "static" / "review.html").read_text().replace("{{TITLE}}", html.escape(CFG["title"])).encode()
store = Store()
BUSY = "Server busy, please try again in a moment."
# At most this many questions in progress at once (answering or waiting for the model); the next ones
# get BUSY right away instead of waiting minutes in the model server's queue.
MAX_ACTIVE = int(os.environ.get("MAX_ACTIVE", "75"))
active = threading.BoundedSemaphore(MAX_ACTIVE)
log_lock = threading.Lock()
# Speech to text (English only), loaded on first use and then kept; one recording at a time.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
MAX_AUDIO = 5_000_000  # bytes; a spoken question is ~50 KB
speech_lock = threading.Lock()
speech_model = None


PIPER_VOICE = os.environ.get("PIPER_VOICE", str(ROOT / ".cache" / "piper" / "en_US-joe-medium.onnx"))
MAX_SPEAK = 2000  # characters; longer answers are not read aloud
# /review and /log.json show every question anyone asked: only with this password (any user name).
# Not set -> both stay closed.
REVIEW_PASSWORD = os.environ.get("REVIEW_PASSWORD", "")
MAX_TURNS = 3  # earlier question/answer pairs the model sees, so follow-ups ("what does he do?") work
voice_lock = threading.Lock()
voice = None
# Names the English voice says wrongly, respelled for the voice only (config "pronunciations").
SAY = {k.lower(): v for k, v in CFG.get("pronunciations", {}).items()}
SAY_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, SAY)) + r")\b", re.I) if SAY else None
# The voice reads unit symbols as letters ("mm" -> "em em").
UNITS = {"mm": "millimetres", "cm": "centimetres", "m": "metres", "km": "kilometres", "kg": "kilograms",
         "t": "tonnes", "kW": "kilowatts", "°C": "degrees Celsius"}


def speech_text(text):
    """Answer text rewritten for the voice only: list dashes dropped, "No. 45" -> "Number 45",
    emails spelled out, dates, ranges and units said in words, names respelled, acronyms (HR,
    CEO) said letter by letter, and phones / PIN codes read digit by digit with a pause between digit
    groups. Checked against the voice's phonemes and by re-transcribing."""
    text = re.sub(r"(?m)^\s*[-*•]\s*", "", text).replace("\n", ". ")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text).replace("**", "")  # Markdown the model may still write
    text = re.sub(r"\bNo\.?\s*(?=\d)", "Number ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
                  lambda m: m.group().replace("@", " at ").replace(".", " dot "), text)
    text = re.sub(r"\b(\d{4})-(\d\d)-(\d\d)\b",
                  lambda m: f"{int(m[3])} {calendar.month_name[int(m[2])]} {m[1]}", text)
    # Phones before ranges, or "+1 555-0102" became "555 to 0102": digit by digit, a pause between groups.
    text = re.sub(r"\+\d+(?:[ -]\d+)*", lambda m: "plus " + ". ".join(
        ", ".join(group) for group in re.split(r"[ -]", m[0][1:])), text)
    text = re.sub(r"(\d)\s*[–—-]\s*(?=\d)", r"\1 to ", text)  # 200–250 -> 200 to 250
    text = re.sub(r"(\d)\s?(" + "|".join(map(re.escape, UNITS)) + r")(?![\w°])",
                  lambda m: f"{m[1]} {UNITS[m[2]]}", text)
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", ", ", text)  # CAD/CAM -> CAD, CAM
    text = re.sub(r"\b19([1-9]\d)\b", r"19 \1", text)  # 1999 -> nineteen ninety-nine
    if SAY_RE:
        text = SAY_RE.sub(lambda m: SAY[m.group().lower()], text)
    text = re.sub(r"\b[A-Z]{2,5}\b", lambda m: " ".join(m.group()), text)
    # PIN codes and other long numbers: 123456 -> "1, 2, 3, 4, 5, 6", not an amount
    return re.sub(r"\d{5,}(?: \d{5,})*", lambda m: ". ".join(", ".join(group) for group in m[0].split()), text)


def speak(text):
    """WAV bytes of the text read aloud (Piper, loaded on first use, one at a time)."""
    global voice
    with voice_lock:
        if voice is None:
            import onnxruntime
            from piper import PiperVoice
            voice = PiperVoice.load(PIPER_VOICE)
            # By default one sentence is spread over every core; on the 24-core GPU laptop, next to other
            # jobs, that took ~10 s per sentence, against ~1 s with 4 threads (measured 2026-10-17).
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            voice.session = onnxruntime.InferenceSession(PIPER_VOICE, sess_options=options,
                                                         providers=["CPUExecutionProvider"])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(speech_text(text), w)
        return buf.getvalue()


def transcribe(audio):
    global speech_model
    with speech_lock:
        if speech_model is None:
            from faster_whisper import WhisperModel  # only needed when someone uses the microphone
            speech_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                        compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
                                        download_root=str(ROOT / ".cache" / "whisper"))
        # The hint lists names and terms the model would otherwise mishear (people's and product names).
        segments, _ = speech_model.transcribe(io.BytesIO(audio), language="en", vad_filter=True,
                                              initial_prompt=CFG.get("speech_hint"))
        return " ".join(s.text.strip() for s in segments).strip()


def log(record):
    with log_lock, LOG.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **record},
                           ensure_ascii=False) + "\n")


def with_hello(hello, pieces):
    """The model's answer, after the greeting that opened the question ("Hello! ")."""
    if hello:
        yield hello
    yield from pieces


class Handler(BaseHTTPRequestHandler):
    def reviewer(self):
        """True if the request carries the review password; otherwise answers 401/403 and returns False."""
        if not REVIEW_PASSWORD:
            self.send_error(403, "Review is turned off: set REVIEW_PASSWORD")
            return False
        auth = self.headers.get("Authorization", "")
        try:
            password = base64.b64decode(auth.removeprefix("Basic ")).decode().partition(":")[2]
        except ValueError:
            password = ""
        if auth.startswith("Basic ") and hmac.compare_digest(password.encode(), REVIEW_PASSWORD.encode()):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="review", charset="UTF-8"')
        self.end_headers()
        return False

    def do_GET(self):
        self.path = self.path.split("?")[0]
        if self.path in ("/review", "/log.json") and not self.reviewer():
            return
        if self.path == "/log.json":  # every logged answer with its latest rating, for the review page
            answers, ratings = [], {}
            for line in (LOG.open() if LOG.exists() else []):
                r = json.loads(line)
                if r["kind"] == "answer":
                    answers.append(r)
                else:
                    ratings[r["id"]] = r["good"]
            for a in answers:
                a["good"] = ratings.get(a["id"])
            body, ctype = json.dumps(answers).encode(), "application/json"
        elif self.path in ("/", "/review"):
            body, ctype = PAGE if self.path == "/" else REVIEW, "text/html; charset=utf-8"
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path == "/transcribe":
            if not 0 < length <= MAX_AUDIO:
                return self.send_error(413 if length else 400)
            try:
                result = {"text": transcribe(self.rfile.read(length))}
            except Exception as e:  # undecodable audio or model failure: the user can still type
                result = {"error": BUSY, "detail": str(e)[:200]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return self.wfile.write(json.dumps(result).encode())
        body = json.loads(self.rfile.read(length))
        if self.path == "/speak":
            text = str(body.get("text", "")).strip()
            if not 0 < len(text) <= MAX_SPEAK:
                return self.send_error(413 if text else 400)
            try:
                audio = speak(text)
            except Exception:  # voice file missing or synthesis failed: the text is still on screen
                return self.send_error(503, BUSY)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            return self.wfile.write(audio)
        if self.path == "/feedback":
            log({"kind": "feedback", "id": body["id"], "good": bool(body["good"])})
            self.send_response(204)
            return self.end_headers()
        if self.path != "/ask":
            return self.send_error(404)

        question = body["question"].strip()[:1000]
        # Sent by the page; trimmed and checked here, since it comes from the browser.
        history = [{"question": str(t.get("question", ""))[:1000], "answer": str(t.get("answer", ""))[:1500]}
                   for t in (body.get("history") if isinstance(body.get("history"), list) else [])[-MAX_TURNS:]
                   if isinstance(t, dict)]
        if not active.acquire(blocking=False):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write((json.dumps({"id": None, "sources": []}) + "\n" + BUSY).encode())
            return log({"kind": "answer", "id": uuid.uuid4().hex[:12], "question": question, "answer": "",
                        "retrieved": [], "model": DEFAULT_LLM, "seconds": 0,
                        "error": f"turned away: {MAX_ACTIVE} questions already in progress"})
        try:
            self.answer(question, history)
        finally:
            active.release()

    def answer(self, question, history):
        answer_id = uuid.uuid4().hex[:12]
        reply = greeting(question, CFG)
        hello, asked = ("", question) if reply else opening(question)  # "Hello, what ...?" -> "Hello! " + answer
        reply = reply or person_contact(asked, store, DATASET) or fixed_answer(asked, store, DATASET)
        chat = not reply and small_talk(asked, store)
        hits = [] if reply or chat else retrieve(asked, [t["question"] for t in history], store, DATASET)
        # Protocol: first line is JSON (id + sources), the rest is the answer text as it streams.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        sources = [{"id": c["id"], "topic": c["topic"], "score": round(s, 3), "text": c["text"]} for s, c in hits]
        self.wfile.write((json.dumps({"id": answer_id, "sources": sources}) + "\n").encode())

        start, parts, error = time.perf_counter(), [], None
        try:
            for piece in with_hello(hello, [reply] if reply else stream(asked, hits, CFG, history=history)):
                parts.append(piece)
                self.wfile.write(piece.encode())
                self.wfile.flush()
        except Exception as e:  # queue full (503), model server down, or no answer within ANSWER_TIMEOUT
            error = str(e)  # the real reason goes to the log; users get a friendly line
            self.wfile.write((("\n\n" if parts else "") + BUSY).encode())
        log({"kind": "answer", "id": answer_id, "question": question, "answer": "".join(parts).strip(),
             "previous": history[-1]["question"] if history else None,  # so a 👎 follow-up can be understood
             "retrieved": [s["id"] for s in sources], "model": "fixed" if reply else DEFAULT_LLM + (" small talk" if chat else ""),
             "seconds": round(time.perf_counter() - start, 1), "error": error})

    def log_message(self, *args):  # silence per-request console noise
        pass


if __name__ == "__main__":
    args = sys.argv[2:]
    host = args[0] if args else "127.0.0.1"
    port = int(args[1]) if len(args) > 1 else 8000
    # Python's default lets only 5 connections wait to be accepted; 75 people arriving at once got
    # "connection reset" before MAX_ACTIVE could even answer them.
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer((host, int(port)), Handler)
    scheme = "http"
    if os.environ.get("TLS_CERT"):
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(os.environ["TLS_CERT"], os.environ["TLS_KEY"])
        # handshake in each request's own thread, so one slow client cannot hold up everyone else
        server.socket = tls.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        scheme = "https"
    print(f"{CFG['title']} ({DATASET}) on {scheme}://{host}:{port}  model={DEFAULT_LLM}  log={LOG}")
    # Load and warm the voice now (CPU, ~60 MB) so the first answer is not delayed ~3 s by loading it.
    threading.Thread(target=speak, args=("Ready.",), daemon=True).start()
    server.serve_forever()
