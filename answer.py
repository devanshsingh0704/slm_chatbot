"""Retrieve top chunks and answer with an Ollama model. Usage: python answer.py <dataset> "question" [model]"""
import difflib
import json
import os
import re
import sys
import urllib.request

import numpy as np

from retrieval import Store, config, expand

# Override to use another Ollama, e.g. the GPU laptop through an SSH tunnel on port 11437.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_LLM = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
# Give up when no answer has started within this many seconds (the model server queues questions and
# answers them one at a time, so a long queue would otherwise leave people waiting minutes).
ANSWER_TIMEOUT = int(os.environ.get("ANSWER_TIMEOUT", "120"))

SYSTEM = """You are {assistant}.
Answer the question using only the context the user gives you. Decide in this order:

1. If the context answers the question, answer it directly in 1-4 sentences or a short list.
2. If the context is about the question's topic but says the detail is not published, not confirmed, or unverified, say that plainly. When the context itself marks a claim with words like "unverified", "not confirmed", or "one source claims", repeat that caveat next to the claim; never present such a claim as fact.
   Naming where a fact comes from (a video, a case study, a company page, "documented", "observed", "named", "stated") is NOT a doubt: state such facts plainly, without adding "unverified".
3. Only if nothing in the context is about the question, reply exactly: "{fallback}"

Never add facts, numbers, names, or steps that are not in the context. Do not write code, do calculations, or answer general-knowledge questions.
Write plain text without Markdown symbols such as double asterisks or link brackets. Never mention "the context" or "the provided information": the reader cannot see it."""


# Greetings are answered directly: sent through the documents they only produced the fallback ("hi" ->
# "I don't have that information"). Matched on the whole message, so "hi, who is the CEO?" still
# goes to the documents; close spellings count ("good morming").
GREETINGS = {
    "hi": "hello", "hii": "hello", "hello": "hello", "hey": "hello", "hi there": "hello", "hello there": "hello",
    "how are you": "how", "how r u": "how", "how are you doing": "how",
    "who are you": "hello", "what are you": "hello",  # listed so they don't fuzzy-match "how are you"
    "hi how are you": "how", "hello how are you": "how", "hey how are you": "how",
    "good morning": "morning", "good afternoon": "afternoon", "good evening": "evening", "good night": "night",
}
GREETING_REPLIES = {
    "hello": "Hello! I'm the {title}. How can I help you today?",
    "how": "I'm doing well, thank you! How can I help you today?",
    "morning": "Good morning! How can I help you today?",
    "afternoon": "Good afternoon! How can I help you today?",
    "evening": "Good evening! How can I help you today?",
    "night": "Good night! Feel free to come back anytime you have a question.",
}


# Words people add to a greeting ("hi chatbot"); dropped before matching.
GREETING_FILLER = {"chatbot", "bot", "assistant", "there", "dear", "buddy", "friend", "sir", "madam"}


def greeting(question, cfg):
    """The reply to a message that is only a greeting, else None."""
    text = " ".join(w for w in re.findall(r"[a-z]+", question.lower()) if w not in GREETING_FILLER)
    match = difflib.get_close_matches(text, GREETINGS, n=1, cutoff=0.8)
    return GREETING_REPLIES[GREETINGS[match[0]]].format(title=cfg["title"]) if match else None


# A message that starts with a greeting and goes on ("Hello, what do you make?") is greeted back first;
# the greeting is cut from the question so search and model see only the question.
OPENING = re.compile(r"(?i)^\s*(?:(hi+|hello|hey)|good (morning|afternoon|evening))\b(?:\s+(?:" + "|".join(GREETING_FILLER)
                     + r")\b)?[\s,.!:-]+(?=\S)")


def opening(question):
    """(greeting to say first, question without it); ("", question) when it doesn't start with a greeting."""
    m = OPENING.match(question)
    if not m:
        return "", question
    return ("Hello! " if m[1] else f"Good {m[2].lower()}! "), question[m.end():]


# "Connect me to Mr. Smith": people named in the documents usually have no direct contact listed. The model,
# even when told how to answer, said "Sorry, I don't have that information", so config "people" gives a fixed reply:
# who they are, that their direct contact is not listed, and the general (or their department's) contact.
CONTACT_WORDS = re.compile(r"\b(contact|connect|reach|talk|speak|call|phone|number|mobile|email|e-mail|mail|meet|whatsapp)\b", re.I)


def person_contact(question, store, dataset):
    """The fixed reply to a request to contact one listed person, else None."""
    cfg = config(dataset)
    if not cfg.get("people") or not CONTACT_WORDS.search(question):
        return None
    words = set(re.findall(r"[a-z]+", expand(question, dataset, store.vocab(dataset)).lower()))
    named = [p for p in cfg["people"] if words & set(p["match"])]
    if len(named) != 1:  # nobody, or ambiguous (two people share the surname)
        return None
    p = named[0]
    first = p["name"].split()[0]
    return (f"{p['name']} is {p['role']}. Direct contact details for {first} aren't listed, "
            f"but {p.get('contact', cfg['general_contact'])}.")


# A fixed answer fires only if the question contains one of its keywords AND is this close to one of its
# examples. Similarity alone failed: short "<company> ..." questions (email, other name) scored 0.87, above real
# address questions (0.81). With the keyword gate, all other test/log questions stay <= 0.72.
FIXED_MATCH = 0.80
_fixed_vecs = {}


def fixed_answer(question, store, dataset):
    """An approved, word-for-word answer from config.json "fixed_answers" when the question means the same
    as one of its examples, else None. Used where the model, given the right facts, still varied how much
    of a list it repeated (a list of site addresses: sometimes only the first)."""
    fixed = config(dataset).get("fixed_answers", [])
    if not fixed:
        return None
    if dataset not in _fixed_vecs:
        _fixed_vecs[dataset] = [(np.array(store.embed(f["examples"])), f["answer"]) for f in fixed]
    text = expand(question, dataset, store.vocab(dataset))
    q = np.array(store.embed([text])[0])
    best = max(((float((vecs @ q).max()) if any(k in text.lower() for k in f["keywords"]) else 0.0, answer)
                for (vecs, answer), f in zip(_fixed_vecs[dataset], fixed)), key=lambda x: x[0])
    return best[1] if best[0] >= FIXED_MATCH else None


# Small talk ("talk to me", "thanks", "tell me a joke") gets its own short prompt without documents: sent with
# document chunks, it got "-", a random fact, or the fallback. A message counts as small talk when it is this
# close to one of these phrases: measured 0.82-0.98 for small talk, at most 0.77 for 231 real questions.
SMALL_TALK = ["talk to me", "let's chat", "can we just chat", "tell me a joke", "make me laugh", "thanks", "thank you",
              "thank you so much", "you are very helpful", "good job", "nice", "great", "ok", "okay", "cool",
              "how is your day", "how is your day going", "what's up", "i am bored", "i'm tired", "what is your name",
              "what's your name", "are you a robot", "are you human", "do you like your job", "you are funny", "bye",
              "goodbye", "see you", "i love you", "you are stupid", "you are wrong", "what can you do",
              "how can you help me", "what are you meant to do", "help", "what do you know",
              "tell me something interesting", "sing a song", "how old are you"]
SMALL_TALK_MATCH = 0.80
SMALL_TALK_SYSTEM = """You are {assistant}; your name is the {title}. The user is making small talk or asking about you, not
asking about the company. Reply in 1-2 warm, natural sentences; light, work-friendly humour is welcome. You answer staff
questions about the company from its documents. Do not state any facts about the company, its people, products, numbers,
or contacts, and do not invent anything about yourself, the weather, the season, or the news.
End by inviting a question about the company. Plain text only."""
_small_talk_vecs = []


def small_talk(question, store):
    """True if the message is small talk rather than a question for the documents."""
    if not _small_talk_vecs:
        _small_talk_vecs.append(np.array(store.embed(SMALL_TALK, query=False)))
    return float((_small_talk_vecs[0] @ np.array(store.embed([question])[0])).max()) >= SMALL_TALK_MATCH


HISTORY_NOTE = """
Earlier messages are the conversation so far: use them only to understand what the question refers to
("he", "it", "that", "what about sales?"). Take every fact from the context."""


def retrieve(question, previous, store, dataset):
    """Chunks for a question; previous = the conversation's earlier questions, oldest first.
    A follow-up ("what does he do?") is searched alone and together with the last two questions, top 2
    each (at most 4 chunks): that found the right chunk in all of followup_questions.csv, alone 11/19."""
    if not previous:
        return store.search(question, dataset)
    hits = {}
    for score, chunk in (store.search(question, dataset, 2)
                         + store.search(" ".join(previous[-2:] + [question]), dataset, 2)):
        hits.setdefault(chunk["id"], (score, chunk))
    return sorted(hits.values(), key=lambda hit: -hit[0])


def stream(question, hits, cfg, model=DEFAULT_LLM, history=()):
    """Yield the answer in pieces as the model generates it, from already-retrieved hits.
    history = earlier turns [{"question", "answer"}], oldest first."""
    context = "\n\n".join(c["text"] for _, c in hits)
    if hits:
        system = SYSTEM.format(assistant=cfg["assistant"], fallback=cfg["fallback"]) + (HISTORY_NOTE if history else "")
    else:  # small talk: no documents
        system = SMALL_TALK_SYSTEM.format(assistant=cfg["assistant"], title=cfg["title"])
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        if turn["answer"]:
            messages.append({"role": "assistant", "content": turn["answer"]})
    # Naming the previous question tells the model which "he" is meant: with three earlier turns
    # (the company, the MD, the CEO) "what does he do?" got the fallback; with this line, the CEO's role.
    previous = f"Previous question: {history[-1]['question']}\n" if history else ""
    messages.append({"role": "user", "content": f"Context:\n{context}\n\n{previous}Question: {question}" if hits
                     else question})
    body = {"model": model, "stream": True, "options": {"temperature": 0, "seed": 0}, "messages": messages}
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=ANSWER_TIMEOUT) as r:
        for line in r:
            part = json.loads(line)
            if part.get("done"):
                return
            yield part["message"]["content"]
    # Ollama reports backend failures (e.g. GPU out of memory) as an empty, unfinished reply.
    raise RuntimeError(f"Ollama stopped before finishing; check its server log. Last message: {part}")


def ask(question, store, dataset, model=DEFAULT_LLM, k=3):
    """Return (answer text, retrieved chunk ids)."""
    if reply := greeting(question, config(dataset)):
        return reply, []
    hello, question = opening(question)
    if reply := person_contact(question, store, dataset) or fixed_answer(question, store, dataset):
        return hello + reply, []
    hits = [] if small_talk(question, store) else store.search(question, dataset, k)
    return hello + "".join(stream(question, hits, config(dataset), model)).strip(), [c["id"] for _, c in hits]


if __name__ == "__main__":
    text, ids = ask(sys.argv[2], Store(), sys.argv[1], *sys.argv[3:4])
    print(text, f"\n(retrieved: {', '.join(ids)})")
