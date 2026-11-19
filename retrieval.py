"""Persistent vector store: one Chroma collection, every chunk tagged with its dataset.
Usage: python retrieval.py <dataset|all> "question\""""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

ROOT = Path(__file__).parent
# Private data can live outside this repo: CHATBOT_DATASETS=/path/to/datasets
DATASETS = Path(os.environ.get("CHATBOT_DATASETS", ROOT / "datasets"))
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge was trained with it


def config(dataset):
    """datasets/<dataset>/config.json: title, assistant, fallback, optional aliases."""
    return json.loads((DATASETS / dataset / "config.json").read_text())


def expand(query, dataset, vocab=()):
    """Append glossary meanings ("CTO" -> "chief technical officer") and spelling fixes ("conatct" ->
    "contact"). The embedder knows neither company abbreviations nor typos: "how to conatct hr" misses
    the contact chunk that "how to contact hr" finds first. Appending keeps what the user typed."""
    aliases = config(dataset).get("aliases", {}) if dataset else {}
    extra = [full for short, full in aliases.items() if re.search(rf"\b{re.escape(short)}\b", query, re.I)]
    return " ".join([query, *extra, *spelling_fixes(query, vocab)])


WORDLIST = Path("/usr/share/dict/words")
ENGLISH = {w.strip().lower() for w in WORDLIST.open(errors="ignore")} if WORDLIST.exists() else set()


def spelling_fixes(query, vocab):
    """For each word of 5+ letters that is neither in the dataset vocabulary nor an English word, the
    closest vocabulary word within 1 edit (2 for 8+ letters), the more frequent one on a tie; swapped
    neighbours count as one edit. Real words are left alone: without that check "weight" became "height"
    and "share" "shape". Closest first: by frequency alone a misspelt name was "fixed" to a more frequent, more distant word."""
    fixes = []
    for w in re.findall(r"[a-z]+", query.lower()):
        if len(w) < 5 or w in vocab or w in ENGLISH:
            continue
        limit = 2 if len(w) >= 8 else 1
        near = [(edits(w, v, limit), -vocab[v], v) for v in vocab if abs(len(v) - len(w)) <= limit]
        near = [n for n in near if n[0] <= limit]
        if near:
            fixes.append(min(near)[2])
    return fixes


def edits(a, b, limit):
    """Optimal-string-alignment distance (Levenshtein plus adjacent swaps), stopping early past limit."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


RARE_CHUNKS = 2  # a word in more chunks than this is not specific enough to pull a chunk in


class Store:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(ROOT / "store"),
                                           settings=chromadb.Settings(anonymized_telemetry=False))
        # ef_search well above the chunk count makes the HNSW index effectively exact, so results do not
        # shift right after a rebuild (seen once: 56/56 -> 54/56). Raise it if a store grows past ~500 chunks.
        self.col = client.get_or_create_collection(
            "chunks", metadata={"model": MODEL},
            configuration={"hnsw": {"space": "cosine", "ef_search": 500, "ef_construction": 200}})
        if self.col.metadata.get("model") != MODEL:
            sys.exit(f"store was built with {self.col.metadata.get('model')}, not {MODEL}; delete store/ and rebuild")
        self.embedder = TextEmbedding(MODEL)
        self._vocab = {}
        self._where = {}

    def embed(self, texts, query=False):
        return [v.tolist() for v in self.embedder.embed([QUERY_PREFIX + t if query else t for t in texts])]

    def sync(self, dataset, chunks):
        """Make the store hold exactly these chunks for this dataset; embed only new or changed ones."""
        current = self.col.get(where={"dataset": dataset}, include=["metadatas"])
        have = {i: m["hash"] for i, m in zip(current["ids"], current["metadatas"])}
        want = {f"{dataset}:{c['id']}": c for c in chunks}
        for c in chunks:
            c["hash"] = hashlib.sha256(c["text"].encode()).hexdigest()[:16]
        changed = [k for k, c in want.items() if have.get(k) != c["hash"]]
        removed = [k for k in have if k not in want]
        if changed:
            self.col.upsert(
                ids=changed,
                embeddings=self.embed([want[k]["text"] for k in changed]),
                documents=[want[k]["text"] for k in changed],
                metadatas=[{"dataset": dataset, "id": want[k]["id"], "topic": want[k]["topic"],
                            "sources": "; ".join(want[k]["sources"]), "hash": want[k]["hash"]} for k in changed])
        if removed:
            self.col.delete(ids=removed)
        return len(changed), len(removed), len(want)

    def vocab(self, dataset):
        """Word counts of a dataset's chunks (cached; a server restart picks up rebuilt chunks)."""
        if dataset not in self._vocab:
            got = self.col.get(where={"dataset": dataset}, include=["documents"])
            self._vocab[dataset] = Counter(re.findall(r"[a-z]+", " ".join(got["documents"]).lower()))
            self._where[dataset] = defaultdict(set)  # word -> ids of the chunks that contain it
            for chunk_id, doc in zip(got["ids"], got["documents"]):
                for w in set(re.findall(r"[a-z]+", doc.lower())):
                    self._where[dataset][w].add(chunk_id)
        return self._vocab[dataset]

    def name_matches(self, query, dataset):
        """Ids of the chunks holding a rare word of the query: not an English word and in at most
        RARE_CHUNKS chunks, i.e. a name or code. Meaning-based search alone ranked the leadership chunk
        below general ones for questions that are only a name ("Who is <name>?"); names carry little meaning."""
        self.vocab(dataset)
        found = set()
        for w in set(re.findall(r"[a-z]+", query.lower())) - ENGLISH:
            ids = self._where[dataset].get(w, ())
            if 0 < len(ids) <= RARE_CHUNKS:
                found |= ids
        return found

    def search(self, query, dataset=None, k=3, query_is_text=False):
        """Return [(cosine score, chunk dict)] best first. dataset=None searches every dataset."""
        query = expand(query, dataset, self.vocab(dataset) if dataset else ())
        vector = self.embed([query], query=not query_is_text)
        res = self.col.query(query_embeddings=vector, n_results=k, where={"dataset": dataset} if dataset else None,
                             include=["documents", "metadatas", "distances"])
        hits = [(1 - d, {**m, "text": t})
                for d, m, t in zip(res["distances"][0], res["metadatas"][0], res["documents"][0])]
        # Chunks naming a rare word of the question replace the weakest results (k stays the same).
        missing = sorted(self.name_matches(query, dataset) - set(res["ids"][0]))[:k - 1] if dataset else []
        if missing:
            got = self.col.get(ids=missing, include=["documents", "metadatas", "embeddings"])
            extra = [(float(sum(a * b for a, b in zip(e, vector[0]))), {**m, "text": t})
                     for e, m, t in zip(got["embeddings"], got["metadatas"], got["documents"])]
            hits = sorted(hits[:k - len(extra)] + extra, key=lambda hit: -hit[0])
        return hits


if __name__ == "__main__":
    dataset = None if sys.argv[1] == "all" else sys.argv[1]
    for score, chunk in Store().search(sys.argv[2], dataset):
        print(f"{score:.3f}  {chunk['dataset']}:{chunk['id']}  {chunk['topic']}")
