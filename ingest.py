"""Ingest new raw files into a dataset. Usage: python ingest.py <dataset>

raw/ files -> markdown -> skip duplicate files -> chunk by headings -> skip/flag duplicate chunks.
Appends draft chunks to datasets/<dataset>/chunks.md and writes ingest_report.md for review.
Files already ingested (same bytes) are ignored on later runs. Review, then run build.py."""
import hashlib
import json
import re
import sys
from datetime import datetime

import numpy as np
import pymupdf
import pymupdf4llm

from retrieval import DATASETS, Store

# Thresholds measured on the first benchmark dataset (same video data as .md/.pdf/.jsonl, see ingest_report.md).
# Duplicates are judged by shared words: embedding similarity ranks sibling sections of one document
# (0.93) above true rewrites (0.86-0.92), so it cannot tell "same text" from "same topic".
DOC_DUP = 0.8       # share of a document's distinct words found in another; both directions = same content
CHUNK_DUP = 0.75    # share of a chunk's content words found in a chunk from another document: skip it,
CHUNK_DUP_JACCARD = 0.4  # ...but only if the two are also similar in size, so a short row is never lost in a long chunk
CHUNK_MAYBE = 0.5   # ... at or above this: flag as possible duplicate
CONFLICT_SIM = 0.75 # embedding similarity to another document's chunk that also states different numbers
CONFLICTS_SHOWN = 25
MAX_WORDS = 220     # target chunk size
MIN_WORDS = 25      # a section intro shorter than this is merged into the following chunk
TABLE_SPLIT = 60    # tables longer than this become one chunk per row, so each row meets its own topic
SKIP_SECTIONS = {"references", "bibliography", "citations", "sources", "video sources"}
FORMAT_RANK = {".md": 0, ".txt": 1, ".pdf": 2, ".jsonl": 3, ".json": 3}  # preferred copy of identical content
IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


_ocr = []


def ocr(image):
    """Text read from an image (path or bytes) by RapidOCR, one line per text line. On 100 real scanned pages
    (MP-DocVQA) it read the answer on 90, a paid OCR (Textract) on 91; pymupdf4llm's own OCR route only on 66."""
    if not _ocr:
        from rapidocr import RapidOCR  # only needed for scans and images
        _ocr.append(RapidOCR())
    return "\n".join(_ocr[0](image).txts or [])


def scanned_pages(doc):
    """Page numbers of a PDF with no text layer (scans), which only OCR can read."""
    return [page.number for page in doc if not page.get_text().strip()]


def ocr_pdf_page(doc, page):
    """OCR one scanned PDF page: its scan image at full resolution if it is a single image, else a 300 dpi render."""
    images = page.get_images()
    data = doc.extract_image(images[0][0])["image"] if len(images) == 1 else page.get_pixmap(dpi=300).tobytes("png")
    return ocr(data)


def extract(path):
    """Return the file's content as markdown, or None if the type is unsupported.
    Scanned PDF pages and images are read by OCR (ocr()); pages with a text layer are read as text."""
    if path.suffix == ".pdf":
        doc = pymupdf.open(path)
        scans = scanned_pages(doc)
        if not scans:
            return pymupdf4llm.to_markdown(str(path))
        return "\n\n".join(f"## {path.stem}, page {n + 1}\n{ocr_pdf_page(doc, doc[n])}" if n in scans
                             else pymupdf4llm.to_markdown(doc, pages=[n]) for n in range(len(doc)))
    if path.suffix.lower() in IMAGES:
        return f"## {path.stem}\n{ocr(path)}"
    if path.suffix in (".md", ".txt"):
        return path.read_text()
    if path.suffix in (".jsonl", ".json"):
        text = path.read_text()
        records = [json.loads(l) for l in text.splitlines() if l.strip()] if path.suffix == ".jsonl" else json.loads(text)
        out = []
        for n, r in enumerate(records if isinstance(records, list) else [records], 1):
            title = next((str(r[k]) for k in ("title", "name", "id") if k in r), f"Record {n}")
            out.append(f"## {title}\n" + "\n".join(
                f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in r.items()))
        return "\n\n".join(out)
    return None


def clean(md):
    md = re.sub(r"\*\*|__", "", md).replace("<br>", " ")
    md = re.sub(r"(?<=[.)])(?:[ \t]+\d{1,2}){1,6}[ \t]*$", "", md, flags=re.M)  # citation markers: "ratio. 1 3"
    return re.sub(r"\n{3,}", "\n\n", md)


def words(text):
    return len(text.split())


def wordset(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def sections(md, fallback_title):
    """Yield (heading path, body lines) for every heading section."""
    stack, body = [], []
    title = next((m.group(1).strip() for m in re.finditer(r"^# (.+)$", md, re.M)), fallback_title)

    def emit():
        path = [t for _, t in stack]
        if not path or stack[0][0] != 1:
            path = [title] + path
        return path, body[:]

    for line in md.splitlines():
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if not m:
            body.append(line)
            continue
        if any(l.strip() for l in body):
            yield emit()
        body.clear()
        level = len(m.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, m.group(2).strip()))
    if any(l.strip() for l in body):
        yield emit()


def table_rows(block):
    """Markdown table lines -> one 'Header: value; ...' string per row, plus whether it looks broken."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")]
            for l in block if not re.fullmatch(r"\|[\s:|-]+\|", l.strip())]
    header = [h or f"column {i + 1}" for i, h in enumerate(rows[0])]
    data = [r for r in rows[1:] if any(r)]
    # PDF tables whose cells spill over into continuation rows come out with an empty first cell.
    broken = len(data) > 2 and sum(not r[0] for r in data) / len(data) > 0.3
    return ["; ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for r in data], broken


def chunk_section(lines):
    """Split one section's body into chunk texts; returns (texts, table_looked_broken)."""
    blocks, cur = [], []
    for line in lines + [""]:
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            blocks.append(cur)
            cur = []
    texts, buf, broken = [], [], False

    def flush():
        if buf:
            texts.append("\n".join(buf))
            buf.clear()

    for block in blocks:
        if all(l.lstrip().startswith("|") for l in block):
            flush()
            rows, bad = table_rows(block)
            broken |= bad
            texts.extend(["\n".join(rows)] if words(" ".join(rows)) <= TABLE_SPLIT else rows)
            continue
        para = "\n".join(block)
        if buf and words("\n".join(buf)) + words(para) > MAX_WORDS:
            flush()
        if words(para) > MAX_WORDS:  # one huge paragraph: split at sentence ends
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if buf and words("\n".join(buf)) + words(sentence) > MAX_WORDS:
                    flush()
                buf.append(sentence)
        else:
            buf.append(para)
    flush()
    return [t for t in texts if t.strip()], broken


STOP = set("""a an and are as at be but by can do does for from has have how in is it its of on or
our that the their this to was we what when where which who why will with you your not""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOP}


UNITS = r"(?:mm|cm|m|km|kg|t|tons?|tonnes?|%|kw|mw|mpa|hz|v|a|°c|crore|lakh|years?)"


def numbers(text):
    """Quantities stated in a text, as values so "8.0mm" and "8 mm" compare equal. A quantity has 2+ digits,
    a decimal point, or a unit; a lone digit ("Building 2", "Videos 1, 2 and 4") is a label and is ignored,
    as are times (00:22), numbers inside IDs (V-01, PROC-05) and list numbering ("3. ")."""
    text = re.sub(r"\d{1,2}:\d{2}", " ", text)
    found = re.findall(rf"(?<![\w-])(\d{{2,}}(?:[.,]\d+)?|\d+[.,]\d+|\d+(?=\s?{UNITS}\b))(?![\d-]|\.\s)", text, re.I)
    return {f"{float(n.replace(',', '')):g}" for n in found}


CONTACT = re.compile(r"@|https?://|www\.|\.com\b|\+?\d[\d ]{8,}")


def phrase_runs(md):
    """Word runs that never cross punctuation, so "rooftop solar, energy-efficient" does not contain
    the phrase "solar energy"."""
    return {" " + " ".join(re.findall(r"[a-z0-9]+", seg.replace("-", " "))) + " "
            for seg in re.split(r"[^\w\s-]", md.lower())}


def single_source_claims(new_docs, all_docs):
    """Labelled claims ("Solar Energy: ...") in the new documents whose label phrase appears in no
    other document. Template fields that repeat inside a document (Status:, Purpose:) and contact
    lines are left out. Being single-source is normal for many facts, so this is a list to scan,
    not a verdict; it is where claims like "7 assembly lines" or a market focus show up."""
    runs = {name: phrase_runs(md) for name, md in all_docs.items()}
    found = {}
    for name, md in new_docs.items():
        labels = re.findall(r"^\s*[-*]?\s*([A-Z][^:\n]{2,40}):\s", md, re.M)
        for m in re.finditer(r"^\s*[-*]\s*([A-Z][^:\n]{2,40}):\s*(.+)$", md, re.M):
            label, value = m.group(1).strip(), m.group(2).strip()
            if labels.count(m.group(1)) >= 3 or CONTACT.search(value):
                continue
            key = " " + " ".join(w for w in re.findall(r"[a-z0-9]+", label.lower().replace("-", " ")) if w not in STOP) + " "
            if not any(key in seg for other, segs in runs.items() if other != name for seg in segs):
                found.setdefault(name, []).append(f"{label}: {value}")
    return found


def main(dataset):
    d = DATASETS / dataset
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    (d / "extracted").mkdir(exist_ok=True)
    report = {"files": [], "dropped": [], "skipped": [], "review": [], "cross": []}

    # 1. New files only (same bytes as an earlier run = already handled).
    new = []
    for path in sorted((d / "raw").iterdir()):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in manifest["files"]:
            if manifest["files"][sha]["name"] != path.name:
                report["files"].append((path.name, f"skipped: identical copy of {manifest['files'][sha]['name']}"))
            continue
        md = extract(path)
        if md is None:
            manifest["files"][sha] = {"name": path.name, "status": "unsupported"}
            report["files"].append((path.name, "skipped: unsupported file type"))
            continue
        new.append({"sha": sha, "path": path, "md": clean(md)})

    # 2. Whole-file duplicates: same content in another format, against earlier runs and this batch.
    kept = [{"name": i["name"], "doc_id": i["doc_id"], "words": wordset((d / "extracted" / f"{i['doc_id']}.md").read_text())}
            for i in manifest["files"].values() if i["status"] == "ingested"]
    next_doc = 1 + max((int(k["doc_id"][1:]) for k in kept), default=0)
    docs = []
    for doc in sorted(new, key=lambda x: (FORMAT_RANK.get(x["path"].suffix, 9), x["path"].name)):
        ws, name = wordset(doc["md"]), doc["path"].name
        overlaps = [(len(ws & k["words"]) / len(ws), len(ws & k["words"]) / len(k["words"]), k) for k in kept]
        same = next((k for a, b, k in overlaps if a >= DOC_DUP and b >= DOC_DUP), None)
        if same:
            manifest["files"][doc["sha"]] = {"name": name, "status": "duplicate", "of": same["name"]}
            report["files"].append((name, f"skipped: same content as {same['name']} ({same['doc_id']})"))
            continue
        note = "".join(f"; {a:.0%} of its words also appear in {k['name']} (subset? its repeated chunks are handled below)"
                       for a, b, k in overlaps if a >= DOC_DUP)
        doc["doc_id"] = f"D{next_doc:02d}"
        next_doc += 1
        kept.append({"name": name, "doc_id": doc["doc_id"], "words": ws})
        docs.append((doc, note))

    all_docs = {i["name"]: (d / "extracted" / f"{i['doc_id']}.md").read_text()
                for i in manifest["files"].values() if i["status"] == "ingested"}
    all_docs.update({doc["path"].name: doc["md"] for doc, _ in docs})
    report["single"] = single_source_claims({doc["path"].name: doc["md"] for doc, _ in docs}, all_docs)

    # 3. Chunk each kept file by its headings.
    chunks = []
    for doc, note in docs:
        n_before, broken_tables, carry = len(chunks), False, ""
        for path, lines in sections(doc["md"], doc["path"].stem):
            headings = [re.sub(r"^[\d.\s]+", "", h).strip() for h in path]
            if headings[-1].lower() in SKIP_SECTIONS:
                report["dropped"].append(f"{doc['path'].name} § {path[-1]}")
                continue
            texts, broken = chunk_section(lines)
            broken_tables |= broken
            # A short intro before sub-headings ("The company maintains several facilities.") would become a
            # near-empty chunk that wins searches on its heading alone, so it joins the next chunk instead.
            if len(texts) == 1 and words(texts[0]) < MIN_WORDS and "|" not in texts[0]:
                carry += texts[0] + "\n"
                continue
            if carry and texts:
                texts[0], carry = carry + texts[0], ""
            # Short label = the last two headings without the document title, which is shared by every
            # chunk of the file and drowns short chunks (retrieval 70% -> 77% on the first benchmark dataset).
            topic = " > ".join(headings[1:][-2:]) or headings[0]
            for text in texts:
                chunks.append({"doc": doc, "topic": topic, "section": path[-1], "body": text,
                               "id": f"{doc['doc_id']}-{len(chunks) - n_before + 1:03d}"})
        if carry:  # short text at the very end of the file: nothing left to merge into, keep it as is
            topic = " > ".join(headings[1:][-2:]) or headings[0]
            chunks.append({"doc": doc, "topic": topic, "section": path[-1], "body": carry.strip(),
                           "id": f"{doc['doc_id']}-{len(chunks) - n_before + 1:03d}"})
        doc["n_chunks"], doc["note"], doc["broken"] = len(chunks) - n_before, note, broken_tables

    # 4. Chunk-level checks, always between different documents (sections of one document are distinct
    #    by design). Pool = chunks already built into the store + chunks accepted earlier in this batch.
    store = Store()
    built = store.col.get(include=["documents", "metadatas", "embeddings"])
    # This dataset's built chunks count only if their file is still in the manifest; otherwise they are
    # leftovers of a reset dataset and would wrongly mark fresh chunks as duplicates.
    live = {i["doc_id"] for i in manifest["files"].values() if i["status"] == "ingested"} - {doc["doc_id"] for doc, _ in docs}
    pool = [{"label": f"{m['dataset']}:{m['id']}", "dataset": m["dataset"], "doc": m["id"].split("-")[0],
             "text": t, "words": content_words(t), "nums": numbers(t), "vec": np.array(e)}
            for t, m, e in zip(built["documents"], built["metadatas"], built["embeddings"])
            if m["dataset"] != dataset or m["id"].split("-")[0] in live]
    texts = [f"{c['topic']}. {' '.join(c['body'].split())}" for c in chunks]
    vecs = store.embed(texts)
    accepted = []
    for i, c in enumerate(chunks):
        me = {"label": c["id"], "dataset": dataset, "doc": c["doc"]["doc_id"], "text": texts[i],
              "words": content_words(c["body"]), "nums": numbers(c["body"]), "vec": np.array(vecs[i])}
        others = [p for p in pool if not (p["dataset"] == dataset and p["doc"] == me["doc"])]
        overlap = lambda p: len(me["words"] & p["words"]) / max(1, min(len(me["words"]), len(p["words"])))
        same_ds = max(((overlap(p), p) for p in others if p["dataset"] == dataset), default=(0, None), key=lambda x: x[0])
        p = same_ds[1]
        if same_ds[0] >= CHUNK_DUP and len(me["words"] & p["words"]) / len(me["words"] | p["words"]) >= CHUNK_DUP_JACCARD:
            report["skipped"].append((c["id"], same_ds[1]["label"], same_ds[0], texts[i]))
            c["doc"]["n_chunks"] -= 1
            continue
        if same_ds[0] >= CHUNK_MAYBE:  # includes high-overlap pairs of very different size
            report["review"].append((c["id"], same_ds[1]["label"], same_ds[0], texts[i], same_ds[1]["text"]))
        cross = max(((overlap(p), p) for p in others if p["dataset"] != dataset), default=(0, None), key=lambda x: x[0])
        if cross[0] >= CHUNK_DUP:
            report["cross"].append((c["id"], cross[1]["label"], cross[0]))
        pool.append(me)
        accepted.append(i)

    # 5. Possible conflicts, once everything is in the pool (checking while chunks are still being added
    #    misses pairs whose second half arrives later). A chunk is compared with its 3 closest chunks from
    #    other documents of this dataset; a pair is flagged when either side states a number the other lacks
    #    ("up to 3 mm" vs "maximum thickness not stated" is a conflict too).
    mine = [p for p in pool if p["dataset"] == dataset]
    seen, conflicts = set(), []
    for me in mine:
        if not me["nums"]:
            continue
        near = sorted(((float(me["vec"] @ p["vec"]), p) for p in mine if p["doc"] != me["doc"]), key=lambda x: -x[0])[:3]
        for sim, p in near:
            pair = tuple(sorted((me["label"], p["label"])))
            if sim >= CONFLICT_SIM and me["nums"] != p["nums"] and pair not in seen:
                seen.add(pair)
                conflicts.append((sim, me["label"], p["label"], sorted(me["nums"] - p["nums"])[:8],
                                  sorted(p["nums"] - me["nums"])[:8], me["text"], p["text"]))
    for doc, _ in docs:
        status = f"new {doc['doc_id']}: {doc['n_chunks']} chunks{doc['note']}"
        if doc["broken"]:
            status += "; WARNING: a table looks broken after PDF extraction, check those chunks"
        path = doc["path"]
        read = len(scanned_pages(pymupdf.open(path))) if path.suffix == ".pdf" else path.suffix.lower() in IMAGES
        if read:
            status += f"; read by OCR ({read} page{'s' if read > 1 else ''}): check names and numbers in its chunks"
        report["files"].append((path.name, status))
    report["conflicts"] = sorted(conflicts, reverse=True)[:CONFLICTS_SHOWN]
    report["n_conflicts"] = len(conflicts)

    # 6. Write: append draft chunks, save extracted text and manifest, write the report.
    out = d / "chunks.md"
    if not out.exists():
        out.write_text(f"# {dataset} chunks: review copy\n\nEdit or delete chunks, then run `python build.py {dataset}`.\n"
                       "Format: `## <id> | <topic>`, a `sources:` line, then the text.\n"
                       "To mark a doubtful claim, put the note right after it: `up to 3 mm (unverified: ...)`.\n")
    with out.open("a") as f:
        for i in accepted:
            c = chunks[i]
            f.write(f"\n## {c['id']} | {c['topic']}\nsources: {c['doc']['path'].name} § {c['section']}\n\n{c['body']}\n")
    for doc, _ in docs:
        (d / "extracted" / f"{doc['doc_id']}.md").write_text(doc["md"])
        manifest["files"][doc["sha"]] = {"name": doc["path"].name, "status": "ingested",
                                         "doc_id": doc["doc_id"], "chunks": doc["n_chunks"]}
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    write_report(d, dataset, report, len(accepted))


def write_report(d, dataset, r, n_added):
    short = lambda t: " ".join(t.split())[:220]
    lines = [f"# Ingest report: {dataset}", f"{datetime.now():%Y-%m-%d %H:%M}. {n_added} chunks appended to chunks.md.", "",
             "## Files"] + [f"- **{name}**: {status}" for name, status in r["files"]]
    if not r["files"]:
        lines.append("- no new files")
    if r["dropped"]:
        lines += ["", "## Sections dropped (reference lists)"] + [f"- {s}" for s in r["dropped"]]
    lines += ["", f"## Duplicate chunks skipped ({len(r['skipped'])}; >= {CHUNK_DUP:.0%} of words shared with another document)"]
    lines += [f"- {cid} = {other} ({s:.0%}): {short(t)}" for cid, other, s, t in r["skipped"]]
    lines += ["", f"## Possible duplicates to review ({len(r['review'])}; {CHUNK_MAYBE:.0%}-{CHUNK_DUP:.0%} shared words)",
              "Delete one if they say the same thing, or merge them."]
    for cid, other, s, t, ot in r["review"]:
        lines += [f"- **{cid}** vs **{other}** ({s:.0%})", f"  - {cid}: {short(t)}", f"  - {other}: {short(ot)}"]
    lines += ["", f"## Possible conflicts to review (top {len(r['conflicts'])} of {r['n_conflicts']})",
              "Same topic in two documents with different numbers. Often one is just more detailed. When they truly "
              "disagree, delete the wrong claim, or write the note right after it, e.g. `7 assembly lines (unverified: "
              "only this source mentions them)`. A general warning at the top of the chunk is ignored by the model."]
    for sim, cid, other, mine, theirs, t, ot in r["conflicts"]:
        lines += [f"- **{cid}** vs **{other}** (similarity {sim:.2f}); only in {cid}: {mine}; only in {other}: {theirs}",
                  f"  - {cid}: {short(t)}", f"  - {other}: {short(ot)}"]
    n_single = sum(map(len, r["single"].values()))
    lines += ["", f"## Facts stated by only one document ({n_single})",
              "Normal for many facts (names, procedures). Look for big or marketing-style claims; if one is doubtful, "
              "add `(unverified: ...)` right after it in chunks.md. A file with many entries here deserves the closest look."]
    for name, claims in r["single"].items():
        lines += [f"- **{name}** ({len(claims)})"] + [f"  - {short(c)[:140]}" for c in claims]
    if r["cross"]:
        lines += ["", f"## Also present in other datasets ({len(r['cross'])}, kept in both)"]
        lines += [f"- {cid} ~ {other} ({s:.0%})" for cid, other, s in r["cross"]]
    lines += ["", f"Next: review chunks.md (and the items above), then run `python build.py {dataset}`."]
    (d / "ingest_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(r["files"])]))
    print(f"skipped {len(r['skipped'])} duplicate chunks, {len(r['review'])} possible duplicates, "
          f"{r['n_conflicts']} possible conflicts, {sum(map(len, r['single'].values()))} single-source facts, {len(r['cross'])} in other datasets -> {d / 'ingest_report.md'}")


if __name__ == "__main__":
    main(sys.argv[1])
