"""Sync a dataset's reviewed chunks.md into the vector store. Usage: python build.py <dataset>
chunks.md format: each chunk is `## <id> | <topic>`, then a `sources:` line, then the text."""
import re
import sys

from retrieval import DATASETS, Store


def parse_chunks(dataset):
    chunks = []
    for block in re.split(r"^## ", (DATASETS / dataset / "chunks.md").read_text(), flags=re.M)[1:]:
        head, rest = block.split("\n", 1)
        cid, topic = (s.strip() for s in head.split("|", 1))
        src_line, text = rest.strip().split("\n", 1)
        assert src_line.startswith("sources:"), f"{cid}: missing sources line"
        chunks.append({
            "id": cid,
            "topic": topic,
            "sources": [s.strip() for s in src_line[len("sources:"):].split(";")],
            # topic prefix makes short chunks self-describing for the embedder
            "text": f"{topic}. {' '.join(text.split())}",
        })
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    return chunks


if __name__ == "__main__":
    dataset = sys.argv[1]
    chunks = parse_chunks(dataset)
    changed, removed, total = Store().sync(dataset, chunks)
    words = sorted(len(c["text"].split()) for c in chunks)
    print(f"{dataset}: {total} chunks ({changed} embedded, {removed} removed); "
          f"words min {words[0]} median {words[len(words) // 2]} max {words[-1]}")
