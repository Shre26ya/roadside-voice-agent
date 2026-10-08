import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_docs: list[tuple[str, str]] = []
_index: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load(knowledge_dir: str = "knowledge") -> None:
    """Chunk every markdown file by heading and build a BM25 index."""
    global _docs, _index
    _docs = []
    for path in Path(knowledge_dir).glob("*.md"):
        sections = re.split(r"\n(?=## )", path.read_text())
        for section in sections:
            title = section.splitlines()[0].lstrip("# ").strip()
            _docs.append((title, section.strip()))
    _index = BM25Okapi([_tokenize(body) for _, body in _docs])


def search(query: str, k: int = 2) -> list[tuple[str, str]]:
    if _index is None:
        load()
    scores = _index.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, _docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:k] if score > 0]