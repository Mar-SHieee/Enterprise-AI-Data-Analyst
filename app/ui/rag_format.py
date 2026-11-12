"""Turn the agent's raw 'retrieved documents' text into clean, readable pieces.

Pure Python (no Streamlit) so it is easy to test.
"""

from __future__ import annotations

import re

_SOURCE = re.compile(
    r"\[S(\d+)\]\s*source:\s*([^\s#]+)(?:#(.+?))?\s*\(similarity=([\d.]+)\)"
)
_TAGS = re.compile(r"</?retrieved_documents[^>]*>", re.IGNORECASE)
_INTRO = re.compile(
    r"Relevant documentation:.*?(?=<retrieved_documents|\Z)", re.DOTALL | re.IGNORECASE
)
_RULE = re.compile(r"(?m)^\s*-{3,}\s*$")
_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def parse_rag_answer(content: str):
    """Return (intro_text, sources). `sources` is empty if this is not a RAG dump."""
    matches = list(_SOURCE.finditer(content))
    if not matches:
        return content, []

    intro = content[: matches[0].start()]
    intro = _TAGS.sub("", _INTRO.sub("", intro)).strip()

    sources = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        path = (m.group(3) or "").strip()
        parts = [p.strip() for p in path.split(">") if p.strip()]
        sources.append(
            {
                "file": m.group(2),
                "title": parts[-1] if parts else m.group(2),
                "crumb": " › ".join(parts[:-1]),
                "match": round(float(m.group(4)) * 100),
                "body": _clean_body(content[m.end() : end], path),
            }
        )
    return intro, sources


def _clean_body(body: str, path: str) -> str:
    body = _TAGS.sub("", body)
    body = _RULE.sub("", body).strip()

    # the first line usually just repeats the heading path
    lines = body.split("\n")
    if lines and lines[0].strip() == path:
        body = "\n".join(lines[1:]).strip()

    # chunk boundaries can leave a dangling ** marker
    if body.count("**") % 2:
        first = body.find("**")
        looks_closing = (
            first > 0
            and not body[first - 1].isspace()
            and (first + 2 >= len(body) or body[first + 2].isspace())
        )
        if looks_closing:
            body = body[:first] + body[first + 2 :]
        else:
            last = body.rfind("**")
            body = body[:last] + body[last + 2 :]

    # chunks are cut at a fixed length — mark the cut
    if body and body[-1].isalnum():
        body += "…"
    return body


def split_blocks(body: str):
    """Split text into [("text", "", str) | ("code", lang, str)] in order."""
    blocks, pos = [], 0
    for m in _FENCE.finditer(body):
        text = body[pos : m.start()].strip()
        if text:
            blocks.append(("text", "", text))
        blocks.append(("code", m.group(1).lower(), m.group(2).strip("\n")))
        pos = m.end()
    tail = body[pos:].strip()
    if tail:
        blocks.append(("text", "", tail))
    return blocks
