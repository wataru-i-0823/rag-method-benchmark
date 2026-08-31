"""Download configured public pages into ignored local data folders."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.request import Request, urlopen


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored = max(0, self.ignored - 1)

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text and not self.ignored:
            self.parts.append(text)


def fetch(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "rag-method-benchmark/0.1 (educational retrieval evaluation)"})
    with urlopen(request, timeout=30) as response:
        return response.read(), response.headers.get_content_charset() or "utf-8"


parser = argparse.ArgumentParser(description="Download configured public sources without committing their contents")
parser.add_argument("--manifest", default="config/public_sources/government_public_information.json")
parser.add_argument("--raw-dir", default="data/raw/fsa-boj-public-information")
parser.add_argument("--output", default="data/processed/fsa_boj_public_information.jsonl")
args = parser.parse_args()

manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
raw_dir = Path(args.raw_dir)
raw_dir.mkdir(parents=True, exist_ok=True)
retrieved_at = datetime.now(UTC).isoformat()
rows = []
for source in manifest["sources"]:
    body, charset = fetch(source["url"])
    (raw_dir / f"{source['id']}.html").write_bytes(body)
    extractor = TextExtractor()
    extractor.feed(body.decode(charset, errors="replace"))
    rows.append({"id": source["id"], "title": source["title"], "text": "\n".join(extractor.parts), "source_url": source["url"], "retrieved_at": retrieved_at})

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
print(json.dumps({"dataset": manifest["dataset"], "documents": len(rows), "output": str(output), "retrieved_at": retrieved_at}, ensure_ascii=False))
