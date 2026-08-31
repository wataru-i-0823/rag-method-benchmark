"""Convert local PDFs into an ignored JSONL corpus for the RAG lab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.pdf_ingest import records_for_pdf


parser = argparse.ArgumentParser(description="Extract PDF pages with PyMuPDF and optional PaddleOCR fallback")
parser.add_argument("pdfs", nargs="*", type=Path, help="Input PDF files")
parser.add_argument("--input-dir", type=Path, help="Directory containing PDF files (searched recursively)")
parser.add_argument("--output", type=Path, default=Path("data/processed/pdf_corpus.jsonl"))
parser.add_argument("--ocr-backend", choices=["none", "paddle"], default="none")
parser.add_argument("--minimum-characters", type=int, default=80, help="OCR pages with fewer non-whitespace characters")
parser.add_argument("--dpi", type=int, default=200, help="Rendering resolution for OCR pages")
args = parser.parse_args()

if args.minimum_characters < 0 or args.dpi <= 0:
    parser.error("--minimum-characters must be non-negative and --dpi must be positive")
pdfs = list(args.pdfs)
if args.input_dir:
    pdfs.extend(args.input_dir.rglob("*.pdf"))
pdfs = sorted({path.resolve() for path in pdfs})
if not pdfs:
    parser.error("PDFを指定するか、--input-dirを指定してください")
missing = [str(path) for path in pdfs if not path.is_file()]
if missing:
    parser.error(f"見つからないPDF: {', '.join(missing)}")

records: list[dict[str, object]] = []
for pdf in pdfs:
    records.extend(records_for_pdf(pdf, args.minimum_characters, args.ocr_backend, args.dpi))

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
summary = {
    "pdfs": len(pdfs),
    "documents": len(records),
    "native_pages": sum(row["extraction_method"] == "pymupdf" for row in records),
    "ocr_pages": sum(row["extraction_method"] == "paddleocr" for row in records),
    "output": str(args.output),
}
print(json.dumps(summary, ensure_ascii=False))
