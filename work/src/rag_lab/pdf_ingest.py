"""PDF ingestion with native-text-first extraction and optional PaddleOCR fallback."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    method: str


def needs_ocr(text: str, minimum_characters: int) -> bool:
    """Treat a page with too little visible text as an image-based scan."""
    return len("".join(text.split())) < minimum_characters


def extract_pages(
    pdf_path: Path,
    minimum_characters: int = 80,
    ocr: Callable[[bytes], str] | None = None,
    dpi: int = 200,
) -> list[ExtractedPage]:
    """Extract native PDF text first; invoke OCR only for insufficient pages."""
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("PDF取り込みにはPyMuPDFが必要です。`uv sync`を実行してください。") from error

    extracted: list[ExtractedPage] = []
    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            text = page.get_text("text", sort=True).strip()
            if not needs_ocr(text, minimum_characters):
                extracted.append(ExtractedPage(index + 1, text, "pymupdf"))
                continue
            if ocr is None:
                extracted.append(ExtractedPage(index + 1, text, "pymupdf_insufficient"))
                continue
            scale = dpi / 72
            image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
            extracted.append(ExtractedPage(index + 1, ocr(image).strip(), "paddleocr"))
    return extracted


@lru_cache(maxsize=1)
def _paddle_engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise RuntimeError(
            "PaddleOCRフォールバックには `uv sync --extra ocr` を実行してください。"
        ) from error
    # The v2-compatible API remains the most portable path for Colab runtimes.
    return PaddleOCR(use_angle_cls=True, lang="japan")


def paddle_ocr(image_bytes: bytes) -> str:
    """Recognize a rendered page with PaddleOCR's Japanese model."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("PaddleOCRの画像変換依存関係をインストールしてください。") from error
    image = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
    result = _paddle_engine().ocr(image, cls=True)
    return "\n".join(line[1][0] for block in result for line in (block or []))


def records_for_pdf(pdf_path: Path, minimum_characters: int, ocr_backend: str, dpi: int) -> list[dict[str, object]]:
    ocr = paddle_ocr if ocr_backend == "paddle" else None
    pages = extract_pages(pdf_path, minimum_characters=minimum_characters, ocr=ocr, dpi=dpi)
    document_id = pdf_path.stem
    return [
        {
            "id": f"{document_id}:p{page.page_number:04d}",
            "title": document_id,
            "text": page.text,
            "source_file": str(pdf_path),
            "page_number": page.page_number,
            "extraction_method": page.method,
        }
        for page in pages
        if page.text
    ]
