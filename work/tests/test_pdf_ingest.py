import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_lab.pdf_ingest import ExtractedPage, extract_pages, needs_ocr


class PdfIngestTests(unittest.TestCase):
    def test_native_text_does_not_need_ocr(self):
        self.assertFalse(needs_ocr("日本銀行は物価の安定を目指して金融政策を運営します。" * 4, 80))

    def test_short_text_needs_ocr(self):
        self.assertTrue(needs_ocr("第3章", 80))

    def test_page_records_have_page_and_method(self):
        page = ExtractedPage(3, "本文", "pymupdf")
        self.assertEqual((page.page_number, page.method), (3, "pymupdf"))

    def test_extracts_native_pdf_text_without_ocr(self):
        import pymupdf

        with TemporaryDirectory() as directory:
            path = Path(directory) / "native.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "Native PDF text for retrieval. " * 5)
            pdf.save(path)
            pdf.close()
            extracted = extract_pages(path, minimum_characters=20)
        self.assertEqual(extracted[0].method, "pymupdf")
        self.assertIn("Native PDF text", extracted[0].text)
