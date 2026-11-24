"""PDF -> page text (silver layer).

Page numbers are 0-indexed to match FinanceBench's evidence_page_num, so a
retrieved chunk can be checked against the expert's evidence page exactly.
"""
from __future__ import annotations

import re

import pypdfium2 as pdfium

_WS = re.compile(r"[ \t]+")


def clean(text: str) -> str:
    text = text.replace("\r", "\n").replace("\x00", "")
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_pages(content: bytes) -> list[str]:
    pdf = pdfium.PdfDocument(content)
    try:
        return [clean(pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]
    finally:
        pdf.close()
