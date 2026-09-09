"""Simple PDF data extraction and chunking."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union
from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    doc_hash: str
    index: int
    page_start: int
    page_end: int
    text: str


@dataclass
class PDFData:
    filename: str
    file_hash: str
    page_count: int
    chunks: list[Chunk]


def load_pdf(source: Union[str, Path, bytes, BinaryIO], filename: str = "document.pdf", chunk_size: int = 1200) -> PDFData:
    """Read a PDF, compute SHA-256 hash, and split extracted text into chunks."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        filename = p.name
        raw = p.read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()

    doc_hash = hashlib.sha256(raw).hexdigest()
    reader = PdfReader(io.BytesIO(raw))
    page_count = len(reader.pages)

    # Extract text per page
    chunks: list[Chunk] = []
    chunk_idx = 0
    current_text = ""
    start_page = 1

    for page_num, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        # Clean hyphenation and excessive spaces
        txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        if not txt:
            continue

        if not current_text:
            start_page = page_num

        current_text = f"{current_text}\n\n{txt}".strip() if current_text else txt

        # If accumulated text reaches chunk_size, create chunk
        while len(current_text) >= chunk_size:
            split_pos = current_text.rfind("\n\n", 0, chunk_size)
            if split_pos == -1:
                split_pos = current_text.rfind(". ", 0, chunk_size)
            if split_pos == -1:
                split_pos = chunk_size

            chunk_text = current_text[:split_pos].strip()
            current_text = current_text[split_pos:].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{doc_hash[:16]}_{chunk_idx}",
                        doc_hash=doc_hash,
                        index=chunk_idx,
                        page_start=start_page,
                        page_end=page_num,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1
            start_page = page_num

    # Remaining text
    if current_text:
        chunks.append(
            Chunk(
                id=f"{doc_hash[:16]}_{chunk_idx}",
                doc_hash=doc_hash,
                index=chunk_idx,
                page_start=start_page,
                page_end=page_count,
                text=current_text,
            )
        )

    return PDFData(
        filename=filename,
        file_hash=doc_hash,
        page_count=page_count,
        chunks=chunks,
    )
