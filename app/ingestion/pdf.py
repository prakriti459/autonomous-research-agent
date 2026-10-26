import json
from pathlib import Path
from typing import Any

from app.config import get_settings


def _chunk_page(chunk: Any) -> int | None:
    pages = []
    metadata = getattr(chunk, "meta", None)
    for item in getattr(metadata, "doc_items", []):
        for provenance in getattr(item, "prov", []):
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int) and page_number > 0:
                pages.append(page_number)
    return min(pages) if pages else None


def parse_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PDF ingestion dependencies are unavailable; install requirements.txt"
        ) from error

    settings = get_settings()
    try:
        document = DocumentConverter().convert(str(pdf_file)).document
    except Exception as error:
        raise RuntimeError(f"PDF parsing failed: {error}") from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=512,
            overlap_tokens=64,
        )
        chunks = list(chunker.chunk(document))
    except Exception as error:
        raise RuntimeError(f"PDF chunking failed: {error}") from error

    processed = []
    for index, chunk in enumerate(chunks):
        text = chunk.text.strip()
        if not text:
            continue
        processed.append(
            {
                "chunk_id": f"chunk_{index}",
                "text": text,
                "page": _chunk_page(chunk),
            }
        )
    if not processed:
        raise ValueError("The PDF did not contain any usable text")
    return processed


def ingest_pdf(pdf_path: str) -> list[dict[str, Any]]:
    chunks = parse_pdf(pdf_path)
    with Path("chunks.json").open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)
    return chunks


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")
    chunks = ingest_pdf(pdf_path)
    for chunk in chunks[:3]:
        print(f"{chunk['chunk_id']} (page {chunk['page']})")
        print(chunk["text"][:1000])
