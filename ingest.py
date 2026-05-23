from pathlib import Path
from typing import List, Dict
import json

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    from transformers import AutoTokenizer

except ImportError as e:
    raise ImportError(
        f"Import failed: {e}"
    )


# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------




# ------------------------------------------------------
# INGESTION FUNCTION
# ------------------------------------------------------

def ingest_pdf(pdf_path: str) -> List[Dict]:

    pdf_file = Path(pdf_path)

    # --------------------------------------------------
    # CHECK PDF EXISTS
    # --------------------------------------------------

    if not pdf_file.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print(f"\n[INFO] Loading PDF: {pdf_path}")

    # --------------------------------------------------
    # STEP 1: Parse PDF
    # --------------------------------------------------

    try:

        converter = DocumentConverter()

        result = converter.convert(
            str(pdf_file)
        )

        document = result.document

        print(
            "[SUCCESS] PDF parsed successfully!"
        )

    except Exception as e:

        raise RuntimeError(
            f"PDF parsing failed:\n{e}"
        )

    # --------------------------------------------------
    # STEP 2: Load Tokenizer
    # --------------------------------------------------

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(
            "[SUCCESS] Tokenizer loaded!"
        )

    except Exception as e:

        raise RuntimeError(
            f"Tokenizer loading failed:\n{e}"
        )

    # --------------------------------------------------
    # STEP 3: Hybrid Semantic Chunking
    # --------------------------------------------------

    try:

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=512,
            overlap_tokens=64,
        )

        chunks = list(
            chunker.chunk(document)
        )

        print(
            f"[SUCCESS] Generated "
            f"{len(chunks)} chunks!"
        )

    except Exception as e:

        raise RuntimeError(
            f"Chunking failed:\n{e}"
        )

    # --------------------------------------------------
    # STEP 4: Clean Chunks
    # --------------------------------------------------

    processed_chunks = []

    for idx, chunk in enumerate(chunks):

        try:

            chunk_text = chunk.text.strip()

            if not chunk_text:
                continue

            processed_chunks.append(
                {
                    "chunk_id": f"chunk_{idx}",
                    "text": chunk_text,
                }
            )

        except Exception as e:

            print(
                f"[WARNING] Failed chunk "
                f"{idx}: {e}"
            )

    print(
        f"[SUCCESS] Final usable chunks: "
        f"{len(processed_chunks)}"
    )

    # --------------------------------------------------
    # STEP 5: Save Chunks to JSON
    # --------------------------------------------------

    try:

        with open(
            "chunks.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                processed_chunks,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "\n[SUCCESS] Chunks saved "
            "to chunks.json"
        )

    except Exception as e:

        print(
            f"\n[WARNING] Failed to "
            f"save JSON:\n{e}"
        )

    return processed_chunks


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":

    pdf_path = input(
    "\nEnter PDF path: "
)
    chunks = ingest_pdf(pdf_path)

    print(
        "\n========== SAMPLE CHUNKS ==========\n"
    )

    for chunk in chunks[:3]:

        print(chunk["chunk_id"])

        print("\n")

        print(
            chunk["text"][:1000]
        )

        print(
            "\n" + "=" * 80 + "\n"
        )