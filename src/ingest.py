#!/usr/bin/env python3
"""
MANTIS: PDF Ingestion Pipeline
=========================================
Libraries: pymupdf (fitz), json only.
"""

import json
import os
import re
import sys
try:
    import fitz  # PyMuPDF
except ImportError:
    print("[FATAL] 'pymupdf' library not found.")
    sys.exit(1)

from security import save_encrypted_json


# --- Configuration ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANUALS_FOLDER = os.path.join(_SCRIPT_DIR, "..", "Manuals")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "..", "data", "knowledge_base.json")

# Platform tags
PLATFORM_PATTERNS = {
    "AH-1": re.compile(r"AH[\-_]?1", re.IGNORECASE),
    "RC-12": re.compile(r"RC[\-_]?12", re.IGNORECASE),
    "RD-12": re.compile(r"RD[\-_]?12", re.IGNORECASE),
    "C-12": re.compile(r"\bC[\-_]?12", re.IGNORECASE),
    "UH-1": re.compile(r"UH[\-_]?1", re.IGNORECASE),
    "EH-1": re.compile(r"EH[\-_]?1", re.IGNORECASE),
    "OH-58": re.compile(r"OH[\-_]?58", re.IGNORECASE),
    "UH-60": re.compile(r"UH[\-_]?60", re.IGNORECASE),
    "CH-47": re.compile(r"CH[\-_]?47", re.IGNORECASE),
    "M1": re.compile(r"\bM1\b", re.IGNORECASE),
    "M2": re.compile(r"\bM2\b", re.IGNORECASE),
    "HMMWV": re.compile(r"HMMWV|HUMVEE", re.IGNORECASE),
}


def clean_text(text: str) -> str:
    """
    Clean extracted text to reduce noise and memory footprint.
    - Collapse multiple whitespace/newlines into single spaces.
    - Strip leading/trailing whitespace.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_platform(filename: str) -> str:
    """
    Auto-detect platform from filename using predefined patterns.
    Returns the platform tag or 'UNKNOWN' if no match.
    """
    for platform, pattern in PLATFORM_PATTERNS.items():
        if pattern.search(filename):
            return platform
    return "UNKNOWN"


def extract_pdf_pages(pdf_path: str) -> list:
    """
    Extract text using a sliding-window chunking approach.
    Chunk size: 500 words, Overlap: 100 words.
    Returns list of (starting_page_number, text) tuples.
    """
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        words_with_pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            cleaned = clean_text(text)
            if cleaned:
                for word in cleaned.split():
                    words_with_pages.append((page_num + 1, word))
        doc.close()
        
        chunk_size = 150
        overlap = 50
        step = chunk_size - overlap
        
        for i in range(0, len(words_with_pages), step):
            window = words_with_pages[i:i + chunk_size]
            if not window:
                break
            start_page = window[0][0]
            chunk_text = " ".join([w[1] for w in window])
            chunks.append((start_page, chunk_text))
            
    except Exception as e:
        print(f"  [ERROR] Failed to process {pdf_path}: {e}")
    return chunks


def ingest_manuals(folder: str) -> list:
    """
    Scan folder for PDFs and build knowledge base chunks.
    Each chunk = one page from a PDF.
    """
    knowledge_base = []
    chunk_id = 0

    if not os.path.isdir(folder):
        print(f"[ERROR] Manuals folder '{folder}' not found!")
        return knowledge_base

    # Get all PDF files
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    print(f"[INFO] Found {len(pdf_files)} PDF(s) in '{folder}/'")

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(folder, pdf_file)
        platform = detect_platform(pdf_file)
        print(f"  Processing: {pdf_file} [Platform: {platform}]")

        pages = extract_pdf_pages(pdf_path)

        for page_num, text in pages:
            chunk_id += 1
            chunk = {
                "id": f"doc{chunk_id}_p{page_num}",
                "text": text,
                "source": pdf_file,
                "page": page_num,
                "platform": platform,
            }
            knowledge_base.append(chunk)

    return knowledge_base


def save_knowledge_base(kb: list, output_path: str) -> None:
    """
    Save encrypted knowledge base to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_encrypted_json(kb, output_path)
    print(f"[INFO] Saved {len(kb)} chunks to '{output_path}'")


def main():
    print("=" * 60)
    print("  MANTIS: PDF Ingestion Pipeline")
    print("=" * 60)
    print()

    # Ingest all PDFs
    knowledge_base = ingest_manuals(MANUALS_FOLDER)

    if not knowledge_base:
        print("[WARN] No content extracted. Check your manuals/ folder.")
        return

    # Save to JSON
    save_knowledge_base(knowledge_base, OUTPUT_FILE)

    # Platforms
    platforms = {}
    for chunk in knowledge_base:
        p = chunk["platform"]
        platforms[p] = platforms.get(p, 0) + 1

    print()
    print("[INFO] Platform Distribution:")
    for platform, count in sorted(platforms.items()):
        print(f"  - {platform}: {count} chunks")

    kb_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n[INFO] Knowledge base size: {kb_size:.1f} KB")
    print("[DONE] Ingestion complete!")


if __name__ == "__main__":
    main()
