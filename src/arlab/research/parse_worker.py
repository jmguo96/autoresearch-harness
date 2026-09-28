"""Text-only PDF parser child. Memory/CPU limits reduce accidental parser exhaustion.

This is a constrained process, not an exploit-resistant OS sandbox. Scanned pages
remain explicitly unread; no OCR or language-model reconstruction is performed.
"""
from __future__ import annotations
import argparse
import resource
from pathlib import Path
from arlab.core.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    from pypdf import PdfReader
    reader = PdfReader(args.input, strict=False)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF is not supported")
    sections, page_map, size, unread = [], [], 0, []
    for index, page in enumerate(reader.pages[:100]):
        text = page.extract_text() or ""
        if not text.strip():
            unread.append(index + 1)
        section = f"[PDF page {index + 1}]\n" + text + "\n"
        remaining = max(0, 500_000 - size)
        section = section[:remaining]
        start_line = "".join(sections).count("\n") + 1
        sections.append(section)
        size += len(section)
        page_map.append({"page": index + 1, "start_line": start_line,
                         "end_line": start_line + section.count("\n")})
        if size >= 500_000:
            break
    write_json(args.output, {"text": "".join(sections), "scope": "pdf_text_only",
                            "pages_total": len(reader.pages), "pages_extracted": len(page_map),
                            "page_map": page_map, "unread_pages": unread, "visuals_verified": False,
                            "truncated": len(page_map) < len(reader.pages) or size >= 500_000,
                            "extraction_limit": "Text only; figures, scans and table alignment NOT verified"})


if __name__ == "__main__":
    main()
