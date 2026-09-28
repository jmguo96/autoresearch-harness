"""Lossless bounded text windows. A very long line also has a continuation cursor."""
from __future__ import annotations


def text_page(text: str, start_line: int = 1, max_lines: int = 60,
              max_characters: int = 5000, offset: int = 0) -> dict:
    lines = text.splitlines(keepends=True)
    if start_line < 1 or offset < 0 or max_lines < 1 or max_characters < 1:
        raise ValueError("Invalid page bounds")
    if start_line > len(lines) + 1:
        raise ValueError("start_line exceeds document")
    if start_line == len(lines) + 1:
        if offset:
            raise ValueError("Offset after end of document")
        return {"text": "", "start_line": start_line, "end_line": start_line - 1,
                "total_lines": len(lines), "next": None, "truncated": False}
    if offset > len(lines[start_line - 1]):
        raise ValueError("Offset exceeds line length")
    index, column, size = start_line - 1, offset, 0
    output = []
    while index < len(lines) and index < start_line - 1 + max_lines and size < max_characters:
        value = lines[index][column:]
        take = min(len(value), max_characters - size)
        output.append(value[:take])
        size += take
        column += take
        if column == len(lines[index]):
            index += 1
            column = 0
        else:
            break
    more = index < len(lines)
    return {"text": "".join(output), "start_line": start_line,
            "end_line": index + (1 if column else 0), "start_offset": offset,
            "total_lines": len(lines), "next": {"start_line": index + 1, "offset": column} if more else None,
            "truncated": more}


def list_page(rows: list, offset: int = 0, limit: int = 10) -> dict:
    end = min(len(rows), offset + limit)
    return {"items": rows[offset:end], "total": len(rows),
            "next_offset": end if end < len(rows) else None}
