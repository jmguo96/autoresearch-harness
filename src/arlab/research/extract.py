"""Conservative text extraction, not a claim of visual/figure understanding."""
from __future__ import annotations
from html.parser import HTMLParser
from urllib.parse import urljoin
import re


class ReadableHTML(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.ignore_depth = 0
        self.title_depth = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.links: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "noscript", "svg"):
            self.ignore_depth += 1
        if tag == "title":
            self.title_depth = True
        if self.ignore_depth:
            return
        if tag in ("p", "div", "section", "li", "tr", "br", "pre", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")
        if tag == "a" and attrs.get("href") and len(self.links) < 200:
            self.links.append({"url": urljoin(self.base_url, attrs["href"])})
        if tag == "math" and attrs.get("alttext"):
            self.parts.append(" " + attrs["alttext"] + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.ignore_depth:
            self.ignore_depth -= 1
        if tag == "title":
            self.title_depth = False
        if not self.ignore_depth and tag in ("p", "div", "section", "li", "tr", "pre", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.ignore_depth:
            return
        if self.title_depth:
            self.title_parts.append(data)
        self.parts.append(data)

    def result(self) -> dict:
        text = "\n".join(re.sub(r"[ \t\r\f\v]+", " ", s).strip()
                         for s in "".join(self.parts).splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"text": text, "title": " ".join(self.title_parts).strip(), "links": self.links,
                "scope": "html_text", "visuals_verified": False,
                "extraction_limit": "No JavaScript, OCR, rendered figures, or table-layout verification"}


def html_text(text: str, url: str = "") -> dict:
    parser = ReadableHTML(url)
    parser.feed(text)
    parser.close()
    return parser.result()
