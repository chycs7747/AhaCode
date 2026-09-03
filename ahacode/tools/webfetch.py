"""webfetch: fetch a URL over http(s) and return its readable text.

Approval-gated, with a scheme check first so no other protocol (file://, ftp://)
can read local resources. HTML is reduced to text with the standard library;
JSON and plain text pass through. An oversized page spills to a file like bash.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

from ahacode.tools import spill
from ahacode.tools.base import Tool, clamp_timeout

# A real User-Agent: many servers answer the default urllib agent with 403.
_UA = "AhaCode/1.0 (+https://github.com/chycs7747/AhaCode)"

# Network timeout for one fetch, unrelated to the LLM gateway timeout (minutes).
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120

# Read at most this many bytes off the wire; the body is truncated past it.
_MAX_BYTES = 5 * 1024 * 1024


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML: drop script/style/head, turn block-level tags
    into line breaks, and keep the <title> as a heading."""

    _SKIP = {"script", "style", "head", "noscript", "svg", "template"}
    _BLOCK = {
        "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
        "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._title: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:  # <title> lives inside the skipped <head>, so check it first
            self._title.append(data)
        elif self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        body = _collapse("".join(self._parts))
        title = " ".join("".join(self._title).split())
        return f"# {title}\n\n{body}".strip() if title else body


def _collapse(raw: str) -> str:
    """Collapse runs of whitespace and blank lines, keeping one blank between blocks."""
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in raw.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def _check_scheme(args: dict) -> str | None:
    """The block reason for a non-http(s) URL, or None."""
    url = str(args.get("url", "")).strip().lower()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "url must start with http:// or https://"
    return None


def _fetch(url: str, timeout: int) -> tuple[str, str, bool]:
    """GET the URL.

    Args:
        url: The address.
        timeout: Seconds to wait.

    Returns:
        (content type, the body decoded per its charset with replacement, whether
        the body was cut at _MAX_BYTES).
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (scheme gated)
        ctype = resp.headers.get_content_type()
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read(_MAX_BYTES + 1)
    truncated = len(raw) > _MAX_BYTES
    try:
        body = raw[:_MAX_BYTES].decode(charset, errors="replace")
    except LookupError:  # an unknown charset name
        body = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
    return ctype, body, truncated


def _webfetch(args: dict) -> str:
    """Fetch the page and hand back its text; a failed fetch comes back as text too."""
    url = str(args.get("url", "")).strip()
    fmt = args.get("format", "text")
    seconds = clamp_timeout(args.get("timeout"), _DEFAULT_TIMEOUT, _MAX_TIMEOUT)
    try:
        ctype, body, truncated = _fetch(url, seconds)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason} for {url}"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return f"could not fetch {url}: {type(exc).__name__}: {exc}"

    if fmt != "html" and "html" in ctype:
        text = _html_to_text(body)
    else:
        text = body
    if truncated:
        text += f"\n[truncated at {_MAX_BYTES // 1024}KB — fetch a more specific URL for the rest]"
    return spill.preview(text.strip() or "(empty response)", prefix="webfetch", noun="page")


def _html_to_text(html: str) -> str:
    """The readable text of an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


WEBFETCH = Tool(
    name="webfetch",
    description=(
        "Fetch a web page over http/https and return its readable text (HTML is "
        "reduced to text; JSON and plain text pass through). Use it to read docs, "
        "articles, or an API response at a known URL. Give a specific URL — this "
        "fetches one page, it does not search the web."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http/https URL to fetch"},
            "format": {
                "type": "string",
                "enum": ["text", "html"],
                "description": "text (default): readable text; html: the raw HTML",
            },
            "timeout": {
                "type": "integer",
                "description": f"Seconds to allow before giving up (max {_MAX_TIMEOUT})",
            },
        },
        "required": ["url"],
    },
    execute=_webfetch,
    requires_approval=True,
    validate=_check_scheme,
    parallelizable=True,  # a read over the network: no local side effect
)
