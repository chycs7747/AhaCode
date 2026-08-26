"""webfetch: fetch a URL over HTTP(S) and return its readable text.

requires_approval=True — it reaches an arbitrary address on the network, so the
user confirms the URL first. A scheme check hard-blocks anything that is not
http/https *before* the prompt, so a `file://` or `gopher://` can never be used
to read a local file or an internal service by another protocol.

The HTML is reduced to text with the standard-library parser (no dependency):
script/style/head noise is dropped, block tags become line breaks, and runs of
whitespace collapse. Non-HTML bodies (JSON, plain text) pass through unchanged.
An oversized page spills to a file exactly like bash, so a long article never
floods the context — the model reads or greps the saved file for the part it needs.

Shape after Kilo Code's webfetch tool (URL in, readable text out, size cap +
timeout, approval-gated); implemented on urllib/html.parser so AhaCode stays
dependency-free.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

from ahacode.text import elide, line_count
from ahacode.tools import spill
from ahacode.tools.base import Tool

# A real User-Agent: many servers answer the default urllib agent with 403.
_UA = "AhaCode/1.0 (+https://github.com/chycs7747/AhaCode)"

# Network timeout for one fetch — its own short cap, unrelated to the LLM gateway
# timeout (which is minutes). A call may raise it up to the ceiling for a slow host.
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120

# Read at most this many bytes off the wire, so a giant download cannot exhaust
# memory; the body is truncated (with a note) past it.
_MAX_BYTES = 5 * 1024 * 1024

# Same spill discipline as bash: past _SPILL_OVER_CHARS the full text goes to a
# file and only a preview returns; if the file cannot be written, truncate instead.
_SPILL_OVER_CHARS = 4_000
_PREVIEW_CHARS = 2_000
_MAX_OUTPUT_CHARS = 30_000


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML: drop script/style/head, turn block-level
    tags into line breaks, and keep the <title> as a heading. Deliberately small —
    it is a readability pass, not a full renderer."""

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
        # <title> lives inside <head> (which is skipped), so check it first.
        if self._in_title:
            self._title.append(data)
        elif self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        body = _collapse("".join(self._parts))
        title = " ".join("".join(self._title).split())
        return f"# {title}\n\n{body}".strip() if title else body


def _collapse(raw: str) -> str:
    """Collapse intra-line whitespace and blank-line runs, so the extracted text
    reads cleanly instead of carrying the source's indentation and empty lines."""
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in raw.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):  # keep a single blank between blocks, not runs
            out.append(ln)
    return "\n".join(out).strip()


def _check_scheme(args: dict) -> str | None:
    """Hard-block anything that is not http/https, before the approval prompt — so
    no other protocol (file://, ftp://, …) can be used to read local resources."""
    url = str(args.get("url", "")).strip().lower()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "url must start with http:// or https://"
    return None


def _resolve_timeout(requested) -> int:
    """Seconds this fetch may take: what it asked for, clamped, else the default."""
    if requested is None:
        return _DEFAULT_TIMEOUT
    try:
        return max(1, min(int(requested), _MAX_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


def _fetch(url: str, timeout: int) -> tuple[str, str, bool]:
    """GET the URL and return (content-type, decoded body, truncated?). Decoding
    follows the response's charset, falling back to utf-8 with replacement so a
    mislabelled page still yields text instead of raising."""
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
    url = str(args.get("url", "")).strip()
    fmt = args.get("format", "text")
    seconds = _resolve_timeout(args.get("timeout"))
    try:
        ctype, body, truncated = _fetch(url, seconds)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason} for {url}"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        # a broken tool must not crash the loop; hand the reason back as text
        return f"could not fetch {url}: {type(exc).__name__}: {exc}"

    # Only strip HTML when the body actually is HTML and the caller wants text;
    # JSON / plain text / a raw-html request pass through as-is.
    if fmt != "html" and "html" in ctype:
        text = _html_to_text(body)
    else:
        text = body
    if truncated:
        text += f"\n[truncated at {_MAX_BYTES // 1024}KB — fetch a more specific URL for the rest]"
    return _finish(text)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _finish(text: str) -> str:
    """Spill if oversized (like bash), else return the text; never empty."""
    text = text.strip() or "(empty response)"
    if len(text) <= _SPILL_OVER_CHARS:
        return text
    path = spill.write(text, prefix="webfetch")
    if path is None:  # nowhere to spill — degrade to truncation
        return elide(text, _MAX_OUTPUT_CHARS)
    where = spill.relative(path)
    header = (
        f"[page was {len(text):,} chars / {line_count(text):,} lines — saved in full to {where}\n"
        f" read it with read(path=\"{where}\", offset=…, limit=…), "
        f"or search it with grep(pattern=…, path=\"{where}\")]\n"
    )
    return header + elide(text, _PREVIEW_CHARS)


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
    requires_approval=True,     # reaches an arbitrary network address — confirm first
    validate=_check_scheme,     # non-http(s) schemes are hard-blocked before that
    parallelizable=True,        # a read over the network — no local side effect, order-free
)
