"""webfetch: URL in, readable text out. No real network — the fetch is stubbed;
the HTML→text pass, the scheme gate, error handling, and spill are exercised
directly."""

import urllib.error

from ahacode import tools
from ahacode.tools import webfetch


# --- the scheme gate: only http/https, checked before approval -------------------

def test_non_http_schemes_are_blocked():
    assert webfetch._check_scheme({"url": "file:///etc/passwd"}) is not None
    assert webfetch._check_scheme({"url": "ftp://host/x"}) is not None
    assert webfetch._check_scheme({"url": "https://example.com"}) is None
    assert webfetch._check_scheme({"url": "http://example.com"}) is None
    # the registered tool wires the check as its validate() (hard block, no prompt)
    assert tools.REGISTRY["webfetch"].validate({"url": "file:///x"}) is not None


# --- HTML → readable text --------------------------------------------------------

def test_html_becomes_text_dropping_noise_and_keeping_the_title():
    html = (
        "<html><head><title>My Page</title><style>.a{color:red}</style></head>"
        "<body><h1>Heading</h1><p>Para   with   spaces</p>"
        "<script>evil()</script><ul><li>one</li><li>two</li></ul></body></html>"
    )
    text = webfetch._html_to_text(html)
    assert text.startswith("# My Page")          # title kept as a heading
    assert "Heading" in text
    assert "Para with spaces" in text            # runs of whitespace collapsed
    assert "one" in text and "two" in text
    assert "evil()" not in text                  # <script> dropped
    assert "color:red" not in text               # <style> dropped


# --- execute(): strip HTML, pass other bodies through, surface errors ------------

def _stub_fetch(monkeypatch, ctype, body, truncated=False):
    monkeypatch.setattr(webfetch, "_fetch", lambda url, timeout: (ctype, body, truncated))


def test_html_response_is_reduced_to_text(monkeypatch):
    _stub_fetch(monkeypatch, "text/html", "<p>Hello <b>world</b></p>")
    assert webfetch._webfetch({"url": "https://x"}) == "Hello world"


def test_non_html_body_passes_through_untouched(monkeypatch):
    _stub_fetch(monkeypatch, "application/json", '{"a": 1}')
    assert webfetch._webfetch({"url": "https://x/api"}) == '{"a": 1}'


def test_format_html_returns_raw_markup(monkeypatch):
    _stub_fetch(monkeypatch, "text/html", "<p>raw</p>")
    assert webfetch._webfetch({"url": "https://x", "format": "html"}) == "<p>raw</p>"


def test_truncation_is_flagged(monkeypatch):
    _stub_fetch(monkeypatch, "text/html", "<p>partial</p>", truncated=True)
    out = webfetch._webfetch({"url": "https://x"})
    assert "partial" in out and "truncated at" in out


def test_http_error_comes_back_as_text_not_an_exception(monkeypatch):
    def boom(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
    monkeypatch.setattr(webfetch, "_fetch", boom)
    out = webfetch._webfetch({"url": "https://x/missing"})
    assert out == "HTTP 404 Not Found for https://x/missing"


def test_network_failure_comes_back_as_text(monkeypatch):
    def boom(url, timeout):
        raise urllib.error.URLError("name or service not known")
    monkeypatch.setattr(webfetch, "_fetch", boom)
    out = webfetch._webfetch({"url": "https://nope.invalid"})
    assert out.startswith("could not fetch https://nope.invalid")


# --- oversized pages spill, like bash --------------------------------------------

def test_a_big_page_spills_to_a_file(monkeypatch, tmp_path):
    saved = tmp_path / "webfetch-1.txt"
    monkeypatch.setattr(webfetch.spill, "write", lambda text, prefix="out": saved)
    monkeypatch.setattr(webfetch.spill, "relative", lambda p: "sessions/tool-output/webfetch-1.txt")
    _stub_fetch(monkeypatch, "text/plain", "x" * (webfetch._SPILL_OVER_CHARS + 100))
    out = webfetch._webfetch({"url": "https://x/big"})
    assert "saved in full to sessions/tool-output/webfetch-1.txt" in out
    assert len(out) < webfetch._SPILL_OVER_CHARS + 100  # only a preview came back


# --- the tool contract -----------------------------------------------------------

def test_tool_contract():
    t = tools.REGISTRY["webfetch"]
    assert t.requires_approval is True and t.parallelizable is True
    assert t.validate is not None
    assert t.parameters["required"] == ["url"]
