"""GitHub issues: fetch them, and parse the markdown the TEGE editor writes.

The editor's FeedbackSystem files two shapes of issue:

  [Bug] <title>      labels: bug, type:<Type>, severity:<Severity>
      **Type:** ...  **Severity:** ...
      ## Description / ## Steps to Reproduce / ## Expected Behavior / ## Actual Behavior
      ## Diagnostics   (a two-column markdown table: Engine, Platform, GPU, FPS, Scene ...)
      <details><summary>Console Log (last N lines)</summary> ``` ... ``` </details>

  [Crash] <exception line>   labels: bug, type:Crash, severity:Critical, auto-submitted
      crash text, including a stack trace

Hand-written issues have none of this and parse to mostly empty fields, which is fine.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field

API = "https://api.github.com"

# Hex addresses, pointer-ish numbers and frame indices change between two runs of the same
# crash. Strip them before comparing crash lines so the same bug hashes the same way.
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_NUM = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")


@dataclass
class Diagnostics:
    engine: str = ""
    platform: str = ""
    gpu: str = ""
    scene: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    url: str = ""
    created_at: str = ""
    updated_at: str = ""

    # Parsed from the body. Empty when the issue was not filed by the editor.
    report_type: str = ""
    severity: str = ""
    description: str = ""
    steps: str = ""
    expected: str = ""
    actual: str = ""
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    log_tail: list[str] = field(default_factory=list)

    @property
    def is_crash(self) -> bool:
        return self.title.startswith("[Crash]") or "type:Crash" in self.labels

    @property
    def auto_submitted(self) -> bool:
        return "auto-submitted" in self.labels

    def short(self, body_chars: int = 600) -> str:
        """One issue as the model sees it in the grouping prompt. Keeps the signal
        (title, type, platform, first error line) and drops the diagnostics table noise."""
        lines = [f"#{self.number} {self.title}"]
        meta = [
            x
            for x in (
                self.report_type,
                self.severity,
                self.diagnostics.platform,
                self.diagnostics.engine,
            )
            if x
        ]
        if meta:
            lines.append("  " + " | ".join(meta))
        text = self.description or self.actual or self.body
        text = _WS.sub(" ", text).strip()
        if text:
            lines.append("  " + text[:body_chars])
        err = first_error_line(self)
        if err:
            lines.append("  first error: " + err)
        return "\n".join(lines)


# ---------------------------------------------------------------- parsing

_SECTION = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_BOLD_META = re.compile(r"\*\*(Type|Severity):\*\*[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", re.MULTILINE)
_LOG_BLOCK = re.compile(r"<details><summary>Console Log[^<]*</summary>\s*```\n?(.*?)```", re.DOTALL)


def parse_body(issue: Issue) -> Issue:
    body = issue.body or ""

    for m in _BOLD_META.finditer(body):
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "type":
            issue.report_type = val
        elif key == "severity":
            issue.severity = val

    # Split on "## Heading" lines. The text after each heading runs to the next heading.
    parts = _SECTION.split(body)
    # parts = [preamble, heading1, text1, heading2, text2, ...]
    sections = {parts[i].strip().lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    issue.description = sections.get("description", "").strip()
    issue.steps = sections.get("steps to reproduce", "").strip()
    issue.expected = sections.get("expected behavior", "").strip()
    issue.actual = sections.get("actual behavior", "").strip()

    diag = sections.get("diagnostics", "")
    for key, val in _TABLE_ROW.findall(diag):
        k, v = key.strip(), val.strip()
        if k in ("Metric", "---") or set(k) <= {"-"}:
            continue
        if k == "Engine":
            issue.diagnostics.engine = v
        elif k == "Platform":
            issue.diagnostics.platform = v
        elif k == "GPU":
            issue.diagnostics.gpu = v
        elif k == "Scene":
            issue.diagnostics.scene = v
        else:
            issue.diagnostics.extra[k] = v

    m = _LOG_BLOCK.search(body)
    if m:
        issue.log_tail = [ln for ln in m.group(1).splitlines() if ln.strip()]

    # Labels carry type and severity too, and are more reliable than the body.
    for lab in issue.labels:
        if lab.startswith("type:") and not issue.report_type:
            issue.report_type = lab[5:]
        if lab.startswith("severity:") and not issue.severity:
            issue.severity = lab[9:]
    return issue


def normalize_crash_line(line: str) -> str:
    """Make two renderings of one crash compare equal: drop addresses and counters,
    collapse whitespace, lowercase."""
    line = _HEX.sub("0x?", line)
    line = _NUM.sub("#", line)
    return _WS.sub(" ", line).strip().lower()


_ERROR_WORDS = ("error", "exception", "assert", "failed", "fatal", "missing", "vk_error", "crash")


def first_error_line(issue: Issue) -> str:
    """The first log line that looks like the thing that went wrong. Crash issues use the
    exception line from the title; bug reports use the console tail."""
    if issue.is_crash:
        return normalize_crash_line(issue.title.removeprefix("[Crash]").strip())
    for ln in issue.log_tail:
        low = ln.lower()
        if any(w in low for w in _ERROR_WORDS):
            return normalize_crash_line(ln)
    return ""


# ---------------------------------------------------------------- fetching


def _token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    # The gh CLI already knows who you are. Borrow its token rather than ask for a second one.
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], check=False, capture_output=True, text=True, timeout=10
        )
        tok = out.stdout.strip()
        return tok or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _request(url: str, token: str | None, method: str = "GET", data: dict | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "enjin-triage",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(data).encode() if data is not None else None
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        link = resp.headers.get("Link", "")
        return json.loads(resp.read().decode()), link


def _next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part[part.find("<") + 1 : part.find(">")]
    return None


def fetch_issues(
    repo: str,
    token: str | None = None,
    since: str | None = None,
    limit: int = 50,
    state: str = "open",
) -> list[Issue]:
    """Open issues on the repo, most recently updated first, pull requests excluded."""
    tok = _token(token)
    params = {
        "state": state,
        "per_page": str(min(limit, 100)),
        "sort": "updated",
        "direction": "desc",
    }
    if since:
        params["since"] = since
    url: str | None = f"{API}/repos/{repo}/issues?" + urllib.parse.urlencode(params)
    out: list[Issue] = []
    while url and len(out) < limit:
        rows, link = _request(url, tok)
        for row in rows:
            if "pull_request" in row:
                continue
            out.append(issue_from_api(row))
            if len(out) >= limit:
                break
        url = _next_link(link)
    return out


def issue_from_api(row: dict) -> Issue:
    issue = Issue(
        number=int(row["number"]),
        title=row.get("title") or "",
        body=row.get("body") or "",
        labels=[
            lab["name"] if isinstance(lab, dict) else str(lab) for lab in row.get("labels", [])
        ],
        url=row.get("html_url", ""),
        created_at=row.get("created_at", ""),
        updated_at=row.get("updated_at", ""),
    )
    return parse_body(issue)


def load_issues_from_files(paths: Iterable[str]) -> list[Issue]:
    """Issues saved as GitHub API JSON (one object or a list per file). Used by the eval set
    and by tests, so nothing here needs the network."""
    out: list[Issue] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else [data]
        out.extend(issue_from_api(r) for r in rows)
    return out


def post_comment(repo: str, number: int, body: str, token: str | None = None) -> str:
    """Post one comment. Only called when the user passed --post."""
    tok = _token(token)
    if not tok:
        raise RuntimeError("no GitHub token: set GITHUB_TOKEN or run `gh auth login`")
    data, _ = _request(f"{API}/repos/{repo}/issues/{number}/comments", tok, "POST", {"body": body})
    return data.get("html_url", "")
