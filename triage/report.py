"""Turn a run into a markdown report, and into the comment that goes on an issue.

The report is the deliverable a maintainer reads over coffee: which issues are duplicates
of which, what the agent found, and what it looked at to find it. Everything a reviewer
would need to distrust a finding is in here too: the prompt version, the model, the tool
calls made, and any place the code overrode the model.
"""

from __future__ import annotations

import datetime as dt
import os

from triage.agent import GroupingResult, Proposal
from triage.cluster import Group
from triage.config import Settings
from triage.issues import Issue
from triage.prompts import PROMPT_VERSION

STATUS_WORDS = {
    "confirmed": "Confirmed",
    "likely": "Likely cause",
    "not_reproduced": "Not reproduced",
    "needs_info": "Needs more information",
    "engine_side": "Engine-side change",
}


def render_report(
    issues: list[Issue],
    grouping: GroupingResult,
    proposals: dict[tuple[int, ...], Proposal],
    settings: Settings,
    provider_name: str,
    model: str,
    usage: dict,
    mcp_connected: bool,
    when: dt.datetime | None = None,
) -> str:
    when = when or dt.datetime.now(dt.UTC)
    by_num = {i.number: i for i in issues}
    dups = grouping.partition.duplicates()
    singles = [g for g in grouping.partition.groups if len(g.numbers) == 1]

    out: list[str] = []
    out.append(f"# Triage report: {settings.repo}")
    out.append("")
    out.append(
        f"Run at {when.strftime('%Y-%m-%d %H:%M UTC')}. "
        f"{len(issues)} open issues, {len(dups)} duplicate groups, "
        f"{len(singles)} issues standing alone."
    )
    out.append("")
    out.append("| | |")
    out.append("|---|---|")
    out.append(f"| Model | {provider_name} / {model} |")
    out.append(f"| Prompt version | {PROMPT_VERSION} |")
    out.append(f"| Grouping | {'model-confirmed' if grouping.used_model else 'heuristic only'} |")
    out.append(f"| Editor | {'connected over MCP' if mcp_connected else 'not connected'} |")
    out.append(f"| Write tools | {'enabled' if settings.allow_write else 'off'} |")
    if usage:
        toks = ", ".join(
            f"{k.replace('_', ' ')} {v}"
            for k, v in sorted(usage.items())
            if isinstance(v, (int, float)) and v
        )
        out.append(f"| Tokens | {toks} |")
    out.append("")
    for n in grouping.notes:
        out.append(f"> {n}")
    if grouping.notes:
        out.append("")

    if dups:
        out.append("## Duplicate groups")
        out.append("")
        for g in dups:
            out.append(_group_heading(g, by_num))
            out.append("")
            for n in g.numbers:
                i = by_num.get(n)
                if i:
                    out.append(f"- #{n} {i.title}" + (f" ({i.url})" if i.url else ""))
            if g.reason:
                out.append("")
                out.append(f"Why: {g.reason} (confidence {g.confidence:.2f}, {g.source})")
            out.append("")
            p = proposals.get(g.key())
            if p:
                out.extend(_proposal_lines(p))
                out.append("")

    out.append("## Issues standing alone")
    out.append("")
    for g in singles:
        i = by_num[g.numbers[0]]
        out.append(f"### #{i.number} {i.title}")
        out.append("")
        p = proposals.get(g.key())
        if p:
            out.extend(_proposal_lines(p))
        else:
            out.append("Not investigated in this run.")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def _group_heading(g: Group, by_num: dict[int, Issue]) -> str:
    title = g.canonical_title or by_num[g.numbers[0]].title
    nums = ", ".join(f"#{n}" for n in g.numbers)
    return f"### {title}  ({nums})"


def _proposal_lines(p: Proposal) -> list[str]:
    lines = [
        f"**{STATUS_WORDS.get(p.status, p.status)}**"
        + (f", confidence {p.confidence:.2f}" if p.confidence else "")
        + (f", duplicate of #{p.duplicate_of}" if p.duplicate_of else "")
    ]
    if p.summary:
        lines.append("")
        lines.append(p.summary)
    if p.root_cause:
        lines.append("")
        lines.append(f"Cause: {p.root_cause}")
    if p.evidence:
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"- `{e}`" if "\n" not in e else f"- {e}" for e in p.evidence)
    if p.proposed_fix:
        lines.append("")
        lines.append(f"Proposed fix: {p.proposed_fix}")
    if p.files:
        lines.append("")
        lines.append("Files: " + ", ".join(f"`{f}`" for f in p.files))
    if p.trace:
        lines.append("")
        lines.append(f"<details><summary>{len(p.trace)} tool calls</summary>")
        lines.append("")
        for t in p.trace:
            mark = "ok" if t.ok else "error"
            args = ", ".join(f"{k}={v!r}" for k, v in t.input.items())
            lines.append(f"- `{t.name}({args})` {mark}, {t.output_chars} chars")
        lines.append("")
        lines.append("</details>")
    for n in p.notes:
        lines.append("")
        lines.append(f"> {n}")
    return lines


def comment_body(group: Group, proposal: Proposal, model: str) -> str:
    """What gets posted on each issue in the group when --post is given. Short, and honest
    about where it came from."""
    others = [n for n in group.numbers]
    head = f"Automated triage ({model}, prompt {PROMPT_VERSION})."
    if len(others) > 1:
        head += " Likely the same defect as " + ", ".join(f"#{n}" for n in others) + "."
    body = [head, ""]
    body.extend(_proposal_lines(Proposal(**{**proposal.to_dict(), "trace": [], "notes": []})))
    body.append("")
    body.append("A person has not reviewed this yet. Treat it as a lead, not a verdict.")
    return "\n".join(body)


def write_report(text: str, out_dir: str, when: dt.datetime | None = None) -> str:
    when = when or dt.datetime.now(dt.UTC)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, when.strftime("triage-%Y%m%d-%H%M%S.md"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path
