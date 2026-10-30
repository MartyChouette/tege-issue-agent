"""The two model-driven steps: confirm the duplicate groups, then investigate each one.

Grouping is one structured call. The heuristic pass has already done the obvious merges;
the model sees those as candidates and returns the final partition as JSON that the code
validates. If the answer is not a partition of exactly these issues, the heuristic result
stands and the report says so. A wrong merge from a model is worse than no merge.

Investigation is a tool loop. The model gets the editor's MCP tools (read-only unless
--allow-write), plus one local tool, propose_fix, which is how it ends. The loop stops on
propose_fix, on the step budget, or when the model ends its turn without proposing, in
which case one more tool-less call turns the transcript into a proposal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from triage import prompts
from triage.cluster import Group, Partition, heuristic_partition, validate_partition
from triage.config import READ_ONLY_TOOLS, Settings
from triage.issues import Issue
from triage.llm import (
    Provider,
    Reply,
    ToolCall,
    ToolSpec,
    assistant_message,
    tool_results_message,
    transcript_as_text,
)
from triage.mcp import McpClient, McpTool


@dataclass
class ToolTrace:
    name: str
    input: dict
    ok: bool
    output_chars: int


@dataclass
class Proposal:
    status: str = "needs_info"
    summary: str = ""
    root_cause: str = ""
    evidence: list[str] = field(default_factory=list)
    proposed_fix: str = ""
    files: list[str] = field(default_factory=list)
    confidence: float = 0.0
    duplicate_of: int = 0
    steps_used: int = 0
    trace: list[ToolTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # anything the loop had to work around

    @classmethod
    def from_dict(cls, d: dict) -> Proposal:
        p = cls()
        for k in ("status", "summary", "root_cause", "proposed_fix"):
            if isinstance(d.get(k), str):
                setattr(p, k, d[k])
        p.evidence = [str(x) for x in d.get("evidence", []) or []]
        p.files = [str(x) for x in d.get("files", []) or []]
        try:
            p.confidence = float(d.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            p.confidence = 0.0
        try:
            p.duplicate_of = int(d.get("duplicate_of", 0) or 0)
        except (TypeError, ValueError):
            p.duplicate_of = 0
        return p

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- grouping


@dataclass
class GroupingResult:
    partition: Partition
    used_model: bool
    notes: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def group_issues(
    issues: list[Issue], provider: Provider | None, settings: Settings
) -> GroupingResult:
    base = heuristic_partition(issues)
    if provider is None or len(issues) < 2:
        return GroupingResult(partition=base, used_model=False)

    blocks = [i.short() for i in sorted(issues, key=lambda i: i.number)]
    cands = [f"- {g.numbers} {g.reason}" for g in base.duplicates()]
    reply = provider.complete(
        system=prompts.SYSTEM_GROUPING,
        messages=[{"role": "user", "content": prompts.grouping_user_message(blocks, cands)}],
        json_schema=prompts.GROUPING_SCHEMA,
        effort=settings.effort,
    )
    notes: list[str] = []
    try:
        data = json.loads(reply.text)
        groups = []
        for g in data["groups"]:
            nums = sorted({int(n) for n in g["issue_numbers"]})
            groups.append(
                Group(
                    numbers=nums,
                    canonical_title=str(g.get("canonical_title", "")),
                    reason=str(g.get("reason", "")),
                    confidence=float(g.get("confidence", 0.5)),
                    source="model",
                )
            )
        groups.sort(key=lambda g: g.numbers[0] if g.numbers else 0)
        model_partition = Partition(groups)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        notes.append(f"model grouping unreadable ({e.__class__.__name__}); kept heuristic groups")
        return GroupingResult(base, used_model=False, notes=notes, usage=reply.usage)

    problems = validate_partition(model_partition, issues)
    if problems:
        notes.append("model grouping rejected: " + "; ".join(problems) + ". Kept heuristic groups.")
        return GroupingResult(base, used_model=False, notes=notes, usage=reply.usage)
    return GroupingResult(model_partition, used_model=True, notes=notes, usage=reply.usage)


# ---------------------------------------------------------------- investigation


def allowed_tools(mcp_tools: list[McpTool], allow_write: bool) -> list[ToolSpec]:
    specs = []
    for t in mcp_tools:
        if allow_write or t.name in READ_ONLY_TOOLS:
            specs.append(ToolSpec(t.name, t.description, t.input_schema))
    return specs


def _propose_spec() -> ToolSpec:
    t = prompts.PROPOSE_FIX_TOOL
    return ToolSpec(t["name"], t["description"], t["input_schema"], strict=True)


def investigate(
    group: Group,
    issues: list[Issue],
    provider: Provider,
    mcp: McpClient | None,
    mcp_tools: list[McpTool],
    settings: Settings,
) -> Proposal:
    by_num = {i.number: i for i in issues}
    members = [by_num[n] for n in group.numbers if n in by_num]
    blocks = [_issue_block(i) for i in members]

    tools = [*allowed_tools(mcp_tools, settings.allow_write), _propose_spec()]
    messages: list[dict] = [
        {
            "role": "user",
            "content": prompts.investigate_user_message(
                group.canonical_title, blocks, settings.allow_write, settings.max_steps
            ),
        }
    ]

    proposal: Proposal | None = None
    trace: list[ToolTrace] = []
    notes: list[str] = []
    steps = 0

    while proposal is None:
        reply = provider.complete(
            system=prompts.SYSTEM_INVESTIGATE,
            messages=messages,
            tools=tools,
            effort=settings.agent_effort,
        )
        messages.append(assistant_message(reply))

        if not reply.wants_tools:
            # Ended its turn in prose. Ask once more for the schema, with no tools.
            notes.append("model ended without propose_fix; extracted proposal from transcript")
            proposal = _extract_proposal(provider, messages, settings)
            break

        results: list[tuple[ToolCall, str, bool]] = []
        for call in reply.tool_calls:
            if call.name == "propose_fix":
                proposal = Proposal.from_dict(call.input)
                results.append((call, "recorded", False))
                continue
            steps += 1
            text, is_error = _run_mcp_tool(mcp, call, tools)
            trace.append(ToolTrace(call.name, call.input, not is_error, len(text)))
            results.append((call, text, is_error))
        messages.append(tool_results_message(results))

        if proposal is None and steps >= settings.max_steps:
            # Out of budget. One last turn with only propose_fix on the table.
            messages.append(
                {
                    "role": "user",
                    "content": "You have used every tool call. "
                    "Call propose_fix now with what you have.",
                }
            )
            reply = provider.complete(
                system=prompts.SYSTEM_INVESTIGATE,
                messages=messages,
                tools=[_propose_spec()],
                effort=settings.agent_effort,
            )
            messages.append(assistant_message(reply))
            for call in reply.tool_calls:
                if call.name == "propose_fix":
                    proposal = Proposal.from_dict(call.input)
            if proposal is None:
                notes.append("step budget hit and no propose_fix; extracted from transcript")
                proposal = _extract_proposal(provider, messages, settings)

    proposal.steps_used = steps
    proposal.trace = trace
    proposal.notes = notes
    return proposal


def _run_mcp_tool(mcp: McpClient | None, call: ToolCall, tools: list[ToolSpec]) -> tuple[str, bool]:
    names = {t.name for t in tools}
    if call.name not in names:
        return (f"tool {call.name!r} is not available in this session", True)
    if mcp is None:
        return ("no editor connected: the MCP server is not reachable in this run", True)
    res = mcp.call_tool(call.name, call.input)
    text = res.text if res.text else "(empty result)"
    # Keep any one result from swamping the context. The model can ask for a narrower read.
    if len(text) > 12000:
        text = text[:12000] + f"\n... [{len(res.text) - 12000} more characters cut]"
    return (text, res.is_error)


def _extract_proposal(provider: Provider, messages: list[dict], settings: Settings) -> Proposal:
    transcript = transcript_as_text(messages)
    reply = provider.complete(
        system=prompts.SYSTEM_EXTRACT_PROPOSAL,
        messages=[{"role": "user", "content": transcript}],
        json_schema=prompts.PROPOSAL_SCHEMA,
        effort="low",
    )
    try:
        return Proposal.from_dict(json.loads(reply.text))
    except (json.JSONDecodeError, TypeError):
        p = Proposal(status="needs_info", summary="The model did not produce a readable proposal.")
        p.notes.append("extraction returned non-JSON")
        return p


def _issue_block(i: Issue) -> str:
    lines = [f"### #{i.number} {i.title}"]
    if i.url:
        lines.append(i.url)
    meta = [
        x
        for x in (
            i.report_type,
            i.severity,
            i.diagnostics.platform,
            i.diagnostics.engine,
            i.diagnostics.gpu,
        )
        if x
    ]
    if meta:
        lines.append(" | ".join(meta))
    for label, text in (
        ("Description", i.description),
        ("Steps", i.steps),
        ("Expected", i.expected),
        ("Actual", i.actual),
    ):
        if text:
            lines.append(f"{label}: {text}")
    if not (i.description or i.actual):
        lines.append(i.body[:3000])
    if i.diagnostics.scene:
        lines.append(f"Scene: {i.diagnostics.scene}")
    if i.log_tail:
        lines.append("Console tail:\n```\n" + "\n".join(i.log_tail[-40:]) + "\n```")
    return "\n".join(lines)


def demo_reply_for_fake(group: Group) -> Reply:
    """A scripted propose_fix call, used by `--provider fake` so the loop runs end to end."""
    return Reply(
        tool_calls=[
            ToolCall(
                id="fake-1",
                name="propose_fix",
                input={
                    "status": "needs_info",
                    "summary": f"Dry run for {group.canonical_title!r}: no model was consulted.",
                    "root_cause": "Not investigated. "
                    "Run with --provider claude or --provider ollama.",
                    "evidence": [],
                    "proposed_fix": "",
                    "files": [],
                    "confidence": 0.0,
                    "duplicate_of": 0,
                },
            )
        ],
        stop="tool_use",
    )
