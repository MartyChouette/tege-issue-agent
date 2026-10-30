"""The investigation loop and the grouping step, driven by the scripted provider and a fake
MCP client. No network, no model."""

import json
import os

from triage.agent import GroupingResult, allowed_tools, group_issues, investigate
from triage.cluster import Group
from triage.config import Settings
from triage.issues import load_issues_from_files
from triage.llm import FakeProvider, Reply, ToolCall
from triage.mcp import McpResult, McpTool
from triage.report import comment_body, render_report

EVAL = os.path.join(os.path.dirname(__file__), "..", "eval", "issues.json")

MCP_TOOLS = [
    McpTool("get_log", "tail", {"type": "object", "properties": {"level": {"type": "string"}}}),
    McpTool("script_errors", "errors", {"type": "object", "properties": {}}),
    McpTool(
        "script_write", "write", {"type": "object", "properties": {"path": {"type": "string"}}}
    ),
    McpTool("build_game", "build", {"type": "object", "properties": {}}),
]


class FakeMcp:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        text = self.answers.get(name, "ok")
        return McpResult(text=text, is_error=False)


def _proposal(**over):
    base = {
        "status": "likely",
        "summary": "Swapchain recreate loses the device on Arc GPUs.",
        "root_cause": "VK_ERROR_DEVICE_LOST right after Recreating swapchain.",
        "evidence": ["[ERROR] [Renderer] vkQueueSubmit failed: VK_ERROR_DEVICE_LOST"],
        "proposed_fix": "Wait for the queue to idle before destroying the old swapchain.",
        "files": ["Engine/src/Renderer/Vulkan/VulkanRenderer.cpp"],
        "confidence": 0.7,
        "duplicate_of": 0,
    }
    base.update(over)
    return base


def test_read_only_mode_hides_write_tools():
    names = {t.name for t in allowed_tools(MCP_TOOLS, allow_write=False)}
    assert names == {"get_log", "script_errors"}
    assert {t.name for t in allowed_tools(MCP_TOOLS, allow_write=True)} == {
        t.name for t in MCP_TOOLS
    }


def test_loop_runs_tools_then_stops_on_propose_fix():
    issues = load_issues_from_files([EVAL])
    group = Group([113, 114], "Device lost on swapchain recreate")
    provider = FakeProvider(
        [
            Reply(
                text="Looking at the log.",
                stop="tool_use",
                tool_calls=[ToolCall("c1", "get_log", {"level": "error"})],
            ),
            Reply(stop="tool_use", tool_calls=[ToolCall("c2", "propose_fix", _proposal())]),
        ]
    )
    mcp = FakeMcp({"get_log": "[ERROR] [Renderer] vkQueueSubmit failed: VK_ERROR_DEVICE_LOST"})
    p = investigate(group, issues, provider, mcp, MCP_TOOLS, Settings(provider="fake"))

    assert p.status == "likely" and p.confidence == 0.7
    assert p.steps_used == 1
    assert mcp.calls == [("get_log", {"level": "error"})]
    assert [t.name for t in p.trace] == ["get_log"]
    # The tool result went back to the model as a tool_result for the same call id.
    second_call_msgs = provider.calls[1]["messages"]
    results = second_call_msgs[-1]["content"]
    assert results[0]["type"] == "tool_result" and results[0]["id"] == "c1"
    assert "DEVICE_LOST" in results[0]["content"]
    # propose_fix was offered as a tool, and write tools were not.
    offered = {t.name for t in provider.calls[0]["tools"]}
    assert "propose_fix" in offered and "script_write" not in offered


def test_blocked_tool_is_refused_without_touching_the_editor():
    issues = load_issues_from_files([EVAL])
    provider = FakeProvider(
        [
            Reply(stop="tool_use", tool_calls=[ToolCall("c1", "script_write", {"path": "x.as"})]),
            Reply(
                stop="tool_use",
                tool_calls=[ToolCall("c2", "propose_fix", _proposal(status="needs_info"))],
            ),
        ]
    )
    mcp = FakeMcp({})
    p = investigate(
        Group([109], "hang"), issues, provider, mcp, MCP_TOOLS, Settings(provider="fake")
    )
    assert mcp.calls == []
    res = provider.calls[1]["messages"][-1]["content"][0]
    assert res["is_error"] and "not available" in res["content"]
    assert p.status == "needs_info"


def test_step_budget_forces_a_proposal():
    issues = load_issues_from_files([EVAL])
    provider = FakeProvider(
        [
            Reply(stop="tool_use", tool_calls=[ToolCall("c1", "get_log", {})]),
            Reply(stop="tool_use", tool_calls=[ToolCall("c2", "get_log", {})]),
            # After the budget message, the model finally proposes.
            Reply(
                stop="tool_use",
                tool_calls=[ToolCall("c3", "propose_fix", _proposal(status="not_reproduced"))],
            ),
        ]
    )
    s = Settings(provider="fake", max_steps=2)
    p = investigate(Group([109], "hang"), issues, provider, FakeMcp({}), MCP_TOOLS, s)
    assert p.steps_used == 2 and p.status == "not_reproduced"
    budget_turn = provider.calls[2]
    assert [t.name for t in budget_turn["tools"]] == ["propose_fix"]
    assert "used every tool call" in budget_turn["messages"][-1]["content"]


def test_prose_ending_is_extracted_into_a_proposal():
    issues = load_issues_from_files([EVAL])
    provider = FakeProvider(
        [
            Reply(text="I think the font is hardcoded in UISystem. Done.", stop="end_turn"),
            Reply(text=json.dumps(_proposal(status="engine_side", confidence=0.6))),
        ]
    )
    p = investigate(
        Group([107, 108], "HUD font"),
        issues,
        provider,
        FakeMcp({}),
        MCP_TOOLS,
        Settings(provider="fake"),
    )
    assert p.status == "engine_side"
    assert any("without propose_fix" in n for n in p.notes)
    extract = provider.calls[1]
    assert extract["tools"] is None and extract["json_schema"] is not None
    assert "I think the font is hardcoded" in extract["messages"][0]["content"]


def test_model_grouping_is_accepted_when_it_is_a_partition():
    issues = load_issues_from_files([EVAL])
    nums = sorted(i.number for i in issues)
    groups = [
        {
            "issue_numbers": [101, 102],
            "canonical_title": "descriptor set destroyed",
            "reason": "same exception",
            "confidence": 0.95,
        },
        {
            "issue_numbers": [107, 108],
            "canonical_title": "HUD font locked",
            "reason": "same limitation",
            "confidence": 0.8,
        },
    ]
    rest = [n for n in nums if n not in (101, 102, 107, 108)]
    groups += [
        {"issue_numbers": [n], "canonical_title": "", "reason": "", "confidence": 1} for n in rest
    ]
    provider = FakeProvider([Reply(text=json.dumps({"groups": groups}))])
    r = group_issues(issues, provider, Settings(provider="fake"))
    assert r.used_model and (107, 108) in {g.key() for g in r.partition.groups}
    assert provider.calls[0]["json_schema"] is not None
    assert "## Candidate groups" in provider.calls[0]["messages"][0]["content"]


def test_model_grouping_that_drops_an_issue_is_rejected():
    issues = load_issues_from_files([EVAL])
    provider = FakeProvider(
        [
            Reply(
                text=json.dumps(
                    {
                        "groups": [
                            {
                                "issue_numbers": [101, 102],
                                "canonical_title": "x",
                                "reason": "",
                                "confidence": 1,
                            }
                        ]
                    }
                )
            )
        ]
    )
    r = group_issues(issues, provider, Settings(provider="fake"))
    assert not r.used_model
    assert any("left out" in n for n in r.notes)
    assert r.partition.numbers() == {i.number for i in issues}


def test_report_and_comment_render():
    issues = load_issues_from_files([EVAL])
    from triage.cluster import heuristic_partition

    part = heuristic_partition(issues)
    grouping = GroupingResult(part, used_model=False, notes=["kept heuristic"])
    from triage.agent import Proposal

    prop = Proposal.from_dict(_proposal())
    prop.trace = []
    key = next(g.key() for g in part.groups if 113 in g.numbers)
    text = render_report(issues, grouping, {key: prop}, Settings(), "fake", "none", {}, False)
    assert "# Triage report: MartyChouette/TEGE" in text
    assert "## Duplicate groups" in text and "#113" in text and "#114" in text
    assert "Likely cause" in text and "VK_ERROR_DEVICE_LOST" in text
    assert "> kept heuristic" in text
    assert chr(0x2014) not in text  # no em dashes anywhere in generated text
    c = comment_body(Group([113, 114], "x"), prop, "none")
    assert "Automated triage" in c and "#113, #114" in c and "not reviewed" in c
