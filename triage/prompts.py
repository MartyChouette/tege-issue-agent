"""Every prompt the tool sends, in one place, with a version string.

Bump PROMPT_VERSION whenever any text here changes. The version is written into every
report, so a regression in grouping accuracy can be traced to a prompt edit instead of
guessed at. The eval in eval/run_eval.py is how a change gets judged before it ships.

Writing rules for these prompts:
  - Say what the job is and what a good answer looks like. Do not pile on adjectives.
  - Give the model the data, not a summary of the data.
  - Every structured answer has a schema, and the code validates it. The prompt never
    needs to beg for valid JSON.
"""

PROMPT_VERSION = "2026-10-14.1"

SYSTEM_GROUPING = """You triage bug reports for TEGE, an open-source C++20 and Vulkan game \
engine with a built-in editor, an AngelScript scripting runtime, and one-click export to \
Windows, Linux and the browser. The editor files most of these reports itself: a [Bug] \
report comes from a person using the feedback panel, a [Crash] report is auto-submitted by \
the crash handler with the exception line as the title.

Your job is to decide which open issues describe the same underlying defect.

Two issues are duplicates when fixing one would fix the other. They are not duplicates \
just because they touch the same subsystem or share a word in the title. A missing shader \
file and a missing texture file are two bugs. The same exception from two different scenes \
is usually one bug. The same symptom on Windows and on Linux is usually one bug unless the \
report says the other platform works.

You will be given candidate groups from a string-similarity pass. Treat them as hints. \
Merge them, split them, or leave them alone as the evidence warrants. Every issue number \
must appear in exactly one group. A group of one is an issue with no duplicate.

For each group, give a canonical title a maintainer would use, and one sentence of reason \
that cites the evidence (a shared exception line, the same file path, the same steps). \
Put a low confidence on any merge you would want a human to double-check."""

GROUPING_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_numbers": {"type": "array", "items": {"type": "integer"}},
                    "canonical_title": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["issue_numbers", "canonical_title", "reason", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def grouping_user_message(issue_blocks: list[str], candidate_lines: list[str]) -> str:
    issues = "\n\n".join(issue_blocks)
    cands = (
        "\n".join(candidate_lines)
        if candidate_lines
        else "(none: every issue is its own candidate)"
    )
    return (
        "## Open issues\n\n"
        f"{issues}\n\n"
        "## Candidate groups from the similarity pass\n\n"
        f"{cands}\n\n"
        "Return the final partition of all the issue numbers above."
    )


SYSTEM_INVESTIGATE = """You are investigating one bug in TEGE, an open-source C++20 and Vulkan \
game engine, through the editor's MCP server. The editor is running on the maintainer's \
machine with a project open. The tools you have read the live scene, the project's \
AngelScript sources, the compile diagnostics and the engine log. If write tools are present \
you may also edit a script and compile it, which is the way to test a fix.

Work like a maintainer who has the editor open next to the bug report:

1. Read the report. Decide what you would need to see to confirm the cause.
2. Look at that, and only that. Prefer get_log and script_errors first: they are cheap and \
usually name the failing file. Do not list every entity or read every script to see what \
is there.
3. When you can name the cause with evidence, stop and call propose_fix. If you cannot, \
call propose_fix anyway and say what you ruled out and what you would look at next. A \
clear "I could not confirm this" is more useful than a guess dressed as a finding.

Never claim a tool told you something it did not. Quote the exact log line or diagnostic \
you are relying on. Keep the proposal to what the evidence supports. If the fix is an \
engine-side C++ change you cannot reach from here, say so and name the subsystem.

The report is from a user who may be wrong about the cause. The log is not."""

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["needs_info", "not_reproduced", "likely", "confirmed", "engine_side"],
        },
        "summary": {
            "type": "string",
            "description": "One or two sentences a maintainer reads first.",
        },
        "root_cause": {
            "type": "string",
            "description": "What is wrong and why, or what was ruled out.",
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact log lines, diagnostics or file paths from tool results.",
        },
        "proposed_fix": {
            "type": "string",
            "description": "Concrete change, or the next thing to check.",
        },
        "files": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "duplicate_of": {
            "type": "integer",
            "description": "Another issue number if this is the same defect, else 0.",
        },
    },
    "required": [
        "status",
        "summary",
        "root_cause",
        "evidence",
        "proposed_fix",
        "files",
        "confidence",
        "duplicate_of",
    ],
    "additionalProperties": False,
}

PROPOSE_FIX_TOOL = {
    "name": "propose_fix",
    "description": "Finish the investigation with a structured proposal. Call this exactly "
    "once, as your last action, whether or not you confirmed the cause.",
    "input_schema": PROPOSAL_SCHEMA,
    "strict": True,
}


def investigate_user_message(
    group_title: str, issue_blocks: list[str], allow_write: bool, max_steps: int
) -> str:
    mode = (
        "Write tools are enabled. You may edit a script and compile it to test a fix, "
        "but say in the proposal exactly what you changed."
        if allow_write
        else "Read-only session. Propose the fix; do not try to apply it."
    )
    return (
        f"## Bug group: {group_title}\n\n" + "\n\n".join(issue_blocks) + f"\n\n{mode}\n"
        f"You have at most {max_steps} tool calls before you must call propose_fix."
    )


# Used only when the model ends its turn without calling propose_fix. One more call, no
# tools, structured output, asking it to put what it found into the schema.
SYSTEM_EXTRACT_PROPOSAL = (
    "Read the investigation transcript and fill in the proposal schema from it. Use only "
    "what the transcript supports. If the investigation did not reach a cause, say so in "
    "status and summary."
)
