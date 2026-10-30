import json

from triage.llm import (
    FakeProvider,
    Reply,
    ToolCall,
    ToolSpec,
    assistant_message,
    claude_tool,
    ollama_tool,
    to_claude_messages,
    to_ollama_messages,
    tool_results_message,
    transcript_as_text,
)


def _conversation():
    call = ToolCall(id="toolu_1", name="get_log", input={"level": "error"})
    reply = Reply(text="Checking the log.", tool_calls=[call], stop="tool_use")
    return [
        {"role": "user", "content": "Investigate #9."},
        assistant_message(reply),
        tool_results_message([(call, "[ERROR] boom", False)]),
    ], call


def test_claude_shape_round_trips_tool_use_and_results():
    msgs, _ = _conversation()
    out = to_claude_messages(msgs)
    assert out[0] == {"role": "user", "content": "Investigate #9."}
    assert out[1]["role"] == "assistant"
    kinds = [b["type"] for b in out[1]["content"]]
    assert kinds == ["text", "tool_use"]
    assert out[1]["content"][1]["id"] == "toolu_1"
    assert out[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "[ERROR] boom",
        "is_error": False,
    }


def test_claude_replays_raw_assistant_content_when_present():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_log", "input": {}},
    ]
    reply = Reply(tool_calls=[ToolCall("toolu_1", "get_log", {})], raw=raw)
    out = to_claude_messages([{"role": "user", "content": "x"}, assistant_message(reply)])
    assert out[1]["content"] is raw


def test_ollama_shape_uses_tool_role_and_function_calls():
    msgs, _ = _conversation()
    out = to_ollama_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"] == {
        "name": "get_log",
        "arguments": {"level": "error"},
    }
    assert out[3] == {"role": "tool", "content": "[ERROR] boom", "tool_name": "get_log"}


def test_tool_definitions_per_backend():
    spec = ToolSpec(
        "script_read", "Read a script", {"type": "object", "properties": {}}, strict=True
    )
    c = claude_tool(spec)
    assert c["strict"] is True and c["input_schema"] == spec.input_schema
    o = ollama_tool(spec)
    assert o["type"] == "function" and o["function"]["parameters"] == spec.input_schema


def test_transcript_text_names_tools_and_results():
    msgs, _ = _conversation()
    text = transcript_as_text(msgs)
    assert "[assistant calls get_log]" in text
    assert "[get_log result]\n[ERROR] boom" in text


def test_fake_provider_defaults_fill_the_schema():
    fake = FakeProvider()
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["needs_info", "ok"]},
            "n": {"type": "integer"},
            "xs": {"type": "array"},
        },
        "required": ["status", "n", "xs"],
        "additionalProperties": False,
    }
    r = fake.complete("s", [{"role": "user", "content": "u"}], json_schema=schema)
    assert json.loads(r.text) == {"status": "needs_info", "n": 0, "xs": []}
    assert fake.calls[0]["json_schema"] is schema
