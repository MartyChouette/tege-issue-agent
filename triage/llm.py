"""One interface over two model backends, plus a scripted fake for tests.

The agent code works with a neutral message format so it never knows which backend it is
talking to:

    {"role": "user", "content": "text"}
    {"role": "assistant", "content": [ {type: text, text} | {type: tool_call, id, name, input} ],
                          "_raw": <backend-native content for exact replay, or None>}
    {"role": "user", "content": [ {type: tool_result, id, name, content, is_error} ]}

Each provider converts to and from its own wire shape. The conversion functions are plain
and pure so tests can check them without a network.

Claude is the default because the investigation loop benefits from a strong model with
thinking. Ollama is here so the whole tool can run on a machine with no vendor account:
point OLLAMA_HOST at a local server and pick any model that supports tool calling.
"""

from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from triage.config import Settings


class RefusalError(RuntimeError):
    """The model declined the request for policy reasons. Not retried."""


class ProviderError(RuntimeError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    strict: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end_turn"
    usage: dict = field(default_factory=dict)
    raw: Any = None  # backend-native assistant content, replayed verbatim

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
        effort: str | None = None,
    ) -> Reply: ...


# ---------------------------------------------------------------- neutral helpers


def assistant_message(reply: Reply) -> dict:
    parts: list[dict] = []
    if reply.text:
        parts.append({"type": "text", "text": reply.text})
    for tc in reply.tool_calls:
        parts.append({"type": "tool_call", "id": tc.id, "name": tc.name, "input": tc.input})
    return {"role": "assistant", "content": parts, "_raw": reply.raw}


def tool_results_message(results: list[tuple[ToolCall, str, bool]]) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "id": tc.id,
                "name": tc.name,
                "content": text,
                "is_error": is_error,
            }
            for tc, text, is_error in results
        ],
    }


def transcript_as_text(messages: list[dict]) -> str:
    """Flatten a conversation to readable text. Used when a second, tool-less request has to
    read what happened in the first."""
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append(f"[{role}]\n{content}")
            continue
        for part in content:
            t = part.get("type")
            if t == "text":
                out.append(f"[{role}]\n{part['text']}")
            elif t == "tool_call":
                out.append(f"[{role} calls {part['name']}]\n{json.dumps(part['input'])}")
            elif t == "tool_result":
                tag = "error" if part.get("is_error") else "result"
                out.append(f"[{part['name']} {tag}]\n{part['content']}")
    return "\n\n".join(out)


def _usage_add(total: dict, usage: dict) -> None:
    for k, v in usage.items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v


# ---------------------------------------------------------------- Claude


def to_claude_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            if m.get("_raw"):
                out.append({"role": "assistant", "content": m["_raw"]})
                continue
            blocks = []
            for p in content:
                if p["type"] == "text":
                    blocks.append({"type": "text", "text": p["text"]})
                elif p["type"] == "tool_call":
                    blocks.append(
                        {"type": "tool_use", "id": p["id"], "name": p["name"], "input": p["input"]}
                    )
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(content, str):
            out.append({"role": "user", "content": content})
        else:
            blocks = []
            for p in content:
                if p["type"] == "tool_result":
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": p["id"],
                            "content": p["content"],
                            "is_error": bool(p.get("is_error")),
                        }
                    )
                elif p["type"] == "text":
                    blocks.append({"type": "text", "text": p["text"]})
            out.append({"role": "user", "content": blocks})
    return out


def claude_tool(spec: ToolSpec) -> dict:
    t = {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
    if spec.strict:
        t["strict"] = True
    return t


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str, fallbacks: bool = True, client=None):
        import anthropic  # imported here so the ollama and fake paths need no SDK

        self.model = model
        self.fallbacks = fallbacks
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.usage: dict = {}

    def complete(self, system, messages, tools=None, json_schema=None, effort=None) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            # Stable system text first, with a cache breakpoint, so the per-issue content that
            # follows is the only part paid at full price on each call.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": to_claude_messages(messages),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort or "medium"},
        }
        if tools:
            kwargs["tools"] = [claude_tool(t) for t in tools]
        if json_schema:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": json_schema}
        if self.fallbacks:
            # Server-side refusal fallback: if a safety classifier declines, the API reruns the
            # request on a fallback model inside the same call.
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}

        a = self._anthropic
        try:
            resp = self.client.messages.create(**kwargs)
        except a.RateLimitError as e:
            raise ProviderError(f"rate limited by Anthropic: {e.message}") from e
        except a.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise ProviderError(f"cannot reach Anthropic API: {e}") from e

        if resp.stop_reason == "refusal":
            detail = getattr(resp, "stop_details", None)
            raise RefusalError(f"model refused: {getattr(detail, 'explanation', '') or ''}")

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                inp = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append(ToolCall(id=block.id, name=block.name, input=inp))
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
        _usage_add(self.usage, usage)
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        return Reply(
            text="".join(text_parts),
            tool_calls=calls,
            stop=resp.stop_reason or "",
            usage=usage,
            raw=raw,
        )


# ---------------------------------------------------------------- Ollama


def to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            text = "".join(p["text"] for p in content if p["type"] == "text")
            calls = [
                {"function": {"name": p["name"], "arguments": p["input"]}}
                for p in content
                if p["type"] == "tool_call"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif isinstance(content, str):
            out.append({"role": "user", "content": content})
        else:
            for p in content:
                if p["type"] == "tool_result":
                    out.append({"role": "tool", "content": p["content"], "tool_name": p["name"]})
                elif p["type"] == "text":
                    out.append({"role": "user", "content": p["text"]})
    return out


def ollama_tool(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, timeout: float = 600):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.usage: dict = {}

    def complete(self, system, messages, tools=None, json_schema=None, effort=None) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = [ollama_tool(t) for t in tools]
        if json_schema:
            payload["format"] = json_schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise ProviderError(
                f"ollama {e.code}: {e.read().decode(errors='replace')[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise ProviderError(f"cannot reach ollama at {self.host}: {e.reason}") from e

        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(
                ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name", ""), input=args)
            )
        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }
        _usage_add(self.usage, usage)
        return Reply(
            text=msg.get("content", "") or "",
            tool_calls=calls,
            stop="tool_use" if calls else "end_turn",
            usage=usage,
            raw=None,
        )


# ---------------------------------------------------------------- Fake


class FakeProvider:
    """Scripted replies for tests and for the offline demo. Each entry is a Reply, or a
    callable taking (system, messages, tools, json_schema) and returning a Reply."""

    name = "fake"

    def __init__(
        self, replies: list[Reply | Callable[..., Reply]] | None = None, model: str = "fake"
    ):
        self.model = model
        self.replies = list(replies or [])
        self.calls: list[dict] = []
        self.usage: dict = {}

    def complete(self, system, messages, tools=None, json_schema=None, effort=None) -> Reply:
        # Copy the transcript as it stood at this call. The loop appends to the same list
        # afterwards, and a test wants to see what the model was shown, not what came later.
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "json_schema": json_schema,
                "effort": effort,
            }
        )
        if not self.replies:
            # With no script, answer every structured request with a sane default and every
            # tool session with "nothing to add", so a demo run completes end to end.
            if json_schema is not None:
                return Reply(text=json.dumps(_default_for_schema(json_schema)))
            return Reply(text="No further action.")
        nxt = self.replies.pop(0)
        return nxt(system, messages, tools, json_schema) if callable(nxt) else nxt


def _default_for_schema(schema: dict) -> Any:
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: _default_for_schema(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    if t == "string":
        return ""
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    return None


# ---------------------------------------------------------------- factory


def make_provider(s: Settings) -> Provider:
    if s.provider == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # The SDK would raise too, but with a stack trace instead of the next step.
            raise ProviderError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY, or run with "
                "--provider ollama for a local model, or --provider fake for a dry run"
            )
        return ClaudeProvider(model=s.resolved_model(), fallbacks=s.fallbacks)
    if s.provider == "ollama":
        return OllamaProvider(model=s.resolved_model(), host=s.ollama_host)
    if s.provider == "fake":
        return FakeProvider()
    raise ProviderError(f"unknown provider {s.provider!r}; use claude, ollama or fake")
