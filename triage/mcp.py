"""A small MCP client for the TEGE editor's server.

The editor speaks MCP over plain HTTP: one JSON-RPC 2.0 request per POST, a JSON-RPC
response back. Protocol version 2024-11-05. Three methods matter here: initialize,
tools/list, tools/call. The server binds to 127.0.0.1 only and runs every tool call on the
editor's main thread, answering with an error after ten seconds if the editor is busy, so
the client timeout sits a little above that.

No SDK is used on purpose. The whole protocol surface this tool needs is forty lines, and
a reader should be able to see every byte that goes over the wire.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
TIMEOUT_S = 15


class McpError(RuntimeError):
    pass


@dataclass
class McpTool:
    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class McpResult:
    text: str
    is_error: bool = False
    raw: Any = None


class McpClient:
    def __init__(self, url: str, timeout: float = TIMEOUT_S):
        self.url = url
        self.timeout = timeout
        self._id = 0
        self.server_info: dict = {}
        self.calls: list[tuple[str, dict]] = []  # every tools/call made, for the report

    # ------------------------------------------------------------ transport

    def _post(self, payload: dict) -> dict | None:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            raise McpError(
                f"cannot reach MCP server at {self.url}: {e.reason}. "
                "Is the editor running with Settings > System > MCP Server on?"
            ) from e
        if not raw.strip():
            return None  # notifications get an empty body
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise McpError(f"MCP server sent non-JSON: {raw[:200]!r}") from e

    def _call(self, method: str, params: dict | None = None) -> Any:
        self._id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            payload["params"] = params
        resp = self._post(payload)
        if resp is None:
            raise McpError(f"empty response to {method}")
        if "error" in resp:
            err = resp["error"]
            raise McpError(f"{method} failed: {err.get('code')} {err.get('message')}")
        return resp.get("result")

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload)

    # ------------------------------------------------------------ protocol

    def initialize(self) -> dict:
        result = self._call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "enjin-triage", "version": "0.1.0"},
            },
        )
        self.server_info = result or {}
        self._notify("notifications/initialized")
        return self.server_info

    def list_tools(self) -> list[McpTool]:
        result = self._call("tools/list") or {}
        tools = []
        for t in result.get("tools", []):
            tools.append(
                McpTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
                )
            )
        tools.sort(key=lambda t: t.name)  # stable order keeps the prompt cache warm
        return tools

    def call_tool(self, name: str, arguments: dict | None = None) -> McpResult:
        arguments = arguments or {}
        self.calls.append((name, arguments))
        try:
            result = self._call("tools/call", {"name": name, "arguments": arguments}) or {}
        except McpError as e:
            return McpResult(text=str(e), is_error=True)
        return McpResult(
            text=content_to_text(result.get("content", [])),
            is_error=bool(result.get("isError", False)),
            raw=result,
        )


def content_to_text(content: list[dict]) -> str:
    """Flatten MCP content blocks to the string the model will read. Images become a note
    with the path if the server gave one; the agent can capture again if it needs pixels."""
    parts = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "image":
            parts.append("[image returned]")
        elif kind == "resource":
            res = block.get("resource", {})
            parts.append(res.get("text") or f"[resource {res.get('uri', '')}]")
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p)
