"""The MCP client against a stand-in for the editor: a tiny JSON-RPC HTTP server that
speaks the same three methods the editor does."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from triage.mcp import McpClient, McpError, content_to_text

TOOLS = [
    {
        "name": "script_errors",
        "description": "Last compile error",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_log",
        "description": "Tail the log",
        "inputSchema": {"type": "object", "properties": {"level": {"type": "string"}}},
    },
]


class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):  # keep pytest output quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        _Handler.seen.append(req)
        method = req.get("method")
        if "id" not in req:  # notification
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake-tege", "version": "0.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = req["params"]["name"]
            if name == "get_log":
                result = {
                    "content": [{"type": "text", "text": "[ERROR] [Renderer] device lost"}],
                    "isError": False,
                }
            elif name == "explode":
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": req["id"],
                        "error": {"code": -32601, "message": "unknown tool"},
                    }
                )
                return
            else:
                result = {"content": [{"type": "text", "text": "no errors"}]}
        else:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "error": {"code": -32601, "message": "method not found"},
                }
            )
            return
        self._send({"jsonrpc": "2.0", "id": req["id"], "result": result})

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/mcp"
    httpd.shutdown()


def test_handshake_then_tools_then_call(server):
    c = McpClient(server)
    info = c.initialize()
    assert info["serverInfo"]["name"] == "fake-tege"
    # initialize, then the initialized notification with no id
    assert _Handler.seen[0]["method"] == "initialize"
    assert (
        _Handler.seen[1]["method"] == "notifications/initialized" and "id" not in _Handler.seen[1]
    )

    tools = c.list_tools()
    assert [t.name for t in tools] == ["get_log", "script_errors"]  # sorted
    res = c.call_tool("get_log", {"level": "error"})
    assert res.text == "[ERROR] [Renderer] device lost" and not res.is_error
    assert c.calls == [("get_log", {"level": "error"})]


def test_jsonrpc_error_on_call_becomes_error_result_not_exception(server):
    c = McpClient(server)
    res = c.call_tool("explode")
    assert res.is_error and "unknown tool" in res.text


def test_unreachable_server_gives_a_useful_message():
    c = McpClient("http://127.0.0.1:1/mcp", timeout=0.5)
    with pytest.raises(McpError) as e:
        c.initialize()
    assert "MCP Server" in str(e.value)


def test_content_flattening():
    assert (
        content_to_text(
            [
                {"type": "text", "text": "a"},
                {"type": "image", "data": "..."},
                {"type": "resource", "resource": {"uri": "file:///x"}},
            ]
        )
        == "a\n[image returned]\n[resource file:///x]"
    )
