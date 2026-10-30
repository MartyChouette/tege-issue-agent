"""Runtime settings. Everything comes from flags or the environment, nothing is hardcoded
to one machine. Secrets never land in this file or in logs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_REPO = "MartyChouette/TEGE"
DEFAULT_MCP_URL = "http://127.0.0.1:8971/mcp"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# MCP tools the agent may call without --allow-write. Everything here only reads state or
# writes a PNG to disk. The write set is whatever tools/list returns minus this list.
READ_ONLY_TOOLS = frozenset(
    {
        "scene_info",
        "list_entities",
        "find_entity",
        "get_component",
        "registered_component_keys",
        "script_list",
        "script_read",
        "script_errors",
        "get_log",
        "list_scenes",
        "editor_viewport_info",
        "capture_view",
        "editor_capture",
    }
)


@dataclass
class Settings:
    repo: str = DEFAULT_REPO
    provider: str = "claude"  # claude | ollama | fake
    model: str | None = None  # None picks the provider default
    effort: str = "medium"  # Claude effort for the grouping pass
    agent_effort: str = "high"  # Claude effort for the investigation loop
    mcp_url: str = DEFAULT_MCP_URL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    allow_write: bool = False  # unlock script_write, set_component, build_game and friends
    max_steps: int = 12  # tool calls per investigation before the agent must conclude
    post: bool = False  # post the report back to GitHub as issue comments
    since: str | None = None  # ISO date; only issues updated after this
    limit: int = 50
    out_dir: str = "reports"
    fallbacks: bool = True  # Claude server-side refusal fallback
    github_token: str | None = field(default=None, repr=False)

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "ollama":
            return DEFAULT_OLLAMA_MODEL
        return DEFAULT_CLAUDE_MODEL

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        s = cls(
            repo=os.environ.get("TRIAGE_REPO", DEFAULT_REPO),
            provider=os.environ.get("TRIAGE_PROVIDER", "claude"),
            mcp_url=os.environ.get("TEGE_MCP_URL", DEFAULT_MCP_URL),
            ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            github_token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        return s
