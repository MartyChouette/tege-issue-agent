"""Command line entry point.

    tege-issue-agent run            fetch issues, group, investigate each group, write a report
    tege-issue-agent group          fetch and group only (no editor needed)
    tege-issue-agent tools          show what the editor's MCP server exposes and what is allowed
    tege-issue-agent eval           score the grouping heuristic (and optionally a model) offline

Nothing is posted anywhere unless --post is given. The report always lands in reports/.
"""

from __future__ import annotations

import argparse
import sys

from triage.config import Settings
from triage.issues import fetch_issues, load_issues_from_files, post_comment


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", help="owner/name (default MartyChouette/TEGE or $TRIAGE_REPO)")
    p.add_argument(
        "--provider",
        choices=["claude", "ollama", "fake"],
        help="model backend (default claude or $TRIAGE_PROVIDER)",
    )
    p.add_argument("--model", help="model id; defaults per provider")
    p.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort for grouping (default medium)",
    )
    p.add_argument("--since", help="only issues updated after this ISO date")
    p.add_argument("--limit", type=int, help="max issues to fetch (default 50)")
    p.add_argument(
        "--from-files",
        nargs="+",
        metavar="JSON",
        help="read issues from GitHub API JSON files instead of the network",
    )
    p.add_argument(
        "--no-fallbacks", action="store_true", help="disable Claude's server-side refusal fallback"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tege-issue-agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="group and investigate, write a report")
    _common(run)
    run.add_argument("--mcp-url", help="editor MCP endpoint (default http://127.0.0.1:8971/mcp)")
    run.add_argument(
        "--allow-write",
        action="store_true",
        help="let the agent use script_write, set_component, build_game and friends",
    )
    run.add_argument("--max-steps", type=int, help="tool calls per investigation (default 12)")
    run.add_argument(
        "--agent-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort inside the investigation loop (default high)",
    )
    run.add_argument(
        "--only",
        type=int,
        nargs="+",
        metavar="N",
        help="investigate only the groups containing these issue numbers",
    )
    run.add_argument("--no-investigate", action="store_true", help="grouping only")
    run.add_argument(
        "--post",
        action="store_true",
        help="post each proposal as a comment on its issues (asks first)",
    )
    run.add_argument("--yes", action="store_true", help="skip the confirmation for --post")
    run.add_argument("--out", help="report directory (default reports/)")

    grp = sub.add_parser("group", help="group duplicates only")
    _common(grp)

    tl = sub.add_parser("tools", help="list the editor's MCP tools")
    tl.add_argument("--mcp-url")
    tl.add_argument("--allow-write", action="store_true")

    ev = sub.add_parser("eval", help="score duplicate grouping on the labeled set")
    ev.add_argument(
        "--provider",
        choices=["claude", "ollama"],
        help="also score the model pass (costs a request)",
    )
    ev.add_argument("--model")
    ev.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ev.add_argument(
        "--min-f1",
        type=float,
        default=0.0,
        help="exit 1 if the heuristic f1 falls below this (for CI)",
    )
    ev.add_argument("--verbose", action="store_true")
    return ap


def settings_from(args: argparse.Namespace) -> Settings:
    keys = (
        "repo",
        "provider",
        "model",
        "effort",
        "agent_effort",
        "mcp_url",
        "allow_write",
        "max_steps",
        "post",
        "since",
        "limit",
        "out",
    )
    over = {k: getattr(args, k, None) for k in keys}
    if over.get("out"):
        over["out_dir"] = over.pop("out")
    else:
        over.pop("out", None)
    for flag in ("allow_write", "post"):
        if over.get(flag) is False:
            over[flag] = None  # do not override the default with an explicit False
    s = Settings.from_env(**over)
    if getattr(args, "no_fallbacks", False):
        s.fallbacks = False
    return s


def _load_issues(args, s: Settings):
    if getattr(args, "from_files", None):
        return load_issues_from_files(args.from_files)
    print(f"fetching open issues from {s.repo} ...", file=sys.stderr)
    return fetch_issues(s.repo, token=s.github_token, since=s.since, limit=s.limit)


def cmd_group(args) -> int:
    from triage.agent import group_issues
    from triage.llm import make_provider

    s = settings_from(args)
    issues = _load_issues(args, s)
    provider = None if s.provider == "fake" else make_provider(s)
    result = group_issues(issues, provider, s)
    by_num = {i.number: i for i in issues}
    for g in result.partition.groups:
        mark = "DUP " if len(g.numbers) > 1 else "    "
        print(f"{mark}{g.numbers}  {g.canonical_title or by_num[g.numbers[0]].title}")
        if len(g.numbers) > 1 and g.reason:
            print(f"      {g.reason} ({g.confidence:.2f}, {g.source})")
    for n in result.notes:
        print(f"note: {n}", file=sys.stderr)
    return 0


def cmd_tools(args) -> int:
    from triage.agent import allowed_tools
    from triage.mcp import McpClient, McpError

    url = args.mcp_url or Settings.from_env().mcp_url
    mcp = McpClient(url)
    try:
        info = mcp.initialize()
        tools = mcp.list_tools()
    except McpError as e:
        print(str(e), file=sys.stderr)
        return 2
    server = info.get("serverInfo", {})
    print(f"{server.get('name', 'server')} {server.get('version', '')} at {url}")
    allowed = {t.name for t in allowed_tools(tools, args.allow_write)}
    for t in tools:
        print(f"  {'+' if t.name in allowed else '-'} {t.name:<28} {t.description[:70]}")
    print("\n+ allowed in this mode, - blocked (pass --allow-write to unlock)")
    return 0


def cmd_eval(args) -> int:
    from triage.evaluate import load_eval, score, score_heuristic

    h = score_heuristic()
    print(h.line("heuristic"))
    if args.verbose:
        for a, b in h.wrong_pairs:
            print(f"  wrong pair   #{a} #{b}")
        for a, b in h.missed_pairs:
            print(f"  missed pair  #{a} #{b}")
    if args.provider:
        from triage.agent import group_issues
        from triage.llm import make_provider

        s = Settings.from_env(provider=args.provider, model=args.model, effort=args.effort)
        issues, expected = load_eval()
        result = group_issues(issues, make_provider(s), s)
        m = score(result.partition, expected)
        print(m.line(args.provider))
        for n in result.notes:
            print(f"  note: {n}")
        if args.verbose:
            for a, b in m.wrong_pairs:
                print(f"  wrong pair   #{a} #{b}")
            for a, b in m.missed_pairs:
                print(f"  missed pair  #{a} #{b}")
    return 1 if h.f1 < args.min_f1 else 0


def cmd_run(args) -> int:
    from triage.agent import demo_reply_for_fake, group_issues, investigate
    from triage.llm import FakeProvider, make_provider
    from triage.mcp import McpClient, McpError
    from triage.report import comment_body, render_report, write_report

    s = settings_from(args)
    issues = _load_issues(args, s)
    if not issues:
        print("no open issues", file=sys.stderr)
        return 0

    provider = None if s.provider == "fake" else make_provider(s)
    grouping = group_issues(issues, provider, s)
    print(
        f"{len(issues)} issues, {len(grouping.partition.duplicates())} duplicate groups",
        file=sys.stderr,
    )

    mcp = None
    mcp_tools = []
    if not args.no_investigate:
        mcp = McpClient(s.mcp_url)
        try:
            mcp.initialize()
            mcp_tools = mcp.list_tools()
            print(f"editor connected: {len(mcp_tools)} MCP tools", file=sys.stderr)
        except McpError as e:
            print(f"no editor: {e}", file=sys.stderr)
            print("investigating from the reports alone; findings will be limited", file=sys.stderr)
            mcp = None

    proposals = {}
    usage: dict = {}
    if not args.no_investigate:
        groups = grouping.partition.groups
        if args.only:
            want = set(args.only)
            groups = [g for g in groups if want & set(g.numbers)]
        for g in groups:
            p = FakeProvider([demo_reply_for_fake(g)]) if provider is None else provider
            print(f"investigating {g.numbers} ...", file=sys.stderr)
            proposals[g.key()] = investigate(g, issues, p, mcp, mcp_tools, s)
        if provider is not None:
            usage = dict(getattr(provider, "usage", {}))
    for k, v in grouping.usage.items():
        if isinstance(v, (int, float)):
            usage[k] = usage.get(k, 0) + v

    model = provider.model if provider else "none (dry run)"
    name = provider.name if provider else "fake"
    text = render_report(issues, grouping, proposals, s, name, model, usage, mcp is not None)
    path = write_report(text, s.out_dir)
    print(f"report: {path}", file=sys.stderr)

    if args.post and proposals:
        n = sum(len(k) for k in proposals)
        if not args.yes:
            ans = input(
                f"post {len(proposals)} proposals as comments on {n} issues in {s.repo}? [y/N] "
            )
            if ans.strip().lower() not in ("y", "yes"):
                print("not posted", file=sys.stderr)
                return 0
        by_key = {g.key(): g for g in grouping.partition.groups}
        for key, prop in proposals.items():
            body = comment_body(by_key[key], prop, model)
            for num in key:
                url = post_comment(s.repo, num, body, token=s.github_token)
                print(f"posted on #{num}: {url}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"run": cmd_run, "group": cmd_group, "tools": cmd_tools, "eval": cmd_eval}[args.cmd](
            args
        )
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # the message is the point; a trace is noise for a CLI user
        from triage.llm import ProviderError, RefusalError
        from triage.mcp import McpError

        if isinstance(e, (ProviderError, RefusalError, McpError)):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise
