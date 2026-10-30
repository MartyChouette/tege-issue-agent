# enjin-triage

The TEGE editor files its own bug reports. A person hits "report a bug" in the feedback
panel, or the crash handler catches an exception, and a GitHub issue appears with the engine
version, platform, GPU, scene path and the last lines of the console log. That is a good
stream of data and a bad reading experience: the same crash arrives ten times with ten
different pointer addresses in the title, and a maintainer spends the first hour of every
session deciding what is actually new.

This tool reads that stream and does the first hour. It groups the duplicates, then opens
each group in the running editor over the engine's MCP server, reads the log and the compile
diagnostics, and writes up what it found and what it would change. The output is a markdown
report. Posting back to GitHub is a flag, off by default.

```
enjin-triage run                    # fetch, group, investigate, write reports/triage-<date>.md
enjin-triage group                  # grouping only, no editor needed
enjin-triage tools                  # what the editor exposes and what the agent may touch
enjin-triage eval --verbose         # score the grouping heuristic on the labeled set
```

## How it decides two issues are one bug

Two passes. The first has no model in it. Crash reports are keyed on their exception line
with addresses and counters stripped, so `descriptor set 0x2a7f1c40 was destroyed` and
`descriptor set 0x5be0a900 was destroyed` are the same key. "Missing file" reports are keyed
on the file name. Everything else is compared on normalized titles, with a shared first
error line in the console tail counting as corroboration. Union-find turns the pairs into
groups.

The second pass hands those groups to the model as candidates, along with every issue in a
compact form, and asks for the final partition as JSON against a schema. The code then
checks that the answer is a partition of exactly the issues it was given. If a number is
missing, doubled or invented, the model's answer is thrown away and the first pass stands.
A wrong merge closes a real bug as a duplicate, so the heuristic wins ties.

The labeled set in `eval/` scores both passes on pairs, the usual way to compare clusterings.
The heuristic alone sits at precision 1.00 and recall 0.80 on it. The one pair it misses,
#107 and #108, shares no words at all: one is a user saying the HUD font is stuck, the other
is a maintainer note that `UISystem` reads its face from the editor theme. Only something
that reads the two bodies can merge them, which is the case the model pass exists for. CI
fails if the heuristic drops below 0.85 F1.

## How it investigates

The editor's MCP server (`docs/MCP.md` in the engine) is a JSON-RPC endpoint on localhost
with about thirty tools: read and write components, list and read AngelScript sources,
compile a script and get diagnostics back, tail the engine log, control play mode, start a
build. The tool here connects to it with a forty-line client, lists the tools, and offers
them to the model as function calls.

Without `--allow-write`, only the reading tools are offered: `get_log`, `script_errors`,
`script_read`, `get_component`, `scene_info` and a few more. The agent cannot change the
scene or a script or kick off a build by accident. With the flag, it can write a script and
compile it, which is the honest way to test a fix, and the proposal has to say what it
changed.

The loop runs until the model calls `propose_fix`, a local tool with a strict schema:
status, summary, root cause, evidence quoted from tool results, the proposed change, the
files involved, a confidence, and whether this is really a duplicate of some other issue.
There is a step budget (twelve tool calls by default). When it runs out, the model gets one
more turn with only `propose_fix` on the table. If the model ends in prose instead, one
extra call with no tools turns the transcript into the schema. Every tool call the agent
made is listed in the report, collapsed, so a reviewer can see what the finding rests on.

## Models

Claude is the default (`claude-opus-5-5`, adaptive thinking, effort `medium` for grouping
and `high` inside the loop, prompt caching on the system text, server-side refusal
fallback on). It needs `ANTHROPIC_API_KEY` in the environment.

Ollama is the no-account path. Run any model that supports tool calling and pass
`--provider ollama`. The default model id is `qwen3:14b`; `OLLAMA_HOST` points at the
server. The compose file brings one up.

`--provider fake` runs the whole pipeline with no model at all: heuristic grouping, a
scripted proposal per group, a real report. It is what the tests use and a good first run.

Both real backends sit behind the same small interface, and the conversions to each wire
format are plain functions with tests, so adding a third backend is one class.

## Running it

```
pip install -e .[dev]
pytest
enjin-triage eval --verbose
enjin-triage run --provider fake --from-files eval/issues.json
```

Against the real repo, with the editor open and Settings > System > MCP Server turned on:

```
export ANTHROPIC_API_KEY=...        # or: --provider ollama
enjin-triage tools                  # confirms the editor is reachable
enjin-triage run --only 19          # one issue, read-only
enjin-triage run                    # everything open
enjin-triage run --post             # asks before posting; --yes skips the prompt
```

GitHub reads go through the `gh` CLI's token if you are logged in, or `GITHUB_TOKEN`.
Public repos need neither for reading.

In Docker:

```
docker compose --profile setup run --rm ollama-pull
docker compose --profile run run --rm triage run --provider ollama --no-investigate
```

The editor stays on the host, since its server binds `127.0.0.1`. The container reaches it
as `host.docker.internal`.

## Prompts

All of them are in `triage/prompts.py` with a version string that is printed in every
report. The grouping prompt says what counts as a duplicate in this codebase (same exception
from two scenes, yes; two different missing files, no). The investigation prompt tells the
model to behave like a maintainer with the editor open: read the report, decide what would
confirm the cause, look at that and only that, quote the exact line it is relying on, and
say "not confirmed" when that is the truth. Changing a prompt means bumping the version and
re-running the eval.

## What is not here yet

- The eval set is fifteen hand-written issues in the editor's format. It should grow from
  real triage decisions on the live repo as they happen.
- The investigation step has no eval. Scoring "was the proposed fix right" needs a set of
  issues with known fixes, which the engine's history can supply.
- Scheduling. Right now it is a command you run. A nightly run on the open issues with the
  report dropped into a discussion thread is the natural next step.

## Layout

```
triage/
  cli.py        subcommands, flags, the --post confirmation
  config.py     settings, read-only tool allowlist
  issues.py     GitHub fetch, editor-report markdown parser
  cluster.py    heuristic grouping (signatures, title similarity, union-find)
  prompts.py    every prompt and schema, versioned
  llm.py        provider interface: Claude, Ollama, scripted fake
  mcp.py        JSON-RPC over HTTP client for the editor
  agent.py      grouping confirmation and the investigation loop
  report.py     markdown report and issue comment
  evaluate.py   pairwise scoring against eval/
eval/           labeled issues and expected groups
tests/          everything above, offline
```

MIT license. Built by marty scott alongside [TEGE](https://github.com/MartyChouette/TEGE).
