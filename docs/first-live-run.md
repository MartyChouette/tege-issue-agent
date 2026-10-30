# First live run, 2026-10-30

The first run against real issues and the running editor, with a local model.

- Model: qwen3:8b through Ollama, on an 8 GB laptop GPU. Nothing left the machine.
- Editor: TEGE 0.9.8, MCP server on, 36 tools listed, write tools off.
- Issues: the 2 open issues on MartyChouette/TEGE. The model pass confirmed they are not
  duplicates. 9 tool calls, about 22k input and 6k output tokens.

What it got right: every quoted line of evidence is real text from the issue it was reading,
and it stayed inside the read-only tool set.

What it could not do: no project was open in the editor, so the script and scene tools came
back empty or with errors, and both proposals are thin. #18 was marked "needs more
information" and named the right file. #19 was already fixed on main before this run
(f26a8a70, 2026-09-24): `oit_accum.frag.spv` is `triangle.frag` compiled with `-DENJIN_OIT`,
and nothing produced it until that commit added a CMake rule. The model's re-clone advice
would have picked up the fix, but it never checked commit history, so it could not say the
issue was already resolved. Checking recent commits against an issue's date belongs in the
agent's tools.

Next run: open a project in the editor first, and compare qwen3:8b against a larger model on
the same two issues.

The full report is below, unedited.

---

# Triage report: MartyChouette/TEGE

Run at 2026-10-30 06:21 UTC. 2 open issues, 0 duplicate groups, 2 issues standing alone.

| | |
|---|---|
| Model | ollama / qwen3:8b |
| Prompt version | 2026-10-14.1 |
| Grouping | model-confirmed |
| Editor | connected over MCP |
| Write tools | off |
| Tokens | input tokens 22327, output tokens 6347 |

## Issues standing alone

### #18 Games cannot set their own HUD font: UISystem is locked to the editor's face

**Needs more information**, confidence 0.80

The project is not open, preventing inspection of logs, scripts, or scene data. The root cause remains the UISystem's hardcoded ImGui::GetFont() calls without font override support.

Cause: UISystem.cpp uses ImGui::GetFont() directly multiple times without font configuration or inheritance mechanisms.

Proposed fix: Add UITheme font reference, scene-load font atlas integration, and UI editor field for persistent project font settings as described.

Files: `Engine/src/GUI/UISystem.cpp`

<details><summary>3 tool calls</summary>

- `script_errors()` error, 22 chars
- `get_log(min_level='error', lines=50)` ok, 26 chars
- `scene_info()` ok, 56 chars

</details>

### #19 Missing file: TEGE/Engine/shaders/oit_accum.frag

**Engine-side change**, confidence 0.95

Missing shader file oit_accum.frag.spv during build

Cause: The build process fails because the required shader file oit_accum.frag.spv is missing from the Engine/shaders directory. This could be due to an incomplete repository clone, a build configuration error, or a missing generation step.

Evidence:
- `FileNotFoundError: '/home/martin/Downloads/TEGE/Engine/shaders/oit_accum.frag.spv'`

Proposed fix: Verify the file exists in the repository, ensure all submodules are cloned, and check build scripts for shader generation steps. If missing, re-clone the repo and rebuild.

<details><summary>6 tool calls</summary>

- `script_list()` error, 22 chars
- `get_log(lines=20, min_level='error')` ok, 26 chars
- `scene_info()` ok, 56 chars
- `script_errors()` error, 22 chars
- `list_scenes()` ok, 53 chars
- `registered_component_keys()` ok, 2764 chars

</details>
