import os

from triage.issues import (
    Issue,
    first_error_line,
    issue_from_api,
    load_issues_from_files,
    normalize_crash_line,
    parse_body,
)

EVAL = os.path.join(os.path.dirname(__file__), "..", "eval", "issues.json")


def test_editor_bug_report_parses_every_section():
    issues = {i.number: i for i in load_issues_from_files([EVAL])}
    i = issues[109]
    assert i.report_type == "Editor"
    assert i.severity == "High"
    assert i.description.startswith("New scene, press play")
    assert "File > New Scene" in i.steps
    assert i.diagnostics.engine == "0.9.2"
    assert i.diagnostics.platform == "Windows 11"
    assert i.diagnostics.extra["Entities"] == "0"
    assert len(i.log_tail) == 3
    assert first_error_line(i).startswith("[error] [scripting]")


def test_diagnostics_table_keeps_gpu_and_scene():
    issues = {i.number: i for i in load_issues_from_files([EVAL])}
    i = issues[107]
    assert i.diagnostics.gpu == "NVIDIA GeForce RTX 3060"
    assert i.diagnostics.scene == "scenes/hud_test.enjin"
    assert i.expected == "The font I picked."


def test_labels_fill_type_when_body_has_none():
    i = issue_from_api(
        {
            "number": 5,
            "title": "[Crash] boom",
            "body": "",
            "labels": [{"name": "type:Crash"}, {"name": "severity:Critical"}],
        }
    )
    assert i.report_type == "Crash"
    assert i.severity == "Critical"
    assert i.is_crash


def test_hand_written_issue_parses_to_empty_fields():
    i = parse_body(Issue(number=1, title="plain", body="just some words"))
    assert i.description == "" and i.steps == "" and i.log_tail == []
    assert first_error_line(i) == ""


def test_crash_line_normalization_drops_addresses_and_counts():
    a = normalize_crash_line("descriptor set 0x2a7f1c40 was destroyed (frame 118)")
    b = normalize_crash_line("descriptor set 0x5BE0A900 was destroyed (frame 9)")
    assert a == b == "descriptor set 0x? was destroyed (frame #)"


def test_short_view_is_compact_and_carries_the_error():
    issues = {i.number: i for i in load_issues_from_files([EVAL])}
    s = issues[113].short()
    assert s.startswith("#113 ")
    assert "first error:" in s and "device_lost" in s
    assert "| Metric |" not in s
