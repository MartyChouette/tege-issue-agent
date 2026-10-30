import os

from triage.cluster import (
    Group,
    Partition,
    heuristic_partition,
    normalize_title,
    signature,
    similarity,
    validate_partition,
)
from triage.evaluate import load_eval, score
from triage.issues import Issue, load_issues_from_files

EVAL = os.path.join(os.path.dirname(__file__), "..", "eval", "issues.json")


def _issues():
    return {i.number: i for i in load_issues_from_files([EVAL])}


def test_same_crash_different_address_is_one_signature():
    i = _issues()
    assert signature(i[101]) == signature(i[102])
    assert signature(i[101]) != signature(i[103])
    assert similarity(i[101], i[102]) == 1.0
    assert similarity(i[101], i[103]) == 0.0


def test_missing_file_signature_uses_the_file_name():
    i = _issues()
    assert signature(i[104]) == "missing:oit_accum.frag"
    assert signature(i[105]) == "missing:oit_accum.frag"
    assert signature(i[106]) == "missing:ui_font.ttf"


def test_title_normalization_strips_prefix_and_versions():
    assert (
        normalize_title("[Bug] Level streaming pops tiles late (Linux, v0.9.2)")
        == "level streaming pops tiles late linux"
    )


def test_shared_console_error_corroborates_different_titles():
    i = _issues()
    assert similarity(i[113], i[114]) >= 0.9


def test_different_symptom_same_system_stays_apart():
    i = _issues()
    assert similarity(i[110], i[112]) < 0.72


def test_heuristic_partition_is_a_partition():
    issues = list(_issues().values())
    p = heuristic_partition(issues)
    assert validate_partition(p, issues) == []
    assert p.numbers() == {i.number for i in issues}


def test_heuristic_catches_the_easy_duplicates_and_misses_the_semantic_one():
    issues, expected = load_eval()
    p = heuristic_partition(issues)
    keys = {g.key() for g in p.groups}
    assert (101, 102) in keys
    assert (104, 105) in keys
    assert (110, 111) in keys
    assert (113, 114) in keys
    # Shares no title words with its twin. Only a model reading the text can merge it.
    assert (107, 108) not in keys
    s = score(p, expected)
    assert s.precision == 1.0
    assert 0.75 <= s.recall < 1.0


def test_validate_partition_reports_each_kind_of_problem():
    issues = [Issue(number=n, title=f"t{n}", body="") for n in (1, 2, 3)]
    p = Partition([Group([1, 2], "a"), Group([2, 9], "b")])
    problems = validate_partition(p, issues)
    assert any("more than one group" in x for x in problems)
    assert any("left out: [3]" in x for x in problems)
    assert any("do not exist: [9]" in x for x in problems)
