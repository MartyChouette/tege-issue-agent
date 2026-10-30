"""Score the duplicate grouping against a labeled set.

A partition is scored on pairs: for every two issues that the labels say are the same bug,
did we put them together (recall), and for every pair we put together, were they really the
same bug (precision). Pair scoring is the standard way to compare clusterings when group
sizes vary, and it gives partial credit for a group that is mostly right.

The labeled set lives in eval/issues.json (GitHub API shape, written in the editor's own
report format) and eval/expected.json. The heuristic pass is scored on every run of the test
suite with no network. The model pass is scored on demand, since it costs money.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from triage.cluster import Partition, heuristic_partition
from triage.issues import Issue, load_issues_from_files

HERE = os.path.dirname(os.path.abspath(__file__))
# The labeled set lives next to the source tree, not inside the package, so an installed
# copy (pip install ., the Docker image) has to be told where it is.
EVAL_DIR = os.environ.get("TRIAGE_EVAL_DIR") or os.path.normpath(os.path.join(HERE, "..", "eval"))


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    exact_groups: int  # predicted groups that match a labeled group exactly
    total_groups: int
    wrong_pairs: list[tuple[int, int]]
    missed_pairs: list[tuple[int, int]]

    def line(self, label: str) -> str:
        return (
            f"{label:<12} precision {self.precision:.2f}  recall {self.recall:.2f}  "
            f"f1 {self.f1:.2f}  exact groups {self.exact_groups}/{self.total_groups}"
        )


def load_eval(eval_dir: str = EVAL_DIR) -> tuple[list[Issue], Partition]:
    issues = load_issues_from_files([os.path.join(eval_dir, "issues.json")])
    with open(os.path.join(eval_dir, "expected.json"), encoding="utf-8") as f:
        exp = json.load(f)
    from triage.cluster import Group

    groups = [Group(numbers=sorted(g), canonical_title="", source="label") for g in exp["groups"]]
    return issues, Partition(groups)


def score(predicted: Partition, expected: Partition) -> Score:
    pp, ep = predicted.pairs(), expected.pairs()
    tp = len(pp & ep)
    precision = tp / len(pp) if pp else 1.0
    recall = tp / len(ep) if ep else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exp_keys = {g.key() for g in expected.groups}
    exact = sum(1 for g in predicted.groups if g.key() in exp_keys)
    return Score(
        precision,
        recall,
        f1,
        exact,
        len(expected.groups),
        wrong_pairs=sorted(pp - ep),
        missed_pairs=sorted(ep - pp),
    )


def score_heuristic(eval_dir: str = EVAL_DIR) -> Score:
    issues, expected = load_eval(eval_dir)
    return score(heuristic_partition(issues), expected)
