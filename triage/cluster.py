"""Duplicate grouping, pass one: no model, no network.

Most duplicates are cheap to spot. Two crash reports with the same exception line are the
same crash. Two "Missing file: X" issues about the same file are one bug. Two titles that
differ by a version number or a word are usually one report filed twice. This pass finds
those with string similarity and hands the model a shortlist of candidate groups plus the
leftovers, so the model spends its attention on the hard calls, not on the obvious ones.

The output is a partition: every issue number appears in exactly one group.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from triage.issues import Issue, first_error_line

TITLE_THRESHOLD = 0.72  # SequenceMatcher ratio on normalized titles
TOKEN_THRESHOLD = 0.6  # Jaccard on title tokens, catches reordered words

_PREFIX = re.compile(r"^\[(bug|crash|feature|enhancement|question)\]\s*", re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9/._ -]+")
_WS = re.compile(r"\s+")
_STOP = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "of",
    "to",
    "is",
    "and",
    "when",
    "with",
    "for",
    "at",
    "it",
    "this",
    "that",
    "not",
    "does",
    "doesnt",
    "do",
    "after",
    "before",
    "from",
    "by",
}
_MISSING_FILE = re.compile(r"missing file:\s*(\S+)", re.IGNORECASE)


@dataclass
class Group:
    numbers: list[int]
    canonical_title: str
    reason: str = ""
    confidence: float = 1.0
    source: str = "heuristic"  # heuristic | model

    def key(self) -> tuple[int, ...]:
        return tuple(sorted(self.numbers))


@dataclass
class Partition:
    groups: list[Group] = field(default_factory=list)

    def duplicates(self) -> list[Group]:
        return [g for g in self.groups if len(g.numbers) > 1]

    def numbers(self) -> set[int]:
        return {n for g in self.groups for n in g.numbers}

    def pairs(self) -> set[tuple[int, int]]:
        """Every unordered pair that shares a group. The eval scores partitions on these."""
        out: set[tuple[int, int]] = set()
        for g in self.groups:
            ns = sorted(g.numbers)
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    out.add((ns[i], ns[j]))
        return out


def normalize_title(title: str) -> str:
    t = _PREFIX.sub("", title).lower()
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"\bv?\d+(\.\d+)+\b", "", t)  # version numbers
    return _WS.sub(" ", t).strip()


def tokens(title: str) -> set[str]:
    return {w for w in normalize_title(title).split() if w not in _STOP and len(w) > 1}


def signature(issue: Issue) -> str:
    """A string two issues share only if they are almost certainly the same bug.
    Empty when there is nothing that strong to go on."""
    if issue.is_crash:
        line = first_error_line(issue)
        if line:
            return "crash:" + line
    m = _MISSING_FILE.search(issue.title) or _MISSING_FILE.search(issue.body[:400])
    if m:
        # Two reports of one missing file quote it at different depths ("TEGE/Engine/shaders/x"
        # from the repo root, "Engine/shaders/x" from the engine). Compare the file name.
        path = m.group(1).replace("\\", "/").lower().strip(".,;:`'\"")
        return "missing:" + path.rsplit("/", 1)[-1]
    return ""


def similarity(a: Issue, b: Issue) -> float:
    sa, sb = signature(a), signature(b)
    if sa and sb:
        return 1.0 if sa == sb else 0.0
    ta, tb = normalize_title(a.title), normalize_title(b.title)
    if not ta or not tb:
        return 0.0
    ratio = SequenceMatcher(None, ta, tb).ratio()
    ka, kb = tokens(a.title), tokens(b.title)
    jacc = len(ka & kb) / len(ka | kb) if (ka or kb) else 0.0
    # A shared first error line in the console tail is strong corroboration for bug reports.
    ea, eb = first_error_line(a), first_error_line(b)
    if ea and ea == eb:
        return max(ratio, jacc, 0.9)
    return max(ratio, jacc if jacc >= TOKEN_THRESHOLD else 0.0)


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def heuristic_partition(issues: list[Issue], threshold: float = TITLE_THRESHOLD) -> Partition:
    by_num = {i.number: i for i in issues}
    uf = _UnionFind(by_num)
    nums = sorted(by_num)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if similarity(by_num[nums[i]], by_num[nums[j]]) >= threshold:
                uf.union(nums[i], nums[j])
    buckets: dict[int, list[int]] = {}
    for n in nums:
        buckets.setdefault(uf.find(n), []).append(n)
    groups = []
    for members in buckets.values():
        members.sort()
        lead = by_num[members[0]]
        reason = ""
        if len(members) > 1:
            sig = signature(lead)
            reason = f"shared {sig.split(':')[0]} signature" if sig else "near-identical titles"
        groups.append(
            Group(
                numbers=members,
                canonical_title=lead.title,
                reason=reason,
                confidence=1.0 if len(members) == 1 else 0.8,
            )
        )
    groups.sort(key=lambda g: g.numbers[0])
    return Partition(groups)


def validate_partition(p: Partition, issues: list[Issue]) -> list[str]:
    """Problems with a partition, as human sentences. Empty means it is a real partition of
    exactly these issues. Used to reject a model answer that dropped or invented numbers."""
    want = {i.number for i in issues}
    seen: list[int] = [n for g in p.groups for n in g.numbers]
    problems = []
    dup = {n for n in seen if seen.count(n) > 1}
    if dup:
        problems.append(f"issues in more than one group: {sorted(dup)}")
    missing = want - set(seen)
    if missing:
        problems.append(f"issues left out: {sorted(missing)}")
    extra = set(seen) - want
    if extra:
        problems.append(f"issues that do not exist: {sorted(extra)}")
    return problems
