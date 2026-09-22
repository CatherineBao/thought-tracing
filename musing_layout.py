"""Directory layout for musing_out.

Runs used to land as one flat pile of ~200 files. The tree is now::

    musing_out/
      runs/<family>/   per-run artifacts: <run_id>.steps.jsonl, .log, _weights.html
      traces/          raw tracer-*.jsonl dumps (big, rarely read by hand)
      reports/         cross-run HTML
      meta/            runs.jsonl plus the small JSON sidecars

Filenames are unchanged, so every reader can keep its old pattern as long as it
globs recursively -- use `find()` rather than `glob.glob("musing_out/...")`.
"""

import glob
import os
import re
from typing import List, Optional

OUT_DIR = "musing_out"

RUNS = "runs"
TRACES = "traces"
REPORTS = "reports"
META = "meta"

# Run-id prefix -> family. Ordered: the first prefix that matches wins, so
# more specific prefixes come first.
_FAMILIES = (
    ("bloomfield",  ("t_bloom", "bloom_", "bl_")),
    ("atla",        ("t_atla", "atla_")),
    ("oppenheimer", ("t_strauss", "vs_")),
    ("ab",          ("ab_", "ab")),
    ("profiles",    ("prof_",)),
    ("tuning",      ("lpd", "lp", "sys", "rm", "exp_")),
    ("eval",        ("q_", "q", "fb_", "fb", "gc_", "gc", "demo",
                     "quality", "gate", "probe")),
    ("smoke",       ("smoke", "sm_", "hard_")),
    ("phase4",      ("p4",)),
    ("phase3",      ("p3",)),
    ("phase2",      ("p2",)),
    ("phase1",      ("p1",)),
    ("phase0",      ("phase0", "base", "lin")),
)

MISC = "misc"

# Trailing "-<corpus>-<role>-<seed>" that run_musing.py appends per context.
_CTX_SUFFIX = re.compile(r"-(?P<corpus>[A-Za-z0-9_]+)-(?:gold|silver|scene)-\d+$")


def base_run_id(run_id: str) -> str:
    """Strip the per-context suffix, e.g. lin-oppenheimer-silver-000 -> lin."""
    return _CTX_SUFFIX.sub("", run_id)


def corpus_of(run_id: str) -> Optional[str]:
    """The corpus run_musing.py baked into the run id, if it is there."""
    m = _CTX_SUFFIX.search(run_id)
    return m.group("corpus") if m else None


def family_for(run_id: str) -> str:
    """Which runs/<family> directory a run's artifacts belong in.

    A known prefix wins. Otherwise fall back to the corpus, so a run with a
    brand-new prefix still lands somewhere meaningful instead of in misc/ --
    the prefix table is a convenience, not something to keep in sync by hand.
    """
    stem = base_run_id(run_id)
    for family, prefixes in _FAMILIES:
        if stem.startswith(prefixes):
            return family
    return corpus_of(run_id) or MISC


def run_dir(run_id: str, out_dir: str = OUT_DIR, create: bool = False) -> str:
    path = os.path.join(out_dir, RUNS, family_for(run_id))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def run_path(run_id: str, filename: str, out_dir: str = OUT_DIR,
             create: bool = False) -> str:
    return os.path.join(run_dir(run_id, out_dir, create), filename)


def _sub(name: str, out_dir: str, create: bool) -> str:
    path = os.path.join(out_dir, name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def traces_dir(out_dir: str = OUT_DIR, create: bool = False) -> str:
    return _sub(TRACES, out_dir, create)


def reports_dir(out_dir: str = OUT_DIR, create: bool = False) -> str:
    return _sub(REPORTS, out_dir, create)


def meta_dir(out_dir: str = OUT_DIR, create: bool = False) -> str:
    return _sub(META, out_dir, create)


def report_path(filename: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(reports_dir(out_dir, create=True), filename)


def meta_path(filename: str, out_dir: str = OUT_DIR) -> str:
    return os.path.join(meta_dir(out_dir, create=True), filename)


def runs_jsonl(out_dir: str = OUT_DIR) -> str:
    return meta_path("runs.jsonl", out_dir)


def find(pattern: str, out_dir: str = OUT_DIR) -> List[str]:
    """Recursive glob for a bare filename pattern, sorted.

    Accepts a path that already includes the out_dir so callers can pass an
    argparse default through unchanged.
    """
    if os.path.isabs(pattern) or pattern.startswith(out_dir + os.sep):
        pattern = os.path.relpath(pattern, out_dir)
    hits = glob.glob(os.path.join(out_dir, "**", pattern), recursive=True)
    # A stray copy left at the top level shares its basename with the filed
    # one, and returning both would count a single run twice. The filed copy
    # wins, so a leftover is shadowed rather than double-counted.
    best = {}
    for path in sorted(set(hits)):
        name = os.path.basename(path)
        filed = os.path.dirname(path) != out_dir.rstrip(os.sep)
        if name not in best or (filed and not best[name][1]):
            best[name] = (path, filed)
    return sorted(p for p, _ in best.values())


def find_one(pattern: str, out_dir: str = OUT_DIR) -> Optional[str]:
    hits = find(pattern, out_dir)
    return hits[0] if hits else None
