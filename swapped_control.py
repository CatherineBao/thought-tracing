"""The control arm almost every phase needs: the same step, a different person.

    from swapped_control import ControlIndex
    idx = ControlIndex()
    ctl = idx.swapped(run_id="bb_Rodriguez-bloomfield-silver-000", step=7)
    ctl.particles      # a slate traced for somebody else on the same 6 sets
    ctl.why            # how it was matched, and what it is NOT comparable on

    python swapped_control.py --list            # which runs have a control at all
    python swapped_control.py --run <id> --step 7

WHAT IT IS FOR. Every design in PRECHECKS that came back null came back null
against a control, and the ones that died died because the control moved with
the test arm: the transcript-side detector scored 47% on the test dyad and 48%
on the control, and three "culture detectors" were retired for raising both
arms together. A swapped-person slate is the cheapest control of that shape --
identical corpus, identical span, identical configuration, identical step, one
thing different -- so a score that does not separate the two was never reading
the person.

WHY MATCHING IS STRICT AND FAILURE IS LOUD. A control that is a bit off on the
span, or a bit off on the configuration, is worse than no control: it turns a
null result into an ambiguous one, and PRECHECKS has two of those already
("routed around twice and both times the result was uninterpretable"). So the
tiers below are explicit, every returned control says which tier it came from,
and a request that cannot be matched returns a refusal carrying the reason
rather than the nearest run.

  tier 1  same span (identical set_ids), same config, same ARM -- the run-id
          stem with the target's own name taken out of it, so cov1s0_Lyubovsky
          and cov1s0_Rovani are one arm and v3_bb_Rodriguez is not
  tier 2  same span, same config
  tier 3  same corpus, span Jaccard >= --min-span-overlap, same config

STEP ALIGNMENT IS NOT FREE. A run's steps are the TARGET's own turns, so two
people on one span do not have the same number of steps -- 41 against 37 on the
atla span, 15 against 6 on bloomfield-0329..0334. Index alignment is the
default because a control should not be interpolated; where a run is shorter
the request simply has no control at that step and says so. `align="fraction"`
is offered for the case where position in the run is the thing being compared,
and it records the step it actually landed on.

CONFIGURATION IS PART OF THE MATCH, AND SO IS THE SPLIT. Two runs that differ
in n_hypotheses do not have comparable slates -- rank is bounded by population
size. And a control drawn from the other side of splits.json would put test
text inside a dev measurement, so the side is checked and a crossing match is
refused.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import musing_layout as ml

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits.json")

# The fields that must agree before two runs' slates are comparable at all.
# alpha and seed are matched where both runs record them: many early runs
# predate the alpha field, and requiring a value nobody logged would refuse
# every control on the corpus the project cares most about.
CONFIG_KEYS = ("n_hypotheses", "scorer", "scorer_mode", "tracer_type", "model",
               "enable_split", "enable_expiry", "character_profile", "baseline_scorer")
SOFT_KEYS = ("alpha", "beta", "eps_frac", "seed")


@dataclass
class Control:
    """One swapped-person slate, plus everything needed to defend it as a control."""
    ok: bool
    why: str
    run_id: Optional[str] = None
    target: Optional[str] = None
    corpus: Optional[str] = None
    step: Optional[int] = None
    requested_step: Optional[int] = None
    tier: Optional[int] = None
    span_overlap: Optional[float] = None
    align: Optional[str] = None
    side: Optional[str] = None
    particles: List[Dict[str, Any]] = field(default_factory=list)
    source_run: Optional[str] = None
    source_target: Optional[str] = None

    def __bool__(self):
        return self.ok


def _arm_of(stem: str, target) -> str:
    """The run-id stem with the target's name removed: the experiment arm.

    run_musing bakes the target into the run id, so two people on one arm never
    share a stem and a stem comparison silently makes tier 1 unreachable --
    which is what it did. Matched case-insensitively because the ids run both
    ways (`fix_katara` against target `Katara`).
    """
    if not target:
        return stem
    low, t = stem.lower(), str(target).lower()
    for form in (f"_{t}", f"-{t}", t):
        i = low.find(form)
        if i >= 0:
            return (stem[:i] + stem[i + len(form):]).strip("_-") or stem
    return stem


def _load_splits():
    if not os.path.exists(SPLITS):
        return None
    with open(SPLITS, encoding="utf-8") as fh:
        return json.load(fh)


class ControlIndex:
    """Every logged run, keyed so a swapped-person match is a lookup.

    Built from meta/runs.jsonl for the configuration and the span, and from the
    steps.jsonl beside it for the slates. A run without a step stream is indexed
    but can never be returned as a control -- there is nothing to return.
    """

    def __init__(self, out_dir: str = ml.OUT_DIR, splits: Optional[dict] = None):
        self.out_dir = out_dir
        self.splits = splits if splits is not None else _load_splits()
        self.runs: Dict[str, dict] = {}
        self._steps_cache: Dict[str, List[dict]] = {}
        self._set_side = self._build_side_map()

        path = ml.runs_jsonl(out_dir)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = d.get("run_id")
                    if not rid:
                        continue
                    # A rerun under the same id replaces its own record. Keeping
                    # the first would describe a run whose steps file is gone.
                    self.runs[rid] = d

        for rid, d in self.runs.items():
            d["_steps_path"] = ml.find_one(f"{rid}.steps.jsonl", out_dir)
            d["_span"] = frozenset(d.get("set_ids") or ())
            d["_stem"] = ml.base_run_id(rid)
            d["_arm"] = _arm_of(d["_stem"], d.get("target_agent"))
            d["_side"] = self._side_of(d["_span"])

    # -- splits -----------------------------------------------------------

    def _build_side_map(self):
        side = {}
        if not self.splits:
            return side
        for corpus in (self.splits.get("corpora") or {}).values():
            for block in corpus.get("blocks", []):
                for sid in block.get("dev", []):
                    side[sid] = "dev"
                for sid in block.get("test", []):
                    side[sid] = "test"
        return side

    def _side_of(self, span):
        """Which side of the split a run's span sits on, or 'mixed'/'unknown'.

        A span that straddles the cut is 'mixed' rather than silently assigned:
        the runs on disk were launched before the split existed, so plenty of
        them do straddle it, and calling one 'dev' would hide that.
        """
        sides = {self._set_side.get(s) for s in span}
        sides.discard(None)
        if not sides:
            return "unknown"
        if len(sides) == 1:
            return next(iter(sides))
        return "mixed"

    # -- steps ------------------------------------------------------------

    def steps(self, run_id: str) -> List[dict]:
        if run_id in self._steps_cache:
            return self._steps_cache[run_id]
        path = (self.runs.get(run_id) or {}).get("_steps_path")
        out: List[dict] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("particles"):
                        out.append(rec)
        self._steps_cache[run_id] = out
        return out

    def slate(self, run_id: str, step: int) -> List[dict]:
        """The scored slate at one step, reduced to the fields a control needs.

        The raw record carries prompts and per-call transcripts. Those are the
        run's own evidence, not the control's, and handing them to a caller
        comparing two people invites scoring the prompt instead of the slate.
        """
        steps = self.steps(run_id)
        if not 0 <= step < len(steps):
            return []
        return [{
            "particle_id": p.get("particle_id"),
            "lineage_id": p.get("lineage_id"),
            "root_id": p.get("root_id"),
            "anchor": p.get("anchor"),
            "text": p.get("text"),
            "weight": p.get("weight"),
            "likelihood": p.get("likelihood"),
            "likelihood_rank": p.get("likelihood_rank"),
        } for p in steps[step]["particles"]]

    # -- matching ---------------------------------------------------------

    @staticmethod
    def _config_match(a, b):
        hard = all(a.get(k) == b.get(k) for k in CONFIG_KEYS)
        soft = all(a.get(k) == b.get(k) for k in SOFT_KEYS
                   if a.get(k) is not None and b.get(k) is not None)
        return hard and soft

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def candidates(self, run_id: str, min_span_overlap: float = 0.6,
                   require_same_side: bool = True):
        """Every run that could serve as this run's swapped-person control.

        Sorted best tier first, then by span overlap, then by run id so the
        choice is deterministic -- a control that changes between two
        invocations is not a control.
        """
        src = self.runs.get(run_id)
        if not src:
            return []
        out = []
        for rid, d in self.runs.items():
            if rid == run_id:
                continue
            if d.get("corpus") != src.get("corpus"):
                continue
            if d.get("target_agent") == src.get("target_agent"):
                continue
            if not d.get("_steps_path"):
                continue
            if not self._config_match(src, d):
                continue
            if require_same_side and src["_side"] != d["_side"]:
                continue
            ov = self._jaccard(src["_span"], d["_span"])
            if src["_span"] and d["_span"] and src["_span"] == d["_span"]:
                tier = 1 if d["_arm"] == src["_arm"] else 2
            elif ov >= min_span_overlap:
                tier = 3
            else:
                continue
            out.append((tier, -ov, rid, d))
        out.sort(key=lambda t: (t[0], t[1], t[2]))
        return [(t, -nov, rid, d) for t, nov, rid, d in out]

    def swapped(self, run_id: str, step: int, align: str = "index",
                min_span_overlap: float = 0.6, require_same_side: bool = True,
                exclude_targets=()) -> Control:
        """A slate traced for a different person, at a comparable step.

        Refuses rather than approximates. Every refusal names what it could not
        satisfy, because "no control found" and "no control that is actually a
        control" are different answers and the second one is the useful one.
        """
        src = self.runs.get(run_id)
        if not src:
            return Control(False, f"unknown run '{run_id}'")
        n_src = len(self.steps(run_id))
        if not n_src:
            return Control(False, f"'{run_id}' has no step stream on disk",
                           source_run=run_id, source_target=src.get("target_agent"))
        if not 0 <= step < n_src:
            return Control(False, f"step {step} out of range for '{run_id}' ({n_src} steps)",
                           source_run=run_id, source_target=src.get("target_agent"))

        cands = [c for c in self.candidates(run_id, min_span_overlap, require_same_side)
                 if c[3].get("target_agent") not in set(exclude_targets)]
        if not cands:
            return Control(False, self._diagnose(run_id, min_span_overlap, require_same_side),
                           source_run=run_id, source_target=src.get("target_agent"))

        too_short = []
        for tier, ov, rid, d in cands:
            steps = self.steps(rid)
            if not steps:
                continue
            if align == "index":
                idx = step
            elif align == "fraction":
                # Round rather than floor: on a 37-step control matched to a
                # 41-step source, flooring biases every request earlier in the
                # run, which is the half where the board has not settled.
                idx = min(len(steps) - 1, round(step * (len(steps) - 1) / max(1, n_src - 1)))
            else:
                return Control(False, f"unknown align '{align}' (index|fraction)")
            if not 0 <= idx < len(steps):
                too_short.append(f"{rid} has {len(steps)} steps")
                continue
            return Control(
                ok=True,
                why=(f"tier {tier}: {'same span' if ov == 1.0 else f'span overlap {ov:.2f}'}, "
                     f"same config, {'same arm, ' if tier == 1 else ''}"
                     f"{d.get('target_agent')} in place of "
                     f"{src.get('target_agent')}; aligned by {align}"
                     + (f" ({step} -> {idx})" if idx != step else "")),
                run_id=rid, target=d.get("target_agent"), corpus=d.get("corpus"),
                step=idx, requested_step=step, tier=tier, span_overlap=round(ov, 4),
                align=align, side=d["_side"], particles=self.slate(rid, idx),
                source_run=run_id, source_target=src.get("target_agent"))

        return Control(False,
                       "every candidate is shorter than the requested step: "
                       + "; ".join(too_short[:4]),
                       source_run=run_id, source_target=src.get("target_agent"))

    def _diagnose(self, run_id, min_span_overlap, require_same_side):
        """Say which constraint did the refusing. A bare 'no match' is unactionable."""
        src = self.runs[run_id]
        same_corpus = [d for rid, d in self.runs.items()
                       if rid != run_id and d.get("corpus") == src.get("corpus")]
        other_person = [d for d in same_corpus if d.get("target_agent") != src.get("target_agent")]
        with_steps = [d for d in other_person if d.get("_steps_path")]
        same_cfg = [d for d in with_steps if self._config_match(src, d)]
        same_side = [d for d in same_cfg if d["_side"] == src["_side"]]
        pool = same_side if require_same_side else same_cfg
        best = max((self._jaccard(src["_span"], d["_span"]) for d in pool), default=0.0)
        return ("no control: "
                f"{len(same_corpus)} runs on {src.get('corpus')}, "
                f"{len(other_person)} on another person, {len(with_steps)} with a step "
                f"stream, {len(same_cfg)} config-compatible, {len(same_side)} on the same "
                f"split side ({src['_side']}); best span overlap {best:.2f} against a "
                f"floor of {min_span_overlap:.2f}")

    # -- reporting --------------------------------------------------------

    def coverage(self, **kw):
        rows = []
        for rid, d in sorted(self.runs.items()):
            if not d.get("_steps_path"):
                continue
            c = self.candidates(rid, **kw)
            rows.append({"run": rid, "corpus": d.get("corpus"),
                         "target": d.get("target_agent"), "side": d["_side"],
                         "steps": len(self.steps(rid)),
                         "controls": len(c),
                         "best_tier": c[0][0] if c else None,
                         "control_targets": sorted({x[3].get("target_agent") for x in c})[:6]})
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=ml.OUT_DIR)
    ap.add_argument("--run")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--align", default="index", choices=["index", "fraction"])
    ap.add_argument("--min-span-overlap", type=float, default=0.6)
    ap.add_argument("--any-side", action="store_true",
                    help="allow a control from the other side of splits.json (it is not one)")
    ap.add_argument("--list", action="store_true", help="control coverage over every run")
    a = ap.parse_args()

    idx = ControlIndex(a.out_dir)

    if a.list or not a.run:
        rows = idx.coverage(min_span_overlap=a.min_span_overlap,
                            require_same_side=not a.any_side)
        have = [r for r in rows if r["controls"]]
        print(f"{len(have)} of {len(rows)} runs with a step stream have a swapped-person "
              f"control at overlap >= {a.min_span_overlap}")
        by_tier = collections.Counter(r["best_tier"] for r in have)
        for tier in sorted(t for t in by_tier if t):
            print(f"  tier {tier}: {by_tier[tier]} runs")
        hdr = f"{'run':<44}{'target':<15}{'side':<8}{'steps':>6}{'ctl':>5}{'tier':>5}  controls"
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in have:
            print(f"{r['run'][:43]:<44}{str(r['target'])[:14]:<15}{r['side']:<8}"
                  f"{r['steps']:>6}{r['controls']:>5}{r['best_tier']:>5}  "
                  f"{', '.join(map(str, r['control_targets']))}")
        return

    ctl = idx.swapped(a.run, a.step, align=a.align,
                      min_span_overlap=a.min_span_overlap,
                      require_same_side=not a.any_side)
    if not ctl:
        raise SystemExit(f"REFUSED: {ctl.why}")
    print(f"{ctl.source_run} ({ctl.source_target}) step {ctl.requested_step}")
    print(f"  -> {ctl.run_id} ({ctl.target}) step {ctl.step}")
    print(f"  {ctl.why}")
    print(f"  side={ctl.side} tier={ctl.tier} span_overlap={ctl.span_overlap}\n")
    for p in sorted(ctl.particles, key=lambda p: p.get("likelihood_rank") or 99):
        print(f"  #{p['likelihood_rank']} w={p['weight']:.3f}  {p['anchor']}")


if __name__ == "__main__":
    main()
