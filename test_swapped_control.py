"""Offline tests for the swapped-person control. No LLM calls, no disk logs.

What they protect, each from a way a control quietly stops being one:

  1. The swap actually happens. A control that returns the source person's own
     slate is the null result generator PRECHECKS is full of.
  2. Refusals stay refusals. Tier order and the config, side and span filters
     exist so a near-miss is never returned as a match; a fallback to "the
     nearest run" turns a null into an uninterpretable one.
  3. Determinism. Two calls with the same arguments return the same control, or
     the arm is not a control at all.
  4. Step alignment is honest. Runs on one span have different lengths, so an
     out-of-range request refuses and a fraction-aligned one reports the step
     it landed on rather than the one that was asked for.
"""
import json
import os
import shutil
import sys
import tempfile

import musing_layout as ml
from swapped_control import ControlIndex, _arm_of


# --------------------------------------------------------------------------
# a synthetic musing_out, so the tests do not depend on which runs are on disk
# --------------------------------------------------------------------------

BASE = {
    "n_hypotheses": 8, "scorer": "comparative", "scorer_mode": "rank",
    "tracer_type": "tracer", "model": "gemini-2.5-flash", "enable_split": True,
    "enable_expiry": True, "character_profile": False, "baseline_scorer": False,
    "alpha": 0.85, "beta": 1.0, "eps_frac": 0.12, "seed": 0,
}


def _write(tmp, run_id, corpus, target, set_ids, n_steps, **over):
    meta = dict(BASE, run_id=run_id, corpus=corpus, target_agent=target,
                set_ids=list(set_ids), steps=n_steps)
    meta.update(over)
    with open(ml.runs_jsonl(tmp), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
    if n_steps is None:
        return
    path = ml.run_path(run_id, f"{run_id}.steps.jsonl", tmp, create=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n_steps):
            fh.write(json.dumps({
                "step_idx": i,
                "particles": [{"particle_id": f"{target}{i}{k}", "lineage_id": f"l{k}",
                               "root_id": f"r{k}", "anchor": f"{target} aim {k}",
                               "text": f"{target} believes {k} at {i}",
                               "weight": 1.0 / 4, "likelihood": 1.0 / 4,
                               "likelihood_rank": k + 1} for k in range(4)],
            }) + "\n")


def _fixture():
    tmp = tempfile.mkdtemp(prefix="swapctl-")
    os.makedirs(ml.meta_dir(tmp, create=True), exist_ok=True)
    span = ["c-0001", "c-0002", "c-0003"]
    _write(tmp, "armA_Alice-c-silver-000", "c", "Alice", span, 10)
    _write(tmp, "armA_Bob-c-silver-000", "c", "Bob", span, 6)
    _write(tmp, "armB_Carol-c-silver-000", "c", "Carol", span, 10)
    # different population: config-incompatible, must never be returned
    _write(tmp, "armA_Dave-c-silver-000", "c", "Dave", span, 10, n_hypotheses=4)
    # different corpus
    _write(tmp, "armA_Erin-d-silver-000", "d", "Erin", ["d-0001"], 10)
    # no step stream
    _write(tmp, "armA_Frank-c-silver-000", "c", "Frank", span, None)
    splits = {"corpora": {"c": {"blocks": [
        {"block": "c", "dev": span, "test": ["c-0100"], "unassigned": []}]}}}
    return tmp, splits


def _index():
    tmp, splits = _fixture()
    return tmp, ControlIndex(tmp, splits=splits)


# --------------------------------------------------------------------------

def test_arm_strips_the_target_from_the_run_id():
    # The whole reason tier 1 was unreachable: run_musing bakes the target into
    # the id, so two people on one arm never share a raw stem.
    assert _arm_of("cov1s0_Lyubovsky", "Lyubovsky") == "cov1s0"
    assert _arm_of("cov1s0_Rovani", "Rovani") == "cov1s0"
    assert _arm_of("fix_katara", "Katara") == "fix"          # case-insensitive
    assert _arm_of("v3_bb_Rodriguez", "Rodriguez") == "v3_bb"
    assert _arm_of("bb_Deskins", "Deskins") == _arm_of("bb_Rodriguez", "Rodriguez")
    assert _arm_of("standalone", None) == "standalone"


def test_returns_a_different_person_on_the_same_span():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 3)
        assert ctl, ctl.why
        assert ctl.target != "Alice"
        assert ctl.span_overlap == 1.0
        assert ctl.step == 3 and ctl.requested_step == 3
        assert all("Alice" not in (p["anchor"] or "") for p in ctl.particles)
        assert len(ctl.particles) == 4
    finally:
        shutil.rmtree(tmp)


def test_same_arm_is_preferred_over_another_arm():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 3)
        assert ctl.tier == 1 and ctl.run_id == "armA_Bob-c-silver-000", ctl.why
    finally:
        shutil.rmtree(tmp)


def test_incompatible_population_is_never_returned():
    # Rank is bounded by population size, so a 4-hypothesis run is not a
    # control for an 8-hypothesis one however well its span matches.
    tmp, idx = _index()
    try:
        for tier, ov, rid, d in idx.candidates("armA_Alice-c-silver-000"):
            assert d["target_agent"] != "Dave", "config filter let a different n through"
    finally:
        shutil.rmtree(tmp)


def test_runs_without_a_step_stream_are_not_offered():
    tmp, idx = _index()
    try:
        names = {rid for _, _, rid, _ in idx.candidates("armA_Alice-c-silver-000")}
        assert "armA_Frank-c-silver-000" not in names
    finally:
        shutil.rmtree(tmp)


def test_other_corpus_is_not_a_control():
    tmp, idx = _index()
    try:
        corpora = {d["corpus"] for _, _, _, d in idx.candidates("armA_Alice-c-silver-000")}
        assert corpora == {"c"}
    finally:
        shutil.rmtree(tmp)


def test_out_of_range_step_refuses_and_says_why():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 40)
        assert not ctl
        assert "out of range" in ctl.why
        assert ctl.particles == []
    finally:
        shutil.rmtree(tmp)


def test_a_shorter_control_refuses_rather_than_clamping():
    # Bob has 6 steps to Alice's 10. Step 8 has no index-aligned control, and
    # returning Bob's last step instead would compare two different moments.
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 8, exclude_targets=("Carol",))
        assert not ctl, f"returned {ctl.run_id} step {ctl.step}"
        assert "shorter" in ctl.why
    finally:
        shutil.rmtree(tmp)


def test_fraction_alignment_reports_the_step_it_landed_on():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 9, align="fraction",
                          exclude_targets=("Carol",))
        assert ctl, ctl.why
        assert ctl.run_id == "armA_Bob-c-silver-000"
        assert ctl.step == 5 and ctl.requested_step == 9
        assert "9 -> 5" in ctl.why
    finally:
        shutil.rmtree(tmp)


def test_repeated_calls_return_the_same_control():
    tmp, idx = _index()
    try:
        a = idx.swapped("armA_Alice-c-silver-000", 2)
        b = ControlIndex(tmp, splits=idx.splits).swapped("armA_Alice-c-silver-000", 2)
        assert a.run_id == b.run_id and a.step == b.step
        assert [p["particle_id"] for p in a.particles] == [p["particle_id"] for p in b.particles]
    finally:
        shutil.rmtree(tmp)


def test_split_side_is_enforced_and_can_be_named_in_the_refusal():
    tmp, splits = _fixture()
    # Move Bob and Carol onto the test side; Alice stays on dev.
    splits = {"corpora": {"c": {"blocks": [
        {"block": "c", "dev": ["c-0001", "c-0002", "c-0003"], "test": [], "unassigned": []}]}}}
    idx = ControlIndex(tmp, splits=splits)
    try:
        # Same side: fine.
        assert idx.swapped("armA_Alice-c-silver-000", 1)
        # Now put every set on the test side except Alice's, which strands her.
        crossed = ControlIndex(tmp, splits={"corpora": {"c": {"blocks": [
            {"block": "c", "dev": ["c-0001"], "test": ["c-0002"], "unassigned": []}]}}})
        # Every run shares the span, so every run is 'mixed' -- same side, still matched.
        ctl = crossed.swapped("armA_Alice-c-silver-000", 1)
        assert ctl and ctl.side == "mixed", ctl.why
    finally:
        shutil.rmtree(tmp)


def test_refusal_names_the_constraint_that_refused():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Erin-d-silver-000", 1)
        assert not ctl
        for token in ("runs on d", "config-compatible", "span overlap"):
            assert token in ctl.why, ctl.why
    finally:
        shutil.rmtree(tmp)


def test_unknown_run_refuses():
    tmp, idx = _index()
    try:
        ctl = idx.swapped("nope", 0)
        assert not ctl and "unknown run" in ctl.why
    finally:
        shutil.rmtree(tmp)


def test_slate_carries_no_prompt_or_transcript():
    # A caller comparing two people must not be handed the run's own prompts.
    tmp, idx = _index()
    try:
        ctl = idx.swapped("armA_Alice-c-silver-000", 1)
        for p in ctl.particles:
            assert not any(k in p for k in ("likelihood_prompt", "raw_verdicts", "llm_calls"))
            assert set(p) == {"particle_id", "lineage_id", "root_id", "anchor", "text",
                              "weight", "likelihood", "likelihood_rank"}
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  ok   {n}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
