"""Offline tests for margin instrumentation.

No LLM calls. What they protect:

  1. `margins` reaching disk. Without the full vector a run cannot be replayed
     to sweep a threshold over it, which is the difference between tuning the
     episode detector for free and paying for a run per candidate.
  2. The `lean_undefined` / `surprise` distinction. They differ precisely on
     routine traffic, and conflating them is the failure trace_log already
     records once ("made the filter mint on nearly every step of a low-signal
     span"). A quantity built by weighting commitments by their margin is
     UNDEFINED whenever the null wins, whether or not the turn was routine.
"""
import types

import numpy as np

import trace_log
from trace_log import StepRecord
from tracer import Tracer
from hypothesis import HypothesesSetV3


class _Logger:
    """Captures the emitted record instead of writing it."""
    def __init__(self):
        self.records = []

    def step(self, rec):
        self.records.append(rec)
        return rec


def _emit(weight_results, n=3):
    t = Tracer.__new__(Tracer)
    t.run_logger = _Logger()
    # Everything _log_step reads off args does so through getattr with a
    # default, so an empty namespace exercises the default path exactly.
    t.args = types.SimpleNamespace()
    t._step_idx = 0
    hyps = HypothesesSetV3('X', [], [], [f't{i}' for i in range(n)],
                           np.ones(n) / n, anchors=[f'A{i}' for i in range(n)])
    trace_log.RECORDER.drain()
    rec = t._log_step(0, hyps, weight_results, operators=[])
    return rec if rec is not None else t.run_logger.records[-1]


def _wr(**kw):
    base = {'prompts': ['p'], 'system_prompt': 's', 'weights': [.5, .3, .2]}
    base.update(kw)
    return base


# --------------------------------------------------------------------------

def test_the_field_exists_on_the_record():
    assert 'margins' in StepRecord.__dataclass_fields__
    assert 'lean_undefined' in StepRecord.__dataclass_fields__


def test_margins_reach_the_record_as_floats():
    rec = _emit(_wr(margins=[30.0, -5.0, -20.0], best_margin=30.0))
    assert rec.margins == [30.0, -5.0, -20.0]
    assert all(isinstance(x, float) for x in rec.margins)


def test_absent_margins_stay_none_rather_than_becoming_an_empty_list():
    # An empty list would read downstream as "scored, nothing positive";
    # None is "this step was never baseline-scored". Different questions.
    assert _emit(_wr()).margins is None
    assert _emit(_wr()).lean_undefined is None


def test_lean_is_undefined_when_the_null_wins():
    rec = _emit(_wr(margins=[-4.0, -9.0, -30.0], best_margin=-4.0))
    assert rec.lean_undefined is True


def test_lean_is_defined_when_any_commitment_beats_the_null():
    rec = _emit(_wr(margins=[1.0, -9.0, -30.0], best_margin=1.0))
    assert rec.lean_undefined is False


def test_a_zero_best_margin_is_undefined_not_defined():
    # Weighting by max(margin, 0) divides by zero here, so the boundary must
    # fall on the undefined side.
    assert _emit(_wr(margins=[0.0, -2.0, -8.0], best_margin=0.0)).lean_undefined is True


def test_lean_undefined_is_not_surprise_on_routine_traffic():
    # THE distinction. Routine turn, null wins: surprise is suppressed because
    # nothing should be minted, but the lean genuinely has no value.
    rec = _emit(_wr(margins=[-3.0, -8.0, -11.0], best_margin=-3.0,
                    off_topic=True, surprise=False))
    assert rec.surprise is False
    assert rec.lean_undefined is True


def test_lean_undefined_agrees_with_surprise_when_the_turn_is_not_routine():
    rec = _emit(_wr(margins=[-3.0, -8.0, -11.0], best_margin=-3.0,
                    off_topic=False, surprise=True))
    assert rec.surprise is True and rec.lean_undefined is True


if __name__ == '__main__':
    import sys
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
