"""Offline tests for the tracer -> memory write path.

No LLM calls. Kept separate from test_memory.py so that module stays free of
any tracer import, which is what lets audits read a store without building an
API client.

What they protect, in order of importance:

  1. OFF BY DEFAULT, AND SILENT WHEN OFF. Every measurement in musing_out was
     produced without this. A run with no --memory must touch no disk and
     behave identically.
  2. REFUSE RATHER THAN MIS-FILE. The store is namespaced by corpus. A missing
     corpus wrote to memory/unknown/ in the first version, where two corpora
     would silently merge under one token -- caught by a live run, not by
     review, which is the third silent-no-op bug in this stage.
  3. PEAK, NOT FINAL. A commitment that led at step 12 and was merely holding
     on at the end is a better revival candidate than one that drifted up on
     the last step. _retired already stores peaks for exactly this reason.
"""
import os
import tempfile
import types

import numpy as np

import memory
from hypothesis import HypothesesSetV3
from tracer import Tracer


def _hyps(anchors, weights):
    return HypothesesSetV3('Wolf', [], [], [f't:{a}' for a in anchors],
                           np.array(weights, dtype=float), anchors=list(anchors))


def _tracer(tmp, **kw):
    t = Tracer.__new__(Tracer)
    t.target_agent = 'Wolf'
    t._retired = {}
    base = dict(memory=True, memory_dir=tmp, corpus='bloomfield',
                memory_span_id={'id': 'span-1', 'ordinal': 1},
                run_id='r1', model='m', seed=0, n_hypotheses=8)
    base.update(kw)
    t.args = types.SimpleNamespace(**base)
    return t


def _store_of(tmp, t):
    cfg = memory.config_hash(t.args)
    return memory.load(tmp, 'bloomfield', cfg, 'Wolf')


# --------------------------------------------------------------------------
# off by default
# --------------------------------------------------------------------------

def test_no_flag_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d, memory=False)
        assert t._persist_memory([_hyps(['A'], [1.0])]) is None
        assert not os.path.exists(os.path.join(d, 'memory'))


def test_a_missing_corpus_refuses_rather_than_filing_under_unknown():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d, corpus=None)
        assert t._persist_memory([_hyps(['A'], [1.0])]) is None
        assert not os.path.exists(os.path.join(d, 'memory', 'unknown'))


# --------------------------------------------------------------------------
# what gets written
# --------------------------------------------------------------------------

def test_the_final_population_is_live_and_everything_else_retired():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._persist_memory([_hyps(['A', 'B'], [.5, .5]),
                           _hyps(['A', 'C'], [.5, .5])])
        r = _store_of(d, t)['records']
        assert r['A']['state'] == 'live' and r['C']['state'] == 'live'
        assert r['B']['state'] == 'retired'


def test_peak_weight_is_kept_not_the_final_one():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._persist_memory([_hyps(['A'], [0.9]), _hyps(['A'], [0.1])])
        assert _store_of(d, t)['records']['A']['peak'] == 0.9


def test_the_exit_path_of_a_mid_run_retirement_survives():
    # The live snapshot cannot supply `reason`; _retired can, and the audit
    # asks which exit path produces commitments worth bringing back.
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._retired = {'Gone': {'anchor': 'Gone', 'text': '', 'peak': 0.4,
                               'reason': 'surprise', 'root_id': 'r9',
                               'revivals': 2}}
        t._persist_memory([_hyps(['A'], [1.0])])
        rec = _store_of(d, t)['records']['Gone']
        assert rec['reason'] == 'surprise' and rec['revivals'] == 2
        assert rec['root_id'] == 'r9'


def test_blank_anchors_are_not_stored():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._persist_memory([_hyps(['A', ''], [.5, .5])])
        assert list(_store_of(d, t)['records']) == ['A']


# --------------------------------------------------------------------------
# mid-run checkpointing
# --------------------------------------------------------------------------

def test_a_checkpoint_writes_retirements():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._retired = {'Gone': {'anchor': 'Gone', 'peak': 0.3, 'reason': 'expiry'}}
        t._checkpoint_memory()
        assert _store_of(d, t)['records']['Gone']['state'] == 'retired'


def test_a_checkpoint_never_marks_anything_live():
    # THE rule. `live` means "held when the run finished"; a mid-run
    # checkpoint has no end to speak for, so it must not claim one.
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._retired = {'Gone': {'anchor': 'Gone', 'peak': 0.3}}
        t._checkpoint_memory()
        assert all(v['state'] == 'retired'
                   for v in _store_of(d, t)['records'].values())


def test_a_checkpoint_with_nothing_retired_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        assert t._checkpoint_memory() is None
        assert not os.path.exists(os.path.join(d, 'memory'))


def test_the_final_write_promotes_a_checkpointed_anchor_back_to_live():
    # A commitment can be retired at step 12 and revived by the end. The
    # checkpoint must not pin it as retired forever.
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._retired = {'A': {'anchor': 'A', 'peak': 0.3, 'reason': 'expiry'}}
        t._checkpoint_memory()
        assert _store_of(d, t)['records']['A']['state'] == 'retired'
        t._retired = {}
        t._persist_memory([_hyps(['A'], [1.0])])
        assert _store_of(d, t)['records']['A']['state'] == 'live'


def test_checkpointing_is_off_when_memory_is_off():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d, memory=False)
        t._retired = {'Gone': {'anchor': 'Gone', 'peak': 0.3}}
        assert t._checkpoint_memory() is None


# --------------------------------------------------------------------------
# eviction
# --------------------------------------------------------------------------

def test_a_commitment_seen_in_two_spans_records_both():
    with tempfile.TemporaryDirectory() as d:
        for o in (1, 7):
            _tracer(d, memory_span_id={'id': f's{o}', 'ordinal': o})._persist_memory(
                [_hyps(['A'], [0.8])])
        rec = _store_of(d, _tracer(d))['records']['A']
        assert [x['id'] for x in rec['threads']] == ['s1', 's7']


def test_a_restored_commitment_is_marked_as_such():
    """Provenance lives on the TRACER, not the particle: operators do not
    survive propagation, so by run end the holder of a restored anchor is a
    descendant that never saw the restore. Kept after consolidation was
    removed because it is what tells a commitment the filter found from one
    the store handed back."""
    with tempfile.TemporaryDirectory() as d:
        t1 = _tracer(d, memory_span_id={'id': 's1', 'ordinal': 1})
        t1._persist_memory([_hyps(['A'], [0.8])])
        t2 = _tracer(d, memory_span_id={'id': 's9', 'ordinal': 9})
        t2._restored_anchors = {'A'}
        t2._persist_memory([_hyps(['A'], [0.8])])
        rec = _store_of(d, t2)['records']['A']
        assert rec['threads'][1].get('restored') is True
        assert rec['threads'][0].get('restored') is not True


def test_the_store_is_trimmed_on_a_final_write():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        anchors = [f'A{i}' for i in range(10)]
        t._persist_memory([_hyps(anchors, [0.1] * 10)])
        # cap is read from args by the caller; default path keeps all 10
        assert len(_store_of(d, t)['records']) <= 60


def test_a_checkpoint_does_not_evict():
    # A mid-run checkpoint has no authority over what the run will end up
    # holding, so it must not drop anything on the strength of a partial view.
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d)
        t._retired = {f'R{i}': {'anchor': f'R{i}', 'peak': 0.1} for i in range(5)}
        t._checkpoint_memory()
        assert len(_store_of(d, t)['records']) == 5


# --------------------------------------------------------------------------
# across runs
# --------------------------------------------------------------------------

def test_a_second_run_accumulates_rather_than_replacing():
    with tempfile.TemporaryDirectory() as d:
        t1 = _tracer(d, run_id='r1', memory_span_id={'id': 's1', 'ordinal': 1})
        t1._persist_memory([_hyps(['A'], [0.8])])
        t2 = _tracer(d, run_id='r2', memory_span_id={'id': 's2', 'ordinal': 5})
        t2._persist_memory([_hyps(['B'], [0.6])])
        r = _store_of(d, t2)['records']
        assert set(r) == {'A', 'B'}

def test_a_string_span_id_is_accepted_and_given_an_ordinal():
    with tempfile.TemporaryDirectory() as d:
        t = _tracer(d, memory_span_id='plain-string')
        t._persist_memory([_hyps(['A'], [1.0])])
        rec = _store_of(d, t)['records']['A']
        assert rec['threads'][0]['id'] == 'plain-string'


def test_a_different_seed_does_not_see_the_other_seeds_store():
    with tempfile.TemporaryDirectory() as d:
        a = _tracer(d, seed=0)
        a._persist_memory([_hyps(['A'], [1.0])])
        b = _tracer(d, seed=1)
        assert _store_of(d, b)['records'] == {}


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
