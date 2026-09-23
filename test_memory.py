"""Offline tests for the durable commitment store.

No LLM calls. What they protect, in order of importance:

  1. A STORED WEIGHT IS NEVER A PRIOR. The store holds peak weights from
     previous runs, on a previous run's normalising scale. They may order
     candidates and nothing else. This is the same rule profiles.py states for
     read_confidence, and the store is where it is most tempting to break.
  2. CONFIG ISOLATION. Memory from a differently-configured filter must refuse
     to load, while memory from the same configuration over a DIFFERENT SPAN
     must load -- that asymmetry is the whole point of a cross-run store, and
     getting it backwards either silently mixes arms or breaks the feature.
  3. THREAD IDENTITY IS CORPUS-WIDE. Ordinals computed over a run's own span
     come out consecutive and are not comparable across runs, which is what
     eviction reads to tell a stale commitment from a current one.
"""
import json
import os
import tempfile
import types

import memory


def _args(**kw):
    base = dict(seed=0, n_hypotheses=8, standard_prompt='v4')
    base.update(kw)
    return types.SimpleNamespace(**base)


def _thread(i):
    return {'id': f't{i}', 'ordinal': i}


def _store():
    return memory.empty('bloomfield', 'cfg0', 'Wolf')


# --------------------------------------------------------------------------
# config isolation
# --------------------------------------------------------------------------

def test_a_different_seed_is_a_different_store():
    # Otherwise seeds cross-contaminate through memory and the seed spread
    # stops measuring seed variation.
    assert memory.config_hash(_args(seed=0)) != memory.config_hash(_args(seed=1))


def test_a_different_span_is_the_SAME_store():
    # The two-half verification depends on this: half 2 covers different turns
    # and must still see half 1's memory.
    a = memory.config_hash(_args())
    b = memory.config_hash(_args(set_ids='x,y', chronological=True, run_id='r2'))
    assert a == b


def test_adding_a_flag_changes_the_hash_rather_than_silently_matching():
    assert memory.config_hash(_args()) != memory.config_hash(_args(methods='ach'))


def test_the_path_alone_separates_two_configs():
    # First line of defence: the store is namespaced by config, so the wrong
    # config simply finds nothing rather than reading the wrong file.
    with tempfile.TemporaryDirectory() as d:
        memory.save(d, 'bloomfield', 'cfgA', 'Wolf', _store())
        assert memory.load(d, 'bloomfield', 'cfgB', 'Wolf')['records'] == {}


def test_a_file_whose_stamp_disagrees_with_its_path_refuses():
    # Second line: a store hand-copied between config directories, or written
    # before the hash function changed, sits at the right path with the wrong
    # stamp. Loading it would silently mix two filters' memories.
    with tempfile.TemporaryDirectory() as d:
        memory.save(d, 'bloomfield', 'cfgA', 'Wolf', _store())
        p = memory.store_path(d, 'bloomfield', 'cfgA', 'Wolf')
        blob = json.load(open(p))
        blob['config_hash'] = 'SOMETHING-ELSE'
        json.dump(blob, open(p, 'w'))
        try:
            memory.load(d, 'bloomfield', 'cfgA', 'Wolf')
        except ValueError as e:
            assert 'config_hash' in str(e)
        else:
            raise AssertionError('a stamp mismatch must refuse')


def test_a_stale_schema_refuses():
    with tempfile.TemporaryDirectory() as d:
        memory.save(d, 'bloomfield', 'cfg0', 'Wolf', _store())
        p = memory.store_path(d, 'bloomfield', 'cfg0', 'Wolf')
        blob = json.load(open(p))
        blob['schema_version'] = memory.SCHEMA_VERSION - 1
        json.dump(blob, open(p, 'w'))
        try:
            memory.load(d, 'bloomfield', 'cfg0', 'Wolf')
        except ValueError as e:
            assert 'schema' in str(e)
        else:
            raise AssertionError('a stale schema must refuse')


def test_a_missing_store_is_empty_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        assert memory.load(d, 'bloomfield', 'cfg0', 'Nobody')['records'] == {}


# --------------------------------------------------------------------------
# thread identity
# --------------------------------------------------------------------------

def test_a_thread_is_a_set_in_this_corpus():
    # Measured: source.thread_id is 1:1 with sets (402 keys, 402 sets), so
    # Slack threading does not group them and each set is already atomic.
    tm = memory.thread_map_for('bloomfield')
    assert len(tm) == 402
    assert tm['bloomfield-0002']['ordinal'] == 2


def test_thread_ids_are_distinct_per_set():
    tm = memory.thread_map_for('bloomfield')
    assert len({v['id'] for v in tm.values()}) == len(tm)


# --------------------------------------------------------------------------
# durability
# --------------------------------------------------------------------------

def test_a_saved_store_round_trips():
    with tempfile.TemporaryDirectory() as d:
        s = _store()
        memory.put(s, anchor='Keep growers confident', peak=0.4, state='live',
                   thread=_thread(1), date='2024-03-01')
        memory.save(d, 'bloomfield', 'cfg0', 'Wolf', s)
        back = memory.load(d, 'bloomfield', 'cfg0', 'Wolf')
        assert back['records']['Keep growers confident']['peak'] == 0.4


def test_a_failed_write_leaves_the_previous_version_intact():
    # Checkpointing at every thread close means a crash lands on a write often.
    with tempfile.TemporaryDirectory() as d:
        s = _store()
        memory.put(s, anchor='A', peak=0.5, state='live')
        p = memory.save(d, 'bloomfield', 'cfg0', 'Wolf', s)
        before = open(p).read()
        class Unserialisable:
            pass
        s['records']['B'] = Unserialisable()
        try:
            memory.save(d, 'bloomfield', 'cfg0', 'Wolf', s)
        except TypeError:
            pass
        assert open(p).read() == before
        leftovers = [f for f in os.listdir(os.path.dirname(p)) if f.endswith('.tmp')]
        assert leftovers == []


def test_peak_is_a_high_water_mark_not_the_latest_value():
    s = _store()
    memory.put(s, anchor='A', peak=0.7)
    memory.put(s, anchor='A', peak=0.1)
    assert s['records']['A']['peak'] == 0.7


def test_the_same_thread_is_not_recorded_twice():
    s = _store()
    memory.put(s, anchor='A', thread=_thread(1))
    memory.put(s, anchor='A', thread=_thread(1))
    assert len(s['records']['A']['threads']) == 1


# --------------------------------------------------------------------------
# the weight rule
# --------------------------------------------------------------------------

def test_restorable_orders_by_peak_but_hands_back_no_weight_to_apply():
    s = _store()
    memory.put(s, anchor='low', peak=0.1, state='live')
    memory.put(s, anchor='high', peak=0.9, state='live')
    rows = memory.restorable(s)
    assert [r['anchor'] for r in rows] == ['high', 'low']
    # The caller restores at uniform weight; nothing here is a weight to use.
    assert all('weight' not in r for r in rows)


def test_restorable_returns_live_only():
    s = _store()
    memory.put(s, anchor='held', state='live')
    memory.put(s, anchor='dropped', state='retired')
    assert [r['anchor'] for r in memory.restorable(s)] == ['held']


# --------------------------------------------------------------------------
# eviction
# --------------------------------------------------------------------------

def test_eviction_prefers_recent_peaks_over_ancient_ones():
    s = _store()
    memory.put(s, anchor='ancient', peak=0.9, thread=_thread(1))
    memory.put(s, anchor='recent', peak=0.6, thread=_thread(20))
    memory.evict(s, thread_ordinal=20, cap=1)
    assert 'recent' in s['records'] and 'ancient' not in s['records']


def test_a_revived_commitment_outranks_one_that_never_returned():
    s = _store()
    memory.put(s, anchor='returned', peak=0.2, thread=_thread(5), revivals=1)
    memory.put(s, anchor='never', peak=0.8, thread=_thread(5))
    memory.evict(s, thread_ordinal=5, cap=1)
    assert 'returned' in s['records']


def test_the_store_is_capped():
    s = _store()
    for i in range(10):
        memory.put(s, anchor=f'a{i}', peak=i / 10)
    memory.evict(s, thread_ordinal=1, cap=3)
    assert len(s['records']) == 3


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
