"""A durable, per-person commitment store that survives between runs.

WHY. `_trace` resets `_accum`, `_retired`, `_low_mass_run` and `_weak_run` at
the top of every run, so nothing the filter learns about a person outlives the
context it learned it in. The only cross-episode representation in the repo is
`<corpus>_profiles.json`, which is fabricated and read once at seeding. A
system meant to model someone's long-term mental state cannot have its memory
end at the set boundary.

WHAT IS STORED. Both halves, which is the part the first draft got wrong:

  live      the standing particles still held when the run ended. Without
            these, "carries a standing board across a corpus" is untrue -- a
            store of only retirements remembers exactly what the filter
            decided to stop believing.
  retired   the existing `_retired` cache, which is already the right shape:
            keyed by anchor, holding PEAK weight, preserving root_id, method,
            standard and the exit path.

A STORED WEIGHT IS NEVER A PRIOR. `profiles.py` bans routing an external
confidence into the filter's weights because those weights are normalised
posteriors over a live population and an outside number is on nobody's scale.
A weight from a PREVIOUS RUN has exactly the same defect. It may order
candidates for revival and eviction; it may never enter `accumulate`, and
restored particles enter at uniform weight.

KEYS ARE EXACT, NOT STEMMED. Measured over 3,272 distinct normalised anchors
from every run on disk: the `audit_methods` stem merges 63% of them, putting
"Ensure data loads completely" and "Ensure data quality" under one key. That
is right for grouping and fatal for identity, so the key is the normalised
anchor string.

THIS IS A LOG, NOT A CONSOLIDATING MEMORY, and the difference is measured.
A consolidation pass lived here and was removed: P-12 found that two runs on
the same person, same config, adjacent spans produce 44 commitments with ZERO
exact overlap, zero stem overlap and zero pairs clearing a 0.6 Jaccard floor.
Promotion requires the same commitment in two threads, so it could never fire.
Rebuilding it needs semantic matching first -- `merge_equivalent_anchors` uses
a model rather than a threshold inside a run for exactly this reason -- not
another lexical rule.

No tracer import: audits and the eval harness must be able to read a store
without constructing an API client.
"""
import hashlib
import json
import os
import tempfile

SCHEMA_VERSION = 3

# Which flags make two runs' memories comparable. The span MUST be excluded --
# the whole point is that run 2 covers different turns than run 1 -- and the
# seed MUST be included, or seeds cross-contaminate through the store and the
# seed spread stops measuring seed variation.
CONFIG_KEYS = (
    'standard_prompt', 'methods', 'n_hypotheses', 'seed',
)

STATES = ('live', 'retired')
DECAY = 0.9            # per thread since last seen; a guess, to be measured
CAP = 60


def thread_map(sets):
    """Assign a thread identity and a chronological ordinal to each set.

    MEASURED, because the design assumed otherwise: `source.thread_id` is
    present on all 402 bloomfield sets and is 1:1 with them -- 402 distinct
    (channel, thread_id) keys for 402 sets, none spanning more than one. Slack
    threading does not group sets in this corpus, so a thread IS a set and the
    planned `--thread-gap` (14 days of silence closes a thread) has nothing to
    close: sets are atomic and already bounded.

    The ORDINAL orders a person's spans, which is what eviction reads to tell
    a commitment last seen a year ago from one last seen last month. Ordering
    is by (date, set_id) so it is stable across runs.

    PASS THE WHOLE CORPUS, never a run's span. Ordinals over a subset come out
    consecutive -- the six color sets are 2,6,9,15,31,39 corpus-wide and
    0,1,2,3,4,5 on their own -- and across runs two spans would each start at
    0, so their ordinals would not be comparable at all. `thread_map_for` is
    the safe entry point; this one is for callers that already hold the full
    list.
    """
    rows = sorted(sets, key=lambda s: (s.get('date') or '', s['set_id']))
    out = {}
    for i, s in enumerate(rows):
        src = s.get('source') or {}
        tid = src.get('thread_id') or s['set_id']
        out[s['set_id']] = {'id': f"{s.get('channel', '?')}:{tid}",
                            'ordinal': i,
                            'set_id': s['set_id'],
                            'date': s.get('date')}
    return out


def thread_map_for(corpus, data_dir='data/musing'):
    """thread_map over the whole corpus, which is the only correct input.

    The safe entry point: a caller holding only the sets its run traced cannot
    accidentally produce span-local ordinals through this.
    """
    with open(f'{data_dir}/{corpus}_dialogue.json', encoding='utf-8') as fh:
        return thread_map(json.load(fh))


def config_hash(args) -> str:
    """Identity of the configuration that produced a memory.

    Records the value of every key in CONFIG_KEYS including the ones that are
    absent, so adding a flag changes the hash rather than silently matching
    older stores that never had it.
    """
    payload = {k: getattr(args, k, None) for k in CONFIG_KEYS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def store_path(root, corpus, cfg, token):
    return os.path.join(root, 'memory', corpus, cfg, f'{token}.json')


def _atomic_write(path, obj):
    """Write via a temp file in the SAME directory, then replace.

    The store is checkpointed at every thread close, so a crash lands on a
    write far more often than it would at run end. os.replace is atomic only
    within a filesystem, hence the same directory rather than /tmp.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def empty(corpus, cfg, token):
    return {'schema_version': SCHEMA_VERSION, 'token': token,
            'corpus': corpus, 'config_hash': cfg, 'records': {}}


def load(root, corpus, cfg, token, allow_mismatch=False):
    """Read a store. A schema or config mismatch refuses rather than merges.

    Mixing memories across configurations would make every arm comparison
    meaningless, and the failure would be invisible -- the store would simply
    contain commitments from a filter that behaved differently.
    """
    path = store_path(root, corpus, cfg, token)
    if not os.path.exists(path):
        return empty(corpus, cfg, token)
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    if d.get('schema_version') != SCHEMA_VERSION and not allow_mismatch:
        raise ValueError(
            f'{path}: schema {d.get("schema_version")} != {SCHEMA_VERSION}')
    if d.get('config_hash') != cfg and not allow_mismatch:
        raise ValueError(
            f'{path}: config_hash {d.get("config_hash")} != {cfg}')
    return d


def save(root, corpus, cfg, token, store):
    path = store_path(root, corpus, cfg, token)
    store['schema_version'] = SCHEMA_VERSION
    store['config_hash'] = cfg
    store['token'] = token
    store['corpus'] = corpus
    _atomic_write(path, store)
    return path


def put(store, *, anchor, text='', standard=None, method=None,
        root_id=None, peak=0.0, state='retired',
        reason=None, thread=None, date=None, run_id=None, revivals=0):
    """Insert or update one commitment, keyed by its exact normalised anchor.

    `thread` is {'id': str, 'ordinal': int} -- the ordinal is what makes
    non-adjacency decidable during consolidation.
    """
    if not anchor:
        return None
    if state not in STATES:
        raise ValueError(f'unknown state {state!r}')
    key = anchor.strip()
    rec = store['records'].get(key)
    if rec is None:
        rec = {'anchor': key, 'text': text, 'standard': standard,
               'method': method, 'root_id': root_id,
               'peak': 0.0, 'state': state, 'reason': reason,
               'threads': [], 'first_seen': date, 'last_seen': date,
               'run_id': run_id, 'revivals': int(revivals or 0)}
        store['records'][key] = rec
    # Peak is a high-water mark: a commitment that once led and faded is a
    # better candidate than one that never left the floor.
    rec['peak'] = max(float(rec.get('peak', 0.0)), float(peak or 0.0))
    rec['state'] = state
    rec['last_seen'] = date or rec.get('last_seen')
    rec['revivals'] = max(int(rec.get('revivals', 0)), int(revivals or 0))
    if text:
        rec['text'] = text
    if standard:
        rec['standard'] = standard
    if method:
        rec['method'] = method
    if reason:
        rec['reason'] = reason
    if thread and not any(t.get('id') == thread.get('id')
                          for t in rec['threads']):
        rec['threads'].append(dict(thread))
    return rec


def evict(store, thread_ordinal, cap=CAP, decay=DECAY):
    """Trim each stratum to its cap by recency-weighted peak.

    Peak alone would let a commitment that led two years ago and never
    returned outrank one that led twice last month, which is the wrong
    ordering for a store whose job is to say what someone is pursuing NOW.
    At equal score a commitment that has been revived is never dropped ahead
    of one that never has -- returning is evidence.

    `decay` is a guess and is labelled as one: report the distribution of
    threads-since-last-seen at eviction before trusting it.
    """
    dropped = []
    rows = list(store['records'].items())
    if len(rows) > cap:

        def score(kv):
            v = kv[1]
            seen = [t.get('ordinal', 0) for t in (v.get('threads') or [])]
            gap = max(0, thread_ordinal - (max(seen) if seen else 0))
            return (int(v.get('revivals', 0)) > 0,
                    float(v.get('peak', 0.0)) * (decay ** gap))

        rows.sort(key=score, reverse=True)
        for k, v in rows[cap:]:
            dropped.append({'anchor': k, 'peak': v.get('peak'),
                            'threads_seen': len(v.get('threads') or [])})
            del store['records'][k]
    return dropped


def restorable(store, limit=None):
    """Commitments to seed a new run from, best first.

    Ordered by peak, but the weight does NOT travel: the caller restores these
    at uniform weight. Ordering is the only thing a stored weight may decide.
    """
    rows = [v for v in store['records'].values() if v.get('state') == 'live']
    rows.sort(key=lambda v: -float(v.get('peak', 0.0)))
    return rows[:limit] if limit else rows
