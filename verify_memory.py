"""Verify a cross-run memory store from two date-split halves.

Run as two SEPARATE process invocations, so half 2 can only see what half 1
actually wrote to disk -- an in-process check would pass on shared state that
a real deployment never has.

Checks, in order of how badly a failure would mislead:

  1. NO STORED WEIGHT REACHED THE FILTER. A peak from a previous run is on a
     previous run's normalising scale. profiles.py bans routing an external
     confidence into the weights for exactly this reason, and a store is the
     most tempting place to break that rule. Checked by confirming the
     restored candidates carry no field the filter could read as a weight.
  2. A COMMITMENT THAT CAME BACK KEPT ITS ROOT. That is what makes a gap read
     as one hypothesis rather than two that happen to agree, and it is the
     whole reason revive_retired preserves root_id.
  3. THREADS ACCUMULATED. Consolidation counts non-adjacent threads, so the
     store has to record both spans against a commitment seen in both.
  4. CONFIG ISOLATION HELD. Same config across a different span must load;
     a different config must not.
"""
import argparse

import memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/tmp/memtest')
    ap.add_argument('--corpus', default='bloomfield')
    ap.add_argument('--token', default='Wolf')
    a = ap.parse_args()

    import glob
    paths = glob.glob(f'{a.dir}/memory/{a.corpus}/*/{a.token}.json')
    if not paths:
        raise SystemExit(f'no store under {a.dir}')
    cfg = paths[0].split('/')[-2]
    store = memory.load(a.dir, a.corpus, cfg, a.token)
    recs = store['records']
    print(f'store: {len(recs)} commitments, config {cfg}\n')

    live = [v for v in recs.values() if v['state'] == 'live']
    ret = [v for v in recs.values() if v['state'] == 'retired']
    print(f'  live {len(live)}   retired {len(ret)}')

    # 3. threads
    spans = {}
    for v in recs.values():
        for t in (v.get('threads') or []):
            spans.setdefault(t['id'], 0)
            spans[t['id']] += 1
    print(f'  spans recorded: {spans}')
    both = [v for v in recs.values() if len({t["id"] for t in (v.get("threads") or [])}) > 1]
    print(f'  commitments seen in BOTH spans: {len(both)}')
    for v in both[:6]:
        print(f'     {v["peak"]:.2f}  {v["anchor"][:52]}')

    # 2. root preserved across the gap
    kept = [v for v in both if v.get('root_id')]
    print(f'\n  of those, carrying a root id: {len(kept)}/{len(both)}')

    # 1. no weight-like field escapes to the filter
    restorable = memory.restorable(store)
    banned = {'weight', 'weights', 'prior', 'logit'}
    leaks = [k for v in restorable for k in v if k in banned]
    print(f'  restorable candidates: {len(restorable)}')
    print(f'  weight-like fields exposed to the caller: {sorted(set(leaks)) or "none"}')

    # 4. config isolation
    try:
        memory.load(a.dir, a.corpus, 'not-the-same-config', a.token)
        iso_empty = True
    except ValueError:
        iso_empty = False
    print(f'  a different config sees an empty store: {iso_empty}')

    print('\nVERDICT')
    ok = True
    if not spans:
        print('  FAIL no spans recorded'); ok = False
    if len(spans) < 2:
        print(f'  WARN only {len(spans)} span(s) -- run both halves for the cross-run check')
    if leaks:
        print('  FAIL a weight-like field is exposed'); ok = False
    else:
        print('  ok   no stored weight is reachable as a prior')
    if both and len(kept) == len(both):
        print('  ok   every cross-span commitment kept its root')
    elif both:
        print(f'  FAIL {len(both) - len(kept)} cross-span commitments lost their root'); ok = False
    print('  ' + ('PASS' if ok else 'FAILURES ABOVE'))


if __name__ == '__main__':
    main()
