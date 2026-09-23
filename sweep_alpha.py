"""Read an alpha sweep off disk: does a stickier prior buy stability, or
just a frozen board?

alpha is the prior's exponent in `w_t ~ w_{t-1}^alpha * L_t^beta`. It was
0.85 on all 133 runs that record it, a prior half-life of 4.3 steps -- the
filter forgets half of what it believed within four turns. P-13 attributed
70% of argmax churn to reweighting rather than to any operator, which makes
alpha the one term that could move it.

THE TRAP THIS EXISTS TO AVOID. Raising alpha MUST lower churn: at alpha=1.0
the prior never decays and the leader can barely move. So a fall in churn is
not evidence of anything on its own -- a filter that has stopped updating
scores perfectly on the churn metric while being strictly worse at its job.

Every column after the first is there to catch that:

  distinct_roots   population still exploring, or collapsed to one idea
  surprise         still notices when no commitment explains a turn
  reversals        evidence can still overturn a leader on the accumulator
  posterior_ess    rises when the posterior is merely being smoothed
  resamples        falls for the same reason

A result is only a result if churn falls while the rest hold.
"""
import argparse
import collections
import glob
import json
import statistics as st


def read(run_id):
    paths = glob.glob(f'musing_out/runs/**/{run_id}*.steps.jsonl', recursive=True)
    if not paths:
        return None
    rows = [json.loads(l) for l in open(paths[0]) if l.strip()]
    if not rows:
        return None
    prev, churn, n = None, 0, 0
    roots, surprise, resamples = set(), 0, 0
    ess = []
    for d in rows:
        ps = d.get('particles') or []
        if not ps:
            continue
        top = max(ps, key=lambda p: p.get('weight') or 0)
        r = top.get('root_id')
        for p in ps:
            roots.add(p.get('root_id'))
        if d.get('surprise'):
            surprise += 1
        if 'resample' in (d.get('operators_fired') or []):
            resamples += 1
        if d.get('posterior_ess') is not None:
            ess.append(float(d['posterior_ess']))
        if prev is not None:
            n += 1
            churn += (r != prev)
        prev = r
    if not n:
        return None
    return {'churn_per_100': 100 * churn / n, 'churn': churn, 'steps': n,
            'roots': len(roots), 'surprise': surprise,
            'resamples': resamples,
            'post_ess': st.median(ess) if ess else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='alpha_')
    ap.add_argument('--alphas', default='0.85,0.90,0.95,0.98')
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()

    alphas = a.alphas.split(',')
    seeds = a.seeds.split(',')
    table = collections.OrderedDict()
    for al in alphas:
        tag = al.replace('.', '')
        rows = [read(f'{a.prefix}a{tag}s{s}') for s in seeds]
        rows = [r for r in rows if r]
        if rows:
            table[al] = rows

    if not table:
        raise SystemExit('no runs found on disk yet')

    print(f"{'alpha':>6}{'half-life':>11}{'churn/100':>11}{'spread':>9}"
          f"{'roots':>7}{'surpr':>7}{'resamp':>8}{'postESS':>9}{'n':>4}")
    base = None
    for al, rows in table.items():
        import math
        hl = math.log(0.5) / math.log(float(al))
        ch = [r['churn_per_100'] for r in rows]
        m = st.mean(ch)
        if base is None:
            base = m
        print(f"{al:>6}{hl:>11.1f}{m:>11.1f}{max(ch)-min(ch):>9.1f}"
              f"{st.mean(r['roots'] for r in rows):>7.1f}"
              f"{st.mean(r['surprise'] for r in rows):>7.1f}"
              f"{st.mean(r['resamples'] for r in rows):>8.1f}"
              f"{st.mean(r['post_ess'] or 0 for r in rows):>9.2f}{len(rows):>4}")

    print('\nCHURN REDUCTION vs alpha=0.85, against its own seed spread:')
    b = table[alphas[0]]
    bs = [r['churn_per_100'] for r in b]
    floor = max(bs) - min(bs)
    print(f"  baseline seed spread (the floor any effect must clear): {floor:.1f}")
    for al, rows in list(table.items())[1:]:
        ch = [r['churn_per_100'] for r in rows]
        d = st.mean(bs) - st.mean(ch)
        verdict = 'clears' if d > floor else 'WITHIN NOISE'
        print(f"  alpha={al}: -{d:>5.1f} per 100   {verdict}")

    print('\nRESPONSIVENESS CHECK -- did it fall by freezing?')
    br, bsurp = st.mean(r['roots'] for r in b), st.mean(r['surprise'] for r in b)
    for al, rows in list(table.items())[1:]:
        rr = st.mean(r['roots'] for r in rows)
        ss = st.mean(r['surprise'] for r in rows)
        flag = []
        if rr < 0.7 * br:
            flag.append(f'roots {rr:.0f} vs {br:.0f} -- population collapsed')
        if ss < 0.7 * bsurp:
            flag.append(f'surprise {ss:.0f} vs {bsurp:.0f} -- stopped noticing misfit')
        print(f"  alpha={al}: " + ('; '.join(flag) if flag else 'roots and surprise hold'))


if __name__ == '__main__':
    main()
