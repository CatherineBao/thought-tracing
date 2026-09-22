"""Re-derive every logged run's likelihoods with the corrected rank mapping.

Zero LLM calls. The rank scorer's ALLOCATION block is ambiguously keyed between
rank position and hypothesis index; parse_allocation read it as hypothesis
index, the model usually meant rank position, and the RANKING block that
settles it was parsed away and discarded. The full scorer output is stored per
step in the tracer dump, so the corrected likelihood is recoverable offline for
every run ever made.

WHAT THIS CAN AND CANNOT TELL YOU
---------------------------------
Exact: the per-step likelihood vector, and the accumulator arithmetic over it.
`accumulate_weights` is imported from tracer, not reimplemented, so the
arithmetic cannot drift from production.

NOT reproduced: resampling, split/merge, perturbation, expiry, and the
re-propagated belief texts. All of those depend on the weights, so a real run
under the fix diverges from this replay after the first step where the argmax
moves. Treat every number here as a screening result and a LOWER BOUND on the
change -- the operators amplify it, they do not damp it. The only way to get
the real trajectory is to re-run.

WHY IT READS THE TRACER DUMP, NOT THE STEPS FILE
------------------------------------------------
`_log_step` zips pre-operator likelihoods positionally onto the POST-operator
population, so on any step where an operator fired, `particles[i].likelihood`
in *.steps.jsonl belongs to a different particle. The tracer dump's
`weight_details` is the raw scorer output, pre-operator and self-consistent.

Usage:
  replay_likelihood.py                      # summary across every run
  replay_likelihood.py --run ep_Katara      # per-step detail for one run
  replay_likelihood.py --alpha 0.85 0.5 0   # sweep the prior exponent
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from typing import Any, Dict, List, Optional

from tracer import accumulate_weights, map_allocation_to_hypotheses, parse_ranking

TRACE_GLOBS = ("musing_out/traces/tracer-*.jsonl", "musing_out/tracer-*.jsonl")
RUNID = re.compile(r"runid-(.*?)_nhypotheses")


def ranks_of(scores: List[float]) -> Dict[int, int]:
    """1 = highest score. Mirrors tracer.py's descending-sort convention."""
    out = {}
    for pos, i in enumerate(sorted(range(len(scores)), key=lambda j: -scores[j]), start=1):
        out[i] = pos
    return out


def pearson(xs, ys) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return cov / den if den else None


def load_steps(path: str) -> List[Dict[str, Any]]:
    """The tracer dump is one JSON object whose 'hypotheses' is the step list."""
    try:
        with open(path, encoding="utf-8") as fh:
            blob = fh.read().strip()
        if not blob:
            return []
        d = json.loads(blob.split("\n")[-1])
    except (json.JSONDecodeError, OSError):
        return []
    steps = d.get("hypotheses")
    return steps if isinstance(steps, list) else []


def replay_run(steps: List[Dict[str, Any]], alpha=0.85, beta=1.0, eps_frac=0.12):
    """Recompute each step's likelihood under both readings, then accumulate.

    Returns per-step records plus the two weight trajectories. Steps whose
    RANKING did not parse are carried with old == new: nothing can be said
    about them, and silently dropping them would flatter the result.
    """
    out, accum_old, accum_new = [], {}, {}
    for idx, h in enumerate(steps):
        wd = h.get("weight_details") or {}
        raw_scores, raw_pred = wd.get("raw_scores"), wd.get("raw_predictions")
        lids = h.get("lineage_ids") or []
        anchors = h.get("anchors") or []
        if (not isinstance(raw_scores, list) or len(raw_scores) < 2
                or raw_scores[0] is None or not lids):
            continue
        try:
            old = [float(x) for x in raw_scores]
        except (TypeError, ValueError):
            continue
        n = len(old)
        if len(lids) < n:
            continue

        # The isolated scorer stores one response per hypothesis, so
        # raw_predictions is a list there and carries no RANKING block at all.
        # Those runs are reported as unresolved rather than silently skipped.
        if isinstance(raw_pred, str):
            order, rank_err = parse_ranking(raw_pred, n)
        else:
            order, rank_err = None, 'no rank-mode scorer output'
        if order is None:
            new, keying, agreement = list(old), 'unresolved', None
        else:
            new, keying, agreement = map_allocation_to_hypotheses(old, order)

        # Likelihoods are normalized inside accumulate_weights; an all-zero
        # verdict is a real one ("nothing predicts this") and stays flat.
        acc_o = accumulate_weights(lids[:n], old, accum_old, alpha, beta, eps_frac)
        acc_n = accumulate_weights(lids[:n], new, accum_new, alpha, beta, eps_frac)
        accum_old, accum_new = acc_o['accum'], acc_n['accum']

        ro, rn = ranks_of(old), ranks_of(new)
        out.append({
            'step': idx, 'n': n, 'keying': keying, 'agreement': agreement,
            'rank_error': rank_err, 'ranking': order,
            'anchors': [str(a) if a is not None else None for a in anchors[:n]],
            'old_scores': old, 'new_scores': new,
            'old_rank': [ro[i] for i in range(n)], 'new_rank': [rn[i] for i in range(n)],
            'old_argmax': max(range(n), key=lambda i: old[i]),
            'new_argmax': max(range(n), key=lambda i: new[i]),
            'model_first': order[0] if order else None,
            'old_w': acc_o['weights'], 'new_w': acc_n['weights'],
            'old_w_argmax': max(range(n), key=lambda i: acc_o['weights'][i]),
            'new_w_argmax': max(range(n), key=lambda i: acc_n['weights'][i]),
        })
    return out


def churn(recs, key):
    """Number of steps where the top-weighted ANCHOR changes."""
    last, c = None, 0
    for r in recs:
        a = r['anchors'][r[key]] if r[key] < len(r['anchors']) else None
        if last is not None and a != last:
            c += 1
        last = a
    return c


def summarize(recs):
    if not recs:
        return None
    idx, ro, rn = [], [], []
    for r in recs:
        for i in range(r['n']):
            idx.append(i)
            ro.append(r['old_rank'][i])
            rn.append(r['new_rank'][i])
    resolved = [r for r in recs if r['keying'] != 'unresolved']
    return {
        'steps': len(recs),
        'unresolved': sum(1 for r in recs if r['keying'] == 'unresolved'),
        'keyed_rank': sum(1 for r in recs if r['keying'] == 'rank'),
        'keyed_hyp': sum(1 for r in recs if r['keying'] == 'hypothesis'),
        'ambiguous': sum(1 for r in recs if r['keying'] == 'ambiguous'),
        'old_argmax_at_0': sum(1 for r in recs if r['old_argmax'] == 0),
        'new_argmax_at_0': sum(1 for r in recs if r['new_argmax'] == 0),
        # The headline: how often the OLD top pick disagreed with the model's
        # own stated 1st place, and how often the NEW one does.
        'old_disagree': sum(1 for r in resolved if r['old_argmax'] != r['model_first']),
        'new_disagree': sum(1 for r in resolved if r['new_argmax'] != r['model_first']),
        'resolved': len(resolved),
        'argmax_moved': sum(1 for r in recs if r['old_argmax'] != r['new_argmax']),
        'w_argmax_moved': sum(1 for r in recs if r['old_w_argmax'] != r['new_w_argmax']),
        'corr_idx_rank_old': pearson(idx, ro),
        'corr_idx_rank_new': pearson(idx, rn),
        'churn_old': churn(recs, 'old_w_argmax'),
        'churn_new': churn(recs, 'new_w_argmax'),
    }


def find_traces(pattern: Optional[str]):
    seen, paths = set(), []
    for g in TRACE_GLOBS:
        for p in sorted(glob.glob(g)):
            rid = RUNID.search(p)
            rid = rid.group(1) if rid else os.path.basename(p)
            if pattern and pattern not in rid:
                continue
            key = (rid, os.path.getsize(p))
            if key in seen:
                continue
            seen.add(key)
            paths.append((rid, p))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="substring of a run id; detail mode when it matches one run")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.85])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eps-frac", type=float, default=0.12)
    ap.add_argument("--steps", type=int, nargs="+", help="detail mode: only these step indices")
    a = ap.parse_args()

    paths = find_traces(a.run)
    if not paths:
        print("no tracer dumps matched"); return

    for alpha in a.alpha:
        if len(a.alpha) > 1:
            print(f"\n{'='*104}\nALPHA = {alpha}\n{'='*104}")
        tot = {k: 0 for k in ('steps', 'unresolved', 'old_argmax_at_0', 'new_argmax_at_0',
                              'old_disagree', 'new_disagree', 'resolved', 'argmax_moved',
                              'w_argmax_moved', 'churn_old', 'churn_new')}
        print(f"{'run':<22}{'steps':>6}{'argmax@0':>12}{'top!=1st':>12}"
              f"{'L moved':>9}{'w moved':>9}{'churn':>11}{'keying r/h/?':>14}")
        print(f"{'':22}{'':>6}{'old->new':>12}{'old->new':>12}{'':>9}{'':>9}{'old->new':>11}")
        print("-" * 104)
        rows = []
        for rid, p in paths:
            recs = replay_run(load_steps(p), alpha, a.beta, a.eps_frac)
            s = summarize(recs)
            if not s:
                continue
            rows.append((rid, s, recs))
            for k in tot:
                tot[k] += s[k]
            print(f"{rid[:21]:<22}{s['steps']:>6}"
                  f"{s['old_argmax_at_0']:>6}->{s['new_argmax_at_0']:<5}"
                  f"{s['old_disagree']:>6}->{s['new_disagree']:<5}"
                  f"{s['argmax_moved']:>9}{s['w_argmax_moved']:>9}"
                  f"{s['churn_old']:>5}->{s['churn_new']:<5}"
                  f"{s['keyed_rank']:>6}/{s['keyed_hyp']}/{s['unresolved']}")
        print("-" * 104)
        print(f"{'TOTAL':<22}{tot['steps']:>6}"
              f"{tot['old_argmax_at_0']:>6}->{tot['new_argmax_at_0']:<5}"
              f"{tot['old_disagree']:>6}->{tot['new_disagree']:<5}"
              f"{tot['argmax_moved']:>9}{tot['w_argmax_moved']:>9}"
              f"{tot['churn_old']:>5}->{tot['churn_new']:<5}")

        allrec = [r for _, _, recs in rows for r in recs]
        if allrec:
            idx = [i for r in allrec for i in range(r['n'])]
            ro = [r['old_rank'][i] for r in allrec for i in range(r['n'])]
            rn = [r['new_rank'][i] for r in allrec for i in range(r['n'])]
            co, cn = pearson(idx, ro), pearson(idx, rn)
            print(f"\ncorr(array index, likelihood rank)   old {co:+.3f}   new {cn:+.3f}")
            print("A rank determined by array position is not evidence about the hypothesis.")
            res = tot['resolved'] or 1
            print(f"top pick disagreed with the model's own 1st place: "
                  f"{tot['old_disagree']}/{res} -> {tot['new_disagree']}/{res}")

    if not a.run:
        return
    # A run id can match more than one dump -- ep_Katara executed twice
    # concurrently, and those are two separate samples, not a stale copy.
    for rid, p in paths:
        recs = replay_run(load_steps(p), a.alpha[0], a.beta, a.eps_frac)
        print(f"\n\n{'='*104}\nPER-STEP DETAIL: {rid}  ({os.path.basename(p)[:60]})\n{'='*104}")
        for r in recs:
            if a.steps and r['step'] not in a.steps:
                continue
            oa, na = r['old_argmax'], r['new_argmax']
            moved = "  <-- ARGMAX MOVED" if oa != na else ""
            print(f"\nstep {r['step']:>3}  keying={r['keying']:<11} "
                  f"agreement={r['agreement'] if r['agreement'] is None else round(r['agreement'],3)}{moved}")
            print(f"   old top: [{oa}] w={r['old_w'][oa]:.3f}  {str(r['anchors'][oa])[:58]}")
            print(f"   new top: [{na}] w={r['new_w'][na]:.3f}  {str(r['anchors'][na])[:58]}")


if __name__ == "__main__":
    main()
