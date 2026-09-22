"""Does the margin vector on a SURPRISE step carry an ordering, or manufacture one?

    python audit_margins.py                       # every run that logged a null
    python audit_margins.py --runs v6_ full_      # id prefixes

THE QUESTION. On a surprise step every live commitment lost to the null, so
every margin is <= 0 and the filter discards the lightest quarter by WEIGHT
(tracer.py, surprise branch of the perturb trigger) -- the margin vector is
computed, logged as `best_margin` only, and otherwise thrown away. Selecting the
discard set by margin instead would spend the one piece of evidence the step
actually produced. But the rank scorer is told to commit to a strict order with
NO TIES, and a surprise step is the worst case for that instruction: everything
failed, so any ordering among the failures may be confabulated. probe_scorer.py
guards the same property for identical hypotheses; this is the same check for
the losing tail.

NO RE-RUN NEEDED. `margins` never reached disk, but `allocation_raw`, `ranking`
and `baseline_score` all did, so the vector reconstructs exactly as
map_allocation_to_hypotheses(allocation_raw, ranking) - baseline_score, using
the SAME function production uses rather than a copy of it.

THREE TESTS, because spread alone proves nothing -- a sampler asked for a strict
order will always produce one:

  spread      how far apart the BOTTOM-k margins sit, surprise steps against
              ordinary ones. Ordinary steps are the control: the same scorer,
              the same forced ranking, but with a genuine winner at the top. If
              the surprise tail is flatter, the order is thinner there than the
              scorer's usual. Same k on both sides -- spread grows with set size,
              so comparing a whole surprise slate against the two-or-three
              losers of an ordinary step would manufacture the result. k is the
              quartile the operator would actually discard.
  persistence a root ranked worst at t, ranked again at t+1: signal persists,
              confabulation resamples. Scored against the chance rate for a set
              of that size, which is the noise floor the spread number lacks.
  overlap     how often margin-selection and weight-selection pick the SAME
              particles, against the k/n a pair of independent picks would share
              by chance. If they mostly agree the change is a no-op however real
              the ordering is, and is not worth the code.
"""

import argparse
import glob
import json
import os
import statistics as st

import musing_layout as ml
from tracer import map_allocation_to_hypotheses


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def margins_of(rec):
    """Per-hypothesis margin over the null, in HYPOTHESIS order, or None.

    n comes from weights_pre, not from `particles`: particles is the
    POST-operator population and a step that perturbed, split or merged has a
    different length there than the slate that was actually scored.
    """
    alloc = rec.get("allocation_raw")
    order = rec.get("ranking")
    base = rec.get("baseline_score")
    pre = rec.get("weights_pre")
    if not alloc or not order or base is None or not pre:
        return None, None
    n = len(pre)
    n_extra = len(alloc) - n
    if n_extra not in (1, 2) or len(order) != len(alloc):
        return None, None
    scores, _keying, _agree = map_allocation_to_hypotheses(alloc, order)
    if scores is None or len(scores) != len(alloc):
        return None, None
    return [float(x) - float(base) for x in scores[:n]], n


def spread(v):
    return (max(v) - min(v)) if len(v) > 1 else 0.0


def bottom(vals, k):
    """Indices of the k smallest, ties broken by index (stable, like the filter)."""
    return set(sorted(range(len(vals)), key=lambda j: (vals[j], j))[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=ml.OUT_DIR)
    ap.add_argument("--runs", nargs="*", default=None, help="run-id prefixes")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.out_dir, "**", "*.steps.jsonl"),
                             recursive=True))
    if a.runs:
        files = [f for f in files
                 if any(os.path.basename(f).startswith(p) for p in a.runs)]

    sur_tail, ord_tail = [], []       # spread of the LOSING margins
    sur_flat, ord_flat = 0, 0         # steps whose losing tail is perfectly flat
    overlap_hits, overlap_tot, same_set = 0, 0, 0
    overlap_chance = []
    persist_hit, persist_tot, persist_chance = 0, 0, []
    n_sur, n_ord, n_runs = 0, 0, 0

    for f in files:
        recs = load(f)
        per_run = []
        for rec in recs:
            if rec.get("surprise") is None:
                continue
            m, n = margins_of(rec)
            if m is None or n < 2:
                continue
            k = max(1, n // 4)
            # The comparable quantity on both sides is the spread of the set the
            # operator would DISCARD, at the same size in both regimes.
            tail = sorted(m)[:max(2, k)]
            if rec.get("surprise"):
                n_sur += 1
                sur_tail.append(spread(tail))
                sur_flat += 1 if spread(tail) == 0 else 0
                # the filter's own selection: lightest by the accumulated weight
                # it held when the trigger evaluated. A resample between the
                # scorer and the operator overwrites those weights with 1/n, so
                # those steps say nothing about the comparison.
                if "resample" not in (rec.get("operators_fired") or []):
                    w = [float(x) for x in rec["weights_pre"]]
                    by_w, by_m = bottom(w, k), bottom(m, k)
                    overlap_hits += len(by_w & by_m)
                    overlap_tot += k
                    overlap_chance.append(k / n)
                    same_set += 1 if by_w == by_m else 0
                per_run.append((rec, m, n, k))
            else:
                n_ord += 1
                ord_tail.append(spread(tail))
                ord_flat += 1 if spread(tail) == 0 else 0

        # persistence: a root in the worst quartile on one surprise step, and
        # on the next one it is scored on. Roots come from the pre-operator
        # snapshot, which is the population the margins belong to.
        for (r1, m1, n1, k1), (r2, m2, n2, _k) in zip(per_run, per_run[1:]):
            p1 = [p["root_id"] for p in (r1.get("pre_operator_particles") or [])]
            p2 = [p["root_id"] for p in (r2.get("pre_operator_particles") or [])]
            if len(p1) != n1 or len(p2) != n2:
                continue
            worst1 = {p1[j] for j in bottom(m1, k1)}
            k2 = max(1, n2 // 4)
            worst2 = {p2[j] for j in bottom(m2, k2)}
            survivors = worst1 & set(p2)
            if not survivors:
                continue
            persist_hit += len(survivors & worst2)
            persist_tot += len(survivors)
            persist_chance.append(k2 / n2)
        if per_run:
            n_runs += 1

    def med(v):
        return st.median(v) if v else float("nan")

    print(f"{n_runs} run(s): {n_sur} surprise step(s), {n_ord} ordinary\n")
    print("SPREAD of the discardable tail, same k both sides (0-100 scale)")
    print(f"  surprise steps  median {med(sur_tail):6.1f}   flat: "
          f"{sur_flat}/{len(sur_tail)}")
    print(f"  ordinary steps  median {med(ord_tail):6.1f}   flat: "
          f"{ord_flat}/{len(ord_tail)}")
    if sur_tail and ord_tail and med(ord_tail):
        print(f"  -> the surprise tail is {med(sur_tail) / med(ord_tail):.2f}x "
              f"the ordinary one")

    print("\nPERSISTENCE of the worst quartile, surprise step to the next")
    if persist_tot:
        rate = persist_hit / persist_tot
        floor = sum(persist_chance) / len(persist_chance)
        print(f"  {persist_hit}/{persist_tot} = {rate:.2f} against a chance "
              f"floor of {floor:.2f}   ({rate / floor:.2f}x)")
    else:
        print("  no consecutive surprise steps sharing a root")

    print("\nOVERLAP with the weight-based set the filter discards today")
    if overlap_tot:
        fl = sum(overlap_chance) / len(overlap_chance)
        print(f"  {overlap_hits}/{overlap_tot} = {overlap_hits / overlap_tot:.2f} "
              f"of picks shared, against {fl:.2f} by chance")
        print(f"  identical set on {same_set}/{len(overlap_chance)} step(s)")
    else:
        print("  no eligible surprise steps")


if __name__ == "__main__":
    main()
