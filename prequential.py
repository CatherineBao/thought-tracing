"""Prequential scoring for the choice-point filter, and Gate 0.

GATE 0 -- run this before anything else is written. Zero model calls.

The Milestone 2 gate was originally worded as "motive-marginal churn far below
the old 55.6 argmax churn". That is not a legal comparison on its face: a new
metric is always free to look better than an old one, and PREREG standing rules
1 and 3 both bite. Gate 0 settles whether the comparison means anything, for
free, because the answer is already on disk:

    a motive's marginal mass on a ONE-MOTIVE portfolio IS its root's mass,
    every v1 particle IS a one-motive portfolio,
    and read_steps rehydrates old logs against the newer schema.

So motive-marginal churn is computable RETROSPECTIVELY on the frozen v1
production pool, through the same portfolio.py code path v2 will use. If it
reproduces the frozen figure, the metric is measuring the same quantity and the
M2 comparison is like-for-like. If it comes back near zero, churn-invariance is a
property of the METRIC rather than of portfolios, and the gate would pass on a
system that changed nothing -- in which case M2 must be restated before any
spend.

ONE CORRECTION TO THE PLAN, FOUND BY WRITING THIS. The plan asserts that
baseline_metrics.json's 56.41 is computed on lineage_id by
trace_log.argmax_churn. It is not. baseline_snapshot.read_run computes its own
churn on `top_root` taken from `roots_of()`, which sums weight per ROOT -- so the
frozen 56.41 is already root-level, and the like-for-like number exists.
trace_log.argmax_churn is a different, lineage-level metric; both are reported
below so the two are never swapped for one another again.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics as st

import musing_layout as ml
import portfolio as P

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline_metrics.json")

# Same floor baseline_snapshot uses: below this a run's churn rate is one or two
# events of noise, and pooling it with a 40-step run misreports both.
MIN_STEPS = 10


# --------------------------------------------------------------------------
# Gate 0
# --------------------------------------------------------------------------

def steps_of(path):
    steps = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [s for s in steps if s.get("particles")]


def marginal_series(steps):
    """Per-step marginal mass, reading each v1 particle as a one-motive portfolio.

    Keyed on root_id, which for a single-anchor particle IS the commitment's
    identity -- hypothesis.update_anchor's ANCHOR IDENTITY == ROOT IDENTITY rule
    is precisely what makes this substitution exact rather than approximate.

    Computed here rather than by constructing real Portfolio objects because the
    logs carry no motive_id and inventing one per step would give every step a
    fresh identity and report churn of 100 per 100 by construction.
    """
    series = []
    for s in steps:
        mass = collections.Counter()
        for p in s.get("particles") or []:
            rid = p.get("root_id") or p.get("lineage_id") or p.get("particle_id")
            if rid is None:
                continue
            mass[rid] += float(p.get("weight") or 0.0)
        if mass:
            series.append(dict(mass))
    return series


def lineage_churn_per_100(steps, tol=1e-9):
    """Churn of the top PARTICLE, keyed on lineage. Reported for contrast only.

    Ties carry the previous leader forward, matching trace_log.argmax_churn:
    after a resample every particle holds 1/N, so a naive max() reports churn on
    almost every step as an artifact of tie-breaking.
    """
    changes = transitions = 0
    prev = None
    for s in steps:
        ws = [(float(p.get("weight") or -1.0), p.get("lineage_id") or p.get("particle_id"))
              for p in s.get("particles") or []]
        if not ws:
            continue
        ws.sort(key=lambda t: -t[0])
        if len(ws) > 1 and abs(ws[0][0] - ws[1][0]) <= tol:
            continue
        top = ws[0][1]
        if prev is not None:
            transitions += 1
            changes += (top != prev)
        prev = top
    return (100.0 * changes / transitions) if transitions else None


def gate0(pool="production_config"):
    """Recompute marginal churn on the v1 logs and diff against the frozen figure."""
    import baseline_snapshot as bs

    meta = bs.run_meta()
    rows = []
    for path in sorted(glob.glob(os.path.join(ml.OUT_DIR, "**", "*.steps.jsonl"),
                                 recursive=True)):
        steps = steps_of(path)
        if len(steps) < MIN_STEPS:
            continue
        run = os.path.basename(path).replace(".steps.jsonl", "")
        m = meta.get(run)
        production = bool(m and bs.is_production(m))
        if pool == "production_config" and not production:
            continue
        series = marginal_series(steps)
        mc = P.marginal_churn(series)
        if mc is None:
            continue
        rows.append({
            "run": run,
            "steps": len(steps),
            "marginal_churn_per_100": mc,
            "lineage_churn_per_100": lineage_churn_per_100(steps),
            "distinct_motive_roots": len({r for s in series for r in s}),
        })

    if not rows:
        return {"pool": pool, "runs": 0,
                "verdict": "NO RUNS -- cannot read the pool, so nothing is established"}

    marg = [r["marginal_churn_per_100"] for r in rows]
    lin = [r["lineage_churn_per_100"] for r in rows if r["lineage_churn_per_100"] is not None]

    frozen = None
    if os.path.exists(BASELINE):
        b = json.load(open(BASELINE, encoding="utf-8"))
        frozen = (b.get("vs_prechecks", {}).get("argmax_churn_per_100", {}) or {}).get("here")

    median = st.median(marg)
    delta = None if frozen is None else round(median - frozen, 4)

    # The verdict is about whether the METRIC is free, not about whether the
    # filter is good. Three bands, fixed here rather than after looking:
    #   near zero      -> churn-invariance is the metric's property. M2 restated.
    #   reproduces     -> same quantity as the frozen root-level figure. M2 legal.
    #   diverges       -> the two are not the same quantity. Say so and find out
    #                     why BEFORE either is used as a gate.
    if median < 5.0:
        verdict = ("VACUOUS -- marginal churn is already ~0 on v1 logs, so the M2 gate "
                   "would pass on a system that changed nothing. RESTATE M2 before any spend.")
    elif frozen is not None and abs(median - frozen) <= 2.0:
        verdict = ("REPRODUCES the frozen root-level figure. Marginal churn and the "
                   "baseline's argmax churn are the same quantity on one-motive "
                   "portfolios, so the M2 comparison is like-for-like.")
    elif frozen is None:
        verdict = "NO FROZEN FIGURE to compare against; median reported without a verdict."
    else:
        verdict = (f"DIVERGES from the frozen figure by {delta:+.2f} per 100. These are "
                   "not the same quantity -- resolve why before either is used as a gate.")

    return {
        "pool": pool,
        "runs": len(rows),
        "steps": sum(r["steps"] for r in rows),
        "marginal_churn_per_100": {
            "median": round(median, 4),
            "mean": round(st.mean(marg), 4),
            "min": round(min(marg), 4),
            "max": round(max(marg), 4),
            "runs_at_zero": sum(1 for v in marg if v == 0.0),
        },
        "lineage_churn_per_100": (
            {"median": round(st.median(lin), 4), "runs": len(lin)} if lin else None),
        "frozen_argmax_churn_per_100": frozen,
        "delta_vs_frozen": delta,
        "verdict": verdict,
        "note": ("baseline_snapshot.read_run computes churn on top_root from roots_of(), "
                 "which sums weight per ROOT -- so the frozen figure is root-level, not "
                 "lineage-level. trace_log.argmax_churn is the lineage-level metric and is "
                 "reported separately above; the two are never interchangeable."),
        "per_run": sorted(rows, key=lambda r: -r["marginal_churn_per_100"]),
    }


# --------------------------------------------------------------------------
# what makes a moment a CHOICE -- clause 4, measured
# --------------------------------------------------------------------------
#
# PREREG defines a choice as a moment, identifiable from what came before, where
# the person had at least two genuinely feasible and distinguishable courses of
# action, AND WHERE PLAUSIBLE MOTIVES WOULD FAVOUR DIFFERENT ONES. The first
# three clauses are properties of the extraction rule and are checked there. The
# fourth is a property of the moment that can only be read after the forecasts
# exist, so it is measured here and REPORTED, never used to select.


def _h(p, base=2.0):
    """Shannon entropy of a distribution given as a mapping or a sequence."""
    vals = list(p.values()) if hasattr(p, 'values') else list(p)
    tot = sum(vals)
    if tot <= 0:
        return 0.0
    out = 0.0
    for v in vals:
        q = v / tot
        if q > 0:
            out -= q * math.log(q, base)
    return out


def forecast_disagreement(distributions, weights=None, normalise=False):
    """How much the competing accounts disagree at one choice point, in bits.

    Jensen-Shannon divergence of the population's forecasts:

        D = H( sum_i w_i p_i )  -  sum_i w_i H( p_i )

    which is exactly the mutual information between "which account is right" and
    "which action is taken". It is zero if and only if every particle forecasts
    identically -- i.e. the moment does not distinguish the hypotheses, whatever
    the transcript looks like -- and it is bounded above by H(w) <= log2(n).

    THIS IS A DIAGNOSTIC, NOT A FILTER. Dropping low-disagreement points would be
    selection on the model's own output: it would discard exactly the moments the
    population found uninformative and report the average of what is left as if
    it were the average of the task. The same error as conditioning on "the
    majority action was not taken". Points are STRATIFIED by it and both strata
    are reported.

    normalise=True divides by the TIGHT bound, min(H(w), log2 k), giving the
    fraction of the disagreement this point could possibly carry.

    BOTH TERMS MATTER AND DIVIDING BY H(w) ALONE IS WRONG. D is bounded by the
    weight entropy -- a population with one dominant account cannot disagree
    much whatever it believes -- and ALSO by log2 k, because D <= H(mixture) and
    a mixture over k options cannot exceed log2 k bits. Two confounds follow,
    and the raw figure is comparable across neither:

      choice type  a type offering more options can score higher for free, so a
                   single corpus-wide median cut would partly sort points BY
                   TYPE rather than by disagreement. Stratify within type, or
                   normalise.
      position     weights start uniform and concentrate as a run proceeds, so
                   D falls over a run by construction. Early points would land
                   in the high stratum on position alone, which is why
                   `position` is recorded on every point and any concentration
                   of lift in the high stratum is checked against it.
    """
    dists = [d for d in distributions if d]
    if len(dists) < 2:
        return 0.0
    keys = sorted({k for d in dists for k in d})
    if weights is None:
        weights = [1.0 / len(dists)] * len(dists)
    weights = [float(w) for w in weights][:len(dists)]
    tot = sum(weights)
    if tot <= 0:
        return 0.0
    weights = [w / tot for w in weights]

    mixture = {k: sum(w * d.get(k, 0.0) for w, d in zip(weights, dists)) for k in keys}
    d = _h(mixture) - sum(w * _h(d) for w, d in zip(weights, dists))
    d = max(0.0, d)                      # float error only; the quantity is >= 0
    if normalise:
        bound = min(_h(weights), math.log(len(keys), 2) if len(keys) > 1 else 0.0)
        return (d / bound) if bound > 0 else 0.0
    return d


def option_separability(distributions):
    """Per option PAIR, the largest gap any single account puts between them.

    Clause 3 of the definition -- "distinguishable" -- stated operationally: if
    no account in the population ever separates two listed options, then as far
    as anything here can tell they are one action written twice, and offering
    both inflates the option count without adding a decision.

    Reads the MAXIMUM over accounts rather than the mean, deliberately: one
    account that separates a pair is enough to make the pair a real fork, and a
    mean would let a crowd of indifferent accounts hide it.

    Caveat, and it is a real one: this measures whether the MODEL separates the
    options, not whether a person would. A pair that scores zero is a prompt to
    look at the option set by hand, not a proof that the options are identical.
    """
    dists = [d for d in distributions if d]
    keys = sorted({k for d in dists for k in d})
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[(a, b)] = max((abs(d.get(a, 0.0) - d.get(b, 0.0)) for d in dists),
                              default=0.0)
    return out


def disagreement_cut(dev_values, quantile=0.5):
    """The low/high split point, FROZEN ON DEV before test is read.

    A median split is a rule, not a number, so it is stated as the rule and the
    resolved value is written down. Re-cutting on test would let the stratum
    boundary move to wherever the lift happened to be.
    """
    vals = sorted(float(v) for v in dev_values)
    if not vals:
        return 0.0
    k = max(0, min(len(vals) - 1, int(round(quantile * (len(vals) - 1)))))
    return vals[k]


def stratify_by_disagreement(points, cut, key='disagreement', within=None):
    """Split points into the two pre-registered strata. Both are reported.

    `within` names a field to stratify WITHIN -- normally the choice type. A
    single corpus-wide cut on a raw (unnormalised) statistic would partly sort
    by type, because a type offering more options can score higher for free.
    Passing within='kind' resolves a separate cut per type; `cut` may then be a
    mapping from type to cut, or a single value applied to each.

    Nothing is ever dropped: every point lands in exactly one stratum, and a
    point with no recorded disagreement falls in LOW, because treating an
    unmeasured point as high would quietly promote it into the stratum the
    claim rests on.
    """
    if within is None:
        lo = [p for p in points if float(p.get(key) or 0.0) <= cut]
        hi = [p for p in points if float(p.get(key) or 0.0) > cut]
        return {'cut': cut, 'within': None, 'low': lo, 'high': hi,
                'n_low': len(lo), 'n_high': len(hi)}
    lo, hi, cuts = [], [], {}
    for p in points:
        g = p.get(within)
        c = cut.get(g, 0.0) if isinstance(cut, dict) else cut
        cuts[g] = c
        (lo if float(p.get(key) or 0.0) <= c else hi).append(p)
    return {'cut': cuts, 'within': within, 'low': lo, 'high': hi,
            'n_low': len(lo), 'n_high': len(hi)}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate0", action="store_true",
                    help="recompute marginal churn on the frozen v1 pool (no model calls)")
    ap.add_argument("--pool", default="production_config",
                    choices=["production_config", "all"])
    ap.add_argument("--json", action="store_true", help="emit the full record")
    a = ap.parse_args()

    if not a.gate0:
        ap.error("nothing to do; --gate0 is the only implemented mode so far")

    out = gate0(a.pool)
    if a.json:
        print(json.dumps(out, indent=2))
        return

    print(f"GATE 0 -- marginal churn on the v1 logs, pool={out['pool']}\n")
    if not out.get("runs"):
        print(out["verdict"])
        return
    mc = out["marginal_churn_per_100"]
    print(f"  runs                       {out['runs']}   steps {out['steps']}")
    print(f"  marginal churn / 100       median {mc['median']}  mean {mc['mean']}  "
          f"range {mc['min']}-{mc['max']}")
    print(f"  runs at zero churn         {mc['runs_at_zero']}")
    if out["lineage_churn_per_100"]:
        print(f"  lineage churn / 100        median {out['lineage_churn_per_100']['median']}  "
              f"(different metric, shown so the two are not swapped)")
    print(f"  frozen argmax churn / 100  {out['frozen_argmax_churn_per_100']}   "
          f"delta {out['delta_vs_frozen']:+}" if out["delta_vs_frozen"] is not None else "")
    print(f"\n  VERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()
