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
