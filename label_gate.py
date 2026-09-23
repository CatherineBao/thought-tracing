"""The hard pre-spend gate on the label distribution. Dev only, exits non-zero.

WHY THIS EXISTS. Phase CP spent 2,481 forecasts before anyone measured the thing
that decided the result: `HOLD` was ~70% of every corpus, so a constant scored
0.656 while the whole 156-motive apparatus reached 0.653 and the pre-registered
bar (`obvious`) sat at 0.533 -- twelve points BELOW a constant. A forecasting
benchmark that is 70% one label cannot distinguish a person model from a
constant, whatever the method. That is not a finding about motives; it is a
finding about the task, and it was available for free before the first call.

Nothing here makes a model call except the two arms in §"directional gates",
which run on a 100-point dev sample before extraction is scaled.

TWO ARMS THAT LOOK ALIKE AND POINT IN OPPOSITE DIRECTIONS. Conflating them is
the failure this module is shaped around, and it is the third inverted-direction
bug in this design, so the directions are declared as data (GATES) and asserted
by test_gate_directions.py rather than living in prose.

    options_only     sees the option labels and NOTHING else.
                     Must NOT beat the class prior. If it does, the option
                     PHRASING carries the answer and the run is void. This is
                     Phase CP's tripwire, corrected: it originally compared
                     blind to CHANCE, which conflates an option set whose
                     wording gives the answer away with one where the modal
                     action is usually right. A blind guesser recovers the class
                     prior for free. Measured: blind 0.622 beat chance 0.411 and
                     VOIDED the run, but LOST to majority 0.656 -- so the
                     diagnosis was a degenerate label distribution, not leakage,
                     and re-running the same extraction would have fixed
                     nothing.

    context_neutral  sees the full transcript plus an equal-length NEUTRAL block.
                     Must NOT be WORSE than the class prior -- a model holding
                     the whole conversation that loses to a constant is not a
                     bar anything should be measured against, and that is
                     exactly what Phase CP's `obvious` did at 0.533 against
                     0.656. And it must leave HEADROOM: if it already sits at
                     ceiling there is nothing for a motive to add, which is the
                     plausible outcome on CaSiNo, where negotiators state their
                     priorities out loud.

RUN IT PER CHOICE TYPE. A corpus can pass in aggregate while one kind of choice
inside it is a constant. CaSiNo deal responses are plausibly dominated by
`Accept`, and pooling them with concessions would hide that.

THE PER-PERSON CHECK IS POOLED, NOT PER-PERSON. "Per-person majority share <=
0.60 for everyone" is undefined at 1-3 choice points per person: anyone with a
single point has a share of 1.0 and fails on day one regardless of data quality.
The pooled leave-one-out accuracy of a per-person constant asks the same
question -- is each individual a constant -- and is defined at every n.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "label_gate.json")


# --------------------------------------------------------------------------
# thresholds -- fixed here, before any distribution is looked at
# --------------------------------------------------------------------------

MAJORITY_MAX = 0.45         # Phase CP measured 0.656
ENTROPY_MIN_FRAC = 0.75     # of log(k), k = mean alternatives offered
LOO_CONSTANT_MAX = 0.60     # a per-person constant must not already be this good
DELTA = 0.05                # the band on both directional arms


# Each gate's DIRECTION as data, so a sign cannot be inverted in prose without a
# test noticing. `better` says which way is passing:
#   'lower'  -> the statistic must not EXCEED the reference
#   'higher' -> the statistic must not FALL BELOW the reference
GATES = {
    'majority_share':      {'better': 'lower',  'reference': MAJORITY_MAX,
                            'asks': 'is one label most of the record'},
    'entropy_frac':        {'better': 'higher', 'reference': ENTROPY_MIN_FRAC,
                            'asks': 'is there behavioural variety to model'},
    'loo_person_constant': {'better': 'lower',  'reference': LOO_CONSTANT_MAX,
                            'asks': 'is each individual already a constant'},
    'options_only':        {'better': 'lower',  'reference': 'majority + delta',
                            'asks': 'does the option phrasing carry the answer'},
    'context_neutral_sanity': {'better': 'higher', 'reference': 'majority - delta',
                            'asks': 'is a full-transcript read worse than a constant'},
    'context_neutral_headroom': {'better': 'lower', 'reference': 'ceiling - headroom',
                            'asks': 'is there room left for a motive to add anything'},
}


# --------------------------------------------------------------------------
# label statistics -- no model calls
# --------------------------------------------------------------------------

def majority_share(labels) -> float:
    """Share of the commonest label. The number that decided Phase CP."""
    labels = list(labels)
    if not labels:
        return 0.0
    return collections.Counter(labels).most_common(1)[0][1] / len(labels)


def normalised_entropy(labels, k: float = None) -> float:
    """Shannon entropy as a fraction of log(k).

    Normalised by the number of alternatives OFFERED rather than the number
    observed. Normalising by observed classes would let a corpus that never
    exercises an option score a perfect 1.0 for using two labels out of seven.
    """
    labels = list(labels)
    if not labels:
        return 0.0
    counts = collections.Counter(labels)
    n = len(labels)
    h = -sum((c / n) * math.log(c / n) for c in counts.values() if c)
    k = float(k if k else len(counts))
    if k <= 1:
        return 0.0
    return h / math.log(k)


def loo_constant_accuracy(by_person) -> float:
    """Pooled leave-one-out accuracy of a PER-PERSON constant.

    For each point, predict that person's commonest OTHER label. Pooled over
    points, not averaged over people, so somebody with 30 points does not weigh
    the same as somebody with one.

    Defined at every n: a person with a single point contributes one prediction
    made from an empty history, which is a miss. That is the honest reading --
    you cannot model a person from nothing -- where per-person majority share
    would have scored them 1.0 and failed the whole corpus.
    """
    hits = total = 0
    for _, labels in by_person.items():
        labels = list(labels)
        for i, truth in enumerate(labels):
            rest = collections.Counter(labels[:i] + labels[i + 1:])
            if not rest:
                total += 1
                continue
            pred = max(sorted(rest), key=lambda a: rest[a])
            hits += (pred == truth)
            total += 1
    return (hits / total) if total else 0.0


def label_stats(points) -> dict:
    """Everything the no-model half of the gate needs, for one choice type."""
    points = list(points)
    labels = [p.get("actual") for p in points if p.get("actual")]
    by_person = collections.defaultdict(list)
    for p in points:
        if p.get("actual"):
            by_person[(p.get("corpus"), p.get("person"))].append(p["actual"])
    ks = [p.get("n_alternatives") or len(p.get("alternatives") or []) for p in points]
    k = (sum(ks) / len(ks)) if ks else 0.0
    return {
        "points": len(points),
        "people": len(by_person),
        "mean_alternatives": round(k, 3),
        "label_counts": dict(collections.Counter(labels).most_common()),
        "majority_share": round(majority_share(labels), 4),
        "entropy_frac": round(normalised_entropy(labels, k), 4),
        "loo_person_constant": round(loo_constant_accuracy(by_person), 4),
    }


# --------------------------------------------------------------------------
# directional gates
# --------------------------------------------------------------------------

def check_options_only(options_only_acc, majority_acc, delta=DELTA) -> dict:
    """PASS when options_only does NOT beat the class prior.

    Direction matters and is the opposite of the next one. Above majority+delta
    the option wording is carrying the answer and no lift in the run means
    anything.
    """
    limit = majority_acc + delta
    return {"gate": "options_only", "value": options_only_acc, "limit": round(limit, 4),
            "better": "lower", "pass": options_only_acc <= limit,
            "verdict_on_fail": "VOID -- the option phrasing leaks the outcome"}


def check_context_neutral(ctx_acc, majority_acc, ceiling=1.0,
                          delta=DELTA, headroom=DELTA) -> dict:
    """TWO gates on one arm, pointing opposite ways.

    sanity   ctx >= majority - delta. A model with the whole transcript that is
             worse than a constant is not a bar. Phase CP: 0.533 vs 0.656.
    headroom ctx <= ceiling - headroom. At ceiling a motive has nothing to add.
    """
    floor = majority_acc - delta
    cap = ceiling - headroom
    return {
        "gate": "context_neutral",
        "value": ctx_acc,
        "sanity": {"floor": round(floor, 4), "better": "higher",
                   "pass": ctx_acc >= floor,
                   "verdict_on_fail": "the bar is below a constant; do not score against it"},
        "headroom": {"cap": round(cap, 4), "better": "lower",
                     "pass": ctx_acc <= cap,
                     "verdict_on_fail": "at ceiling; fall back to the silent subgroup"},
        "pass": ctx_acc >= floor and ctx_acc <= cap,
    }


def evaluate(stats, options_only_acc=None, context_neutral_acc=None,
             majority_acc=None, ceiling=1.0) -> dict:
    """Combine the no-model statistics with whichever arms have been measured."""
    checks = [
        {"gate": "majority_share", "value": stats["majority_share"],
         "limit": MAJORITY_MAX, "better": "lower",
         "pass": stats["majority_share"] <= MAJORITY_MAX},
        {"gate": "entropy_frac", "value": stats["entropy_frac"],
         "limit": ENTROPY_MIN_FRAC, "better": "higher",
         "pass": stats["entropy_frac"] >= ENTROPY_MIN_FRAC},
        {"gate": "loo_person_constant", "value": stats["loo_person_constant"],
         "limit": LOO_CONSTANT_MAX, "better": "lower",
         "pass": stats["loo_person_constant"] <= LOO_CONSTANT_MAX},
    ]
    if majority_acc is None:
        majority_acc = stats["majority_share"]
    if options_only_acc is not None:
        checks.append(check_options_only(options_only_acc, majority_acc))
    if context_neutral_acc is not None:
        checks.append(check_context_neutral(context_neutral_acc, majority_acc, ceiling))
    return {
        "stats": stats,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
        "arms_measured": [c["gate"] for c in checks],
        "arms_pending": [g for g in ("options_only", "context_neutral")
                         if g not in [c["gate"] for c in checks]],
    }


# --------------------------------------------------------------------------

def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=HERE, text=True).strip()
    except Exception:
        return None


def by_choice_type(points):
    """Group by the `kind` a converter stamped, or one bucket if it did not.

    A corpus can pass in aggregate while one kind inside it is a constant, and
    pooling is how that stays invisible.
    """
    groups = collections.defaultdict(list)
    for p in points:
        groups[p.get("kind") or "(unlabelled)"].append(p)
    return dict(groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--side", default="dev", choices=["dev", "test"],
                    help="dev only; reading test here would burn the one look")
    ap.add_argument("--by-choice-type", action="store_true")
    ap.add_argument("--options-only-acc", type=float, default=None,
                    help="measured accuracy of the options-only arm (100-point dev sample)")
    ap.add_argument("--context-neutral-acc", type=float, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.side != "dev":
        ap.error("the gate runs on dev. Standing rule 6: one look at test, later.")

    import choice_points as cp
    points = cp.load(corpus=a.corpus, side=a.side)
    if not points:
        print(f"no choice points for {a.corpus}/{a.side} -- nothing to gate", file=sys.stderr)
        return 2

    groups = by_choice_type(points) if a.by_choice_type else {"(all)": points}
    report = {"corpus": a.corpus, "side": a.side, "git_commit": git_head(),
              "thresholds": {"majority_share_max": MAJORITY_MAX,
                             "entropy_frac_min": ENTROPY_MIN_FRAC,
                             "loo_person_constant_max": LOO_CONSTANT_MAX,
                             "delta": DELTA},
              "gates": GATES, "by_choice_type": {}}

    ok = True
    for kind, pts in sorted(groups.items()):
        res = evaluate(label_stats(pts), a.options_only_acc, a.context_neutral_acc)
        report["by_choice_type"][kind] = res
        ok = ok and res["pass"]
        print(f"\n{a.corpus}/{a.side}  kind={kind}  n={res['stats']['points']}  "
              f"people={res['stats']['people']}")
        for c in res["checks"]:
            mark = "PASS" if c["pass"] else "FAIL"
            print(f"  [{mark}] {c['gate']:<26} {c['value']}  "
                  f"({'must be <= ' if c.get('better') == 'lower' else 'must be >= '}"
                  f"{c.get('limit', c.get('sanity', {}).get('floor'))})")
        if res["arms_pending"]:
            print(f"  ... not yet measured: {', '.join(res['arms_pending'])}")

    report["pass"] = ok
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {a.out}   overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
