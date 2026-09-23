"""Freeze the numbers every later phase is measured against.

    python baseline_snapshot.py                 # writes baseline_metrics.json
    python baseline_snapshot.py --check         # fail if the logs no longer agree

WHY FREEZE THEM. PRECHECKS quotes churn at 55.6 per 100, likelihood rank
movement at 1.79 places and weight rank movement at 1.03, each computed once in
a shell and never written down anywhere a later run could diff against. A claim
that an intervention moved churn is unreadable without the number it moved from
AND the run set that number came from, so both go in the file.

WHAT IS RECOMPUTED HERE RATHER THAN COPIED. Every figure below is derived from
the logs on disk by this script. Where it disagrees with PRECHECKS the
disagreement is reported in `vs_prechecks` rather than reconciled: PRECHECKS
computed some of these over a hand-picked run set that was never recorded, and
quietly matching it would be fitting the script to the number.

THE RUN SET IS PART OF THE METRIC. Runs differ in population size, alpha,
scorer mode and corpus, and rank movement is bounded by population size -- a
pool that mixes n=4 and n=8 runs reports a number that belongs to neither. So
every metric carries `runs` and `steps`, and the production-config subset
(alpha=0.85, n=8, rank scorer) is reported separately from the pooled figure.

IDENTITY FOR RANK MOVEMENT. Ranks are tracked per ROOT, because that is what
argmax churn is defined on and mixing the two keys would let a claim about one
be read as a claim about the other. A root's likelihood rank is the best rank
among its particles and its weight is their sum. Particle-level movement keyed
on lineage_id is reported beside it, so the two are never confused.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import math
import os
import statistics as st
import subprocess

import musing_layout as ml
from audit_ties import blocks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "baseline_metrics.json")
MIN_STEPS = 10  # below this a run's churn rate is one or two events of noise


# --------------------------------------------------------------------------
# per-run reduction over a steps.jsonl stream
# --------------------------------------------------------------------------

def roots_of(step):
    """Root -> (best likelihood rank, summed weight) for one step's slate.

    A root can hold several particles after a split or a resample duplicate.
    Summing weight and taking the best rank is the same reduction argmax churn
    makes implicitly when it reads the top particle's root_id.
    """
    best, mass = {}, collections.Counter()
    for p in step.get("particles") or []:
        rid = p.get("root_id")
        if rid is None:
            continue
        r = p.get("likelihood_rank")
        if r is not None and (rid not in best or r < best[rid]):
            best[rid] = r
        mass[rid] += float(p.get("weight") or 0.0)
    return best, mass


def rank_map(scores, reverse=False):
    """Dense 1-based ranks. reverse=True ranks the largest value first."""
    order = sorted(scores, key=lambda k: (-scores[k] if reverse else scores[k], str(k)))
    return {k: i + 1 for i, k in enumerate(order)}


def mean_abs_move(prev, cur):
    """Mean |rank change| over the keys present in BOTH steps.

    Keys that appear or vanish are dropped rather than scored against a
    sentinel: a hypothesis that was minted this step did not move, and giving it
    a movement would report population turnover as instability of the ordering.
    """
    shared = set(prev) & set(cur)
    if not shared:
        return None, 0
    return st.mean(abs(cur[k] - prev[k]) for k in shared), len(shared)


def shuffle_ceiling(n):
    """Mean |rank change| under a uniformly random permutation of n items."""
    return (n * n - 1) / (3.0 * n) if n > 1 else 0.0


def read_run(path):
    steps = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    steps = [s for s in steps if s.get("particles")]
    if len(steps) < MIN_STEPS:
        return None

    churn = lik_top_change = transitions = 0
    lik_moves, wt_moves, lin_moves = [], [], []
    pops, tie_blocks, tied_surprise = [], [], []
    surprise = resamples = 0
    op_events = []
    prev_root = prev_lik = prev_wt = prev_lin = None

    for s in steps:
        best, mass = roots_of(s)
        if not mass:
            continue
        pops.append(len(s["particles"]))
        w = [float(x) for x in (s.get("weights_pre") or []) if x is not None]
        if w:
            tie_blocks.append(blocks(w))
            if s.get("surprise"):
                nrep = max(1, len(w) // 4)
                pool = sorted(w)[:max(2 * nrep, nrep + 1)]
                tied_surprise.append(1.0 if max(pool) - min(pool) <= 0 else 0.0)
        if s.get("surprise"):
            surprise += 1
        ops = s.get("operators_fired") or []
        if "resample" in ops:
            resamples += 1

        top_root = max(mass, key=lambda k: (mass[k], str(k)))
        lik_rank = rank_map(best)                      # smaller rank is better
        wt_rank = rank_map(mass, reverse=True)
        lin = {p.get("lineage_id"): p.get("likelihood_rank")
               for p in s["particles"] if p.get("likelihood_rank") is not None}
        lik_top = min(best, key=lambda k: (best[k], str(k))) if best else None

        if prev_root is not None:
            transitions += 1
            changed = top_root != prev_root
            churn += changed
            op_events.append((tuple(sorted(ops)), changed))
            if lik_top is not None and prev_lik is not None:
                lik_top_change += (lik_top != prev_lik)
            m, _ = mean_abs_move(prev_wt, wt_rank)
            if m is not None:
                wt_moves.append(m)
            m, _ = mean_abs_move(prev_rank_lik, lik_rank)
            if m is not None:
                lik_moves.append(m)
            m, _ = mean_abs_move(prev_lin, lin)
            if m is not None:
                lin_moves.append(m)

        prev_root, prev_lik = top_root, lik_top
        prev_rank_lik, prev_wt, prev_lin = lik_rank, wt_rank, lin

    if not transitions:
        return None
    pop = st.median(pops)
    return {
        "run": os.path.basename(path).replace(".steps.jsonl", ""),
        "steps": len(steps),
        "transitions": transitions,
        "population_median": pop,
        "churn": churn,
        "churn_per_100": 100.0 * churn / transitions,
        "likelihood_rank_move": st.mean(lik_moves) if lik_moves else None,
        "weight_rank_move": st.mean(wt_moves) if wt_moves else None,
        "lineage_rank_move": st.mean(lin_moves) if lin_moves else None,
        "shuffle_ceiling": shuffle_ceiling(pop),
        "likelihood_top_changes_root": lik_top_change,
        "likelihood_top_change_rate": lik_top_change / transitions,
        "surprise_steps": surprise,
        "resamples": resamples,
        "tie_block_median": st.median(tie_blocks) if tie_blocks else None,
        "tie_block_first_half": st.median(tie_blocks[:len(tie_blocks) // 2]) if len(tie_blocks) > 3 else None,
        "tie_block_second_half": st.median(tie_blocks[len(tie_blocks) // 2:]) if len(tie_blocks) > 3 else None,
        "weak_half_tied_on_surprise": st.mean(tied_surprise) if tied_surprise else None,
        "_op_events": op_events,
        "_tie_blocks": tie_blocks,
        "_tied_surprise": tied_surprise,
    }


# --------------------------------------------------------------------------
# run metadata, so a pool can be narrowed to one configuration
# --------------------------------------------------------------------------

def run_meta():
    path = ml.runs_jsonl()
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("run_id"):
                meta[d["run_id"]] = d       # later rows win: a rerun replaces its own record
    return meta


def is_production(meta):
    """The configuration every logged finding in PRECHECKS was measured under."""
    return (meta.get("alpha") == 0.85
            and meta.get("n_hypotheses") == 8
            and meta.get("scorer_mode") == "rank"
            and meta.get("baseline_scorer") is False)


# --------------------------------------------------------------------------
# pooled statistics
# --------------------------------------------------------------------------

def summarise(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return None
    return {
        "median": round(st.median(vals), 4),
        "mean": round(st.mean(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "runs": len(vals),
    }


def pooled_ties(rows):
    """Tie statistics over STEPS, which is how audit_ties defines them.

    Everything else here is a median over runs, because a run is the unit an
    intervention is applied to. The tie figures are the exception: PRECHECKS
    quotes them pooled over steps, and reporting a run-median under the same
    name would invite a comparison between two different statistics.

    `surprise` is the catch. No run in the production pool logs it -- the flag
    arrived after those runs were written -- so the tie-on-surprise figure can
    only be computed on the non-production runs that do, and it is reported with
    that coverage attached rather than as a blank.
    """
    blk = [x for r in rows for x in r["_tie_blocks"]]
    tied = [x for r in rows for x in r["_tied_surprise"]]
    with_surprise = [r for r in rows if r["surprise_steps"]]
    return {
        "tie_block_median_over_steps": round(st.median(blk), 4) if blk else None,
        "steps_measured": len(blk),
        "weak_half_tied_on_surprise": round(st.mean(tied), 4) if tied else None,
        "surprise_steps_measured": len(tied),
        "runs_logging_surprise": len(with_surprise),
        "runs_in_pool": len(rows),
        "coverage_note": ("`surprise` is absent from every run that predates the flag; "
                          "a tie-on-surprise figure covering 0 runs is reported as null, "
                          "not as 0."),
    }


def operator_table(rows):
    """P(churn) by operator, pooled over every transition, against the lift.

    Attribution is per-operator and not per-combination: the combinations are
    sparse and a table over them is mostly cells of one. A step firing two
    operators counts toward both, which is why the shares do not sum to 1 --
    this bounds each operator's involvement, it does not partition the churn.
    """
    total = collections.Counter()
    fired = collections.Counter()
    churn_by = collections.Counter()
    none_steps = none_churn = 0
    for r in rows:
        for ops, changed in r["_op_events"]:
            total["all"] += 1
            churn_by["all"] += changed
            if not ops:
                none_steps += 1
                none_churn += changed
            for op in set(ops):
                fired[op] += 1
                churn_by[op] += changed
    base = churn_by["all"] / total["all"] if total["all"] else 0.0
    out = {"_baseline_p_churn": round(base, 4), "_transitions": total["all"],
           "(none)": {"steps": none_steps,
                      "p_churn": round(none_churn / none_steps, 4) if none_steps else None,
                      "lift": round(none_churn / none_steps - base, 4) if none_steps else None}}
    for op, n in fired.most_common():
        out[op] = {"steps": n, "p_churn": round(churn_by[op] / n, 4),
                   "lift": round(churn_by[op] / n - base, 4)}
    return out


def vacuity_screen(limit=None):
    """The offline vacuity screen, pooled over the tracer dumps that support it.

    Imported lazily: it pulls in tracer.py, which is 4k lines and not needed by
    anything else here, and a snapshot of the cheap metrics should not fail
    because a heavy import did.
    """
    try:
        import audit_vacuity as av
        from replay_likelihood import find_traces  # noqa: F401
    except Exception as exc:
        return {"error": f"screen unavailable: {exc}"}

    paths = sorted(glob.glob(os.path.join(ml.traces_dir(), "*.jsonl")))
    if limit:
        paths = paths[:limit]
    suspects, screened, runs_with_recs = [], 0, 0
    for p in paths:
        try:
            recs = av.read_run(p)
        except Exception:
            continue
        if not recs:
            continue
        runs_with_recs += 1
        rows = av.screen(recs)
        screened += len(rows)
        for row in rows:
            if row.get("suspect"):
                suspects.append({"run": os.path.basename(p), "anchor": row["anchor"],
                                 "life": row["life"], "mass": round(row["mass"], 4),
                                 "rank": round(row["rank"], 3),
                                 "trigger": round(row["trigger"], 4)})
    suspects.sort(key=lambda s: -s["mass"])
    return {
        "traces_scanned": len(paths),
        "traces_with_records": runs_with_recs,
        "commitments_screened": screened,
        "suspects": len(suspects),
        "suspect_rate": round(suspects and len(suspects) / screened or 0.0, 4),
        "top": suspects[:15],
        "definition": ("heavy (mean weight > 0.14), long-lived (>= 8 steps), top-ranked "
                       "(mean rank <= n/2) and a split candidate on <= 10% of its steps"),
    }


# --------------------------------------------------------------------------

def build(vacuity_limit=None):
    meta = run_meta()
    rows = []
    for path in sorted(glob.glob(os.path.join(ml.OUT_DIR, "**", "*.steps.jsonl"),
                                 recursive=True)):
        r = read_run(path)
        if r:
            r["meta_found"] = r["run"] in meta
            r["production_config"] = bool(meta.get(r["run"]) and is_production(meta[r["run"]]))
            r["corpus"] = (meta.get(r["run"]) or {}).get("corpus") or ml.corpus_of(r["run"])
            rows.append(r)

    prod = [r for r in rows if r["production_config"]]
    pools = {"all": rows, "production_config": prod}

    metrics = {}
    for name, pool in pools.items():
        if not pool:
            continue
        metrics[name] = {
            "runs": len(pool),
            "steps": sum(r["steps"] for r in pool),
            "transitions": sum(r["transitions"] for r in pool),
            "runs_at_zero_churn": sum(1 for r in pool if r["churn"] == 0),
            "population_median": st.median([r["population_median"] for r in pool]),
            "argmax_churn_per_100": summarise(pool, "churn_per_100"),
            "likelihood_rank_move": summarise(pool, "likelihood_rank_move"),
            "weight_rank_move": summarise(pool, "weight_rank_move"),
            "lineage_rank_move": summarise(pool, "lineage_rank_move"),
            "shuffle_ceiling": summarise(pool, "shuffle_ceiling"),
            "likelihood_top_change_rate": summarise(pool, "likelihood_top_change_rate"),
            "tie_block_median": summarise(pool, "tie_block_median"),
            "weak_half_tied_on_surprise": summarise(pool, "weak_half_tied_on_surprise"),
            "tie_block_grows": {
                "runs_compared": sum(1 for r in pool if r["tie_block_first_half"] is not None),
                "runs_growing": sum(1 for r in pool
                                    if r["tie_block_first_half"] is not None
                                    and r["tie_block_second_half"] > r["tie_block_first_half"]),
                "first_half_median": round(st.median([r["tie_block_first_half"] for r in pool
                                                      if r["tie_block_first_half"] is not None]), 4),
                "second_half_median": round(st.median([r["tie_block_second_half"] for r in pool
                                                       if r["tie_block_second_half"] is not None]), 4),
            },
            "pooled_ties": pooled_ties(pool),
            "operators": operator_table(pool),
            "by_corpus": {c: round(st.median([r["churn_per_100"] for r in pool if r["corpus"] == c]), 2)
                          for c in sorted({r["corpus"] for r in pool if r["corpus"]})},
        }

    ref = metrics.get("production_config") or metrics.get("all") or {}
    quoted = {"argmax_churn_per_100": 55.6, "likelihood_rank_move": 1.79,
              "weight_rank_move": 1.03, "likelihood_top_change_rate": 0.72,
              "tie_block_median": 0.17, "weak_half_tied_on_surprise": 0.55}
    vs = {}
    ref_name = "production_config" if "production_config" in metrics else "all"
    POOLED = {"tie_block_median": "tie_block_median_over_steps",
              "weak_half_tied_on_surprise": "weak_half_tied_on_surprise"}

    def read(pool_name, key):
        pool = metrics.get(pool_name) or {}
        if key in POOLED:
            return (pool.get("pooled_ties") or {}).get(POOLED[key])
        return (pool.get(key) or {}).get("median")

    for k, q in quoted.items():
        # The production pool is the like-for-like comparison, but no run in it
        # logs `surprise`, so that one figure falls back to the pool that does
        # and says so rather than reporting a null next to a quoted number.
        source = ref_name
        got = read(ref_name, k)
        if got is None and ref_name != "all":
            got, source = read("all", k), "all (production pool does not log it)"
        vs[k] = {"prechecks": q, "here": got, "source": source,
                 "delta": round(got - q, 4) if got is not None else None}

    return {
        "_README": (
            "Baseline for every later phase. Recomputed from the logs by "
            "baseline_snapshot.py; `python baseline_snapshot.py --check` fails if the "
            "logs no longer produce these numbers. Read `pools` before quoting any "
            "figure -- rank movement is bounded by population size and the pooled "
            "number belongs to whatever mix of runs the pool holds."),
        "frozen_at": dt.date.today().isoformat(),
        "git_commit": git_head(),
        "generator": "baseline_snapshot.py",
        "min_steps_per_run": MIN_STEPS,
        "pools": {
            "all": "every steps.jsonl with >= 10 scored steps, any configuration",
            "production_config": "alpha=0.85, n=8, rank scorer, no baseline scorer -- "
                                 "the configuration PRECHECKS was measured under",
        },
        "metrics": metrics,
        "vs_prechecks": vs,
        "vs_prechecks_note": (
            "Differences are reported, not reconciled. PRECHECKS computed these over a "
            "run set it did not record, and tuning this script until the numbers match "
            "would be fitting the measurement to the claim. What matters downstream is "
            "that a later phase compares against THIS file, recomputed the same way."),
        "vacuity_screen": vacuity_screen(vacuity_limit),
        "per_run": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="recompute and exit non-zero if the frozen file disagrees")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="--check: allowed drift on a pooled median")
    ap.add_argument("--vacuity-limit", type=int, default=None,
                    help="scan only the first N tracer dumps (the screen is the slow part)")
    a = ap.parse_args()

    payload = build(a.vacuity_limit)

    if a.check:
        frozen = json.load(open(a.out, encoding="utf-8"))
        drift = []
        for pool, m in payload["metrics"].items():
            old = (frozen.get("metrics") or {}).get(pool) or {}
            for key, val in m.items():
                if not isinstance(val, dict) or "median" not in val:
                    continue
                was = (old.get(key) or {}).get("median")
                if was is None or abs(was - val["median"]) > a.tolerance:
                    drift.append(f"{pool}.{key}: {was} -> {val['median']}")
        if drift:
            raise SystemExit("BASELINE DRIFT:\n  " + "\n  ".join(drift))
        print(f"baseline matches {a.out}")
        return

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}")
    for pool, m in payload["metrics"].items():
        print(f"\n[{pool}] {m['runs']} runs, {m['transitions']} transitions, "
              f"population median {m['population_median']:.0f}")
        for key in ("argmax_churn_per_100", "likelihood_rank_move", "weight_rank_move",
                    "shuffle_ceiling", "likelihood_top_change_rate",
                    "tie_block_median", "weak_half_tied_on_surprise"):
            s = m.get(key)
            if s:
                print(f"  {key:<28} median {s['median']:>8.3f}  mean {s['mean']:>8.3f}"
                      f"  range {s['min']:.3f}-{s['max']:.3f}  (n={s['runs']})")
        print(f"  runs at zero churn: {m['runs_at_zero_churn']}")
        pt = m["pooled_ties"]
        print(f"  pooled over steps: tie block {pt['tie_block_median_over_steps']} "
              f"({pt['steps_measured']} steps); weak half tied on surprise "
              f"{pt['weak_half_tied_on_surprise']} "
              f"({pt['runs_logging_surprise']}/{pt['runs_in_pool']} runs log surprise)")
    print("\nvs PRECHECKS (reported, not reconciled):")
    for k, v in payload["vs_prechecks"].items():
        print(f"  {k:<28} quoted {v['prechecks']:<8} here {v['here']}"
              f"   [{v['source']}]")
    vs = payload["vacuity_screen"]
    if "error" not in vs:
        print(f"\nvacuity screen: {vs['suspects']} suspects of {vs['commitments_screened']} "
              f"commitments over {vs['traces_with_records']}/{vs['traces_scanned']} traces")


if __name__ == "__main__":
    main()
