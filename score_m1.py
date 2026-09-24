"""Score Milestone 1. The bar is frozen on dev before test is read.

Everything registered in PREREG is applied here and nothing is chosen after
looking: THETA is fitted per arm on dev; the bar is whichever of obvious/majority
scores better on dev; the bands on `placebo` and `swap` are Phase CP's verbatim;
the permutation test clusters on the person; and the silent subgroup -- defined
before any forecast existed -- is reported separately because it is the only part
of CaSiNo where the motive is not spoken aloud.
"""
import collections, json, math, random, statistics as st, sys
import forecast_likelihood as F

BANDS = "swap >= real -> CONTROL MOVED WITH IT | 0.5*real <= swap < real -> WEAK | swap < 0.5*real -> EFFECT"


def ranks_of(mean_rank, alts):
    """Consensus order from mean rank POSITION across renderings (Borda).

    Not from an averaged distribution: averaging distributions needs a theta
    chosen before the dev fit, and at theta=1 it is uniform, which would make
    every consensus order alphabetical noise."""
    return [alts.index(a) for a in sorted(alts, key=lambda x: (mean_rank.get(x, 99), x))]


def recs_for(rows, arm):
    out = []
    for r in rows:
        d = r.get(f"dist_{arm}")
        if not d or r["actual"] not in r["alts"]:
            continue
        out.append((ranks_of(d, r["alts"]), r["alts"].index(r["actual"]), len(r["alts"]),
                    r["person_uid"], r["cp_id"]))
    return out


def ll(recs, theta):
    return [math.log(max(F.rank_to_distribution(rk, n, theta).get(a, 0.0), 1e-12))
            for rk, a, n, _, _ in recs]


def fit(recs):
    best, bll = 1.0, -1e18
    for th in [round(0.05 * i, 2) for i in range(1, 20)]:
        s = sum(ll(recs, th))
        if s > bll:
            best, bll = th, s
    return best


def majority_ll(dev_rows, rows):
    c = collections.Counter(r["actual"] for r in dev_rows)
    out = []
    for r in rows:
        alts = r["alts"]
        tot = sum(c.get(a, 0) for a in alts) + len(alts)
        out.append(math.log((c.get(r["actual"], 0) + 1) / tot))    # Laplace
    return out


def clustered_perm(a, b, clusters, iters=20000, seed=0):
    """Paired permutation on the per-point difference, clustered on the person.

    Signs are flipped PER CLUSTER, not per point: a person's points are not
    independent, and flipping them independently would understate the p.
    """
    d = [x - y for x, y in zip(a, b)]
    by = collections.defaultdict(list)
    for v, c in zip(d, clusters):
        by[c].append(v)
    keys = list(by)
    obs = st.mean(d)
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        tot, n = 0.0, 0
        for k in keys:
            s = 1 if rng.random() < 0.5 else -1
            for v in by[k]:
                tot += s * v; n += 1
        if tot / n >= obs:
            hits += 1
    return (hits + 1) / (iters + 1)


def report(rows, dev_rows, label, thetas, bar_name, out):
    arms = ["context_neutral", "obvious", "hypothesis", "placebo", "swap"]
    base = {}
    for arm in arms:
        recs = recs_for(rows, arm)
        base[arm] = {"recs": recs, "ll": ll(recs, thetas[arm])}
    maj = majority_ll(dev_rows, rows)
    barvals = maj if bar_name == "majority" else base["obvious"]["ll"]

    print(f"\n=== {label}   n={len(rows)} points, {len({r['person_uid'] for r in rows})} people")
    print(f"    bar = {bar_name} (frozen on dev)")
    print(f"    {'arm':<18}{'theta':>7}{'logscore':>11}{'lift/bar':>11}{'top-1':>8}")
    print(f"    {'majority':<18}{'-':>7}{st.mean(maj):>11.4f}"
          f"{st.mean(maj) - st.mean(barvals):>+11.4f}{'-':>8}")
    lifts = {}
    for arm in arms:
        recs, l = base[arm]["recs"], base[arm]["ll"]
        if not recs:
            continue
        # align the bar to this arm's points
        ids = {r[4] for r in recs}
        bar_aligned = [b for b, r in zip(barvals, rows) if r["cp_id"] in ids]
        lift = st.mean(l) - st.mean(bar_aligned)
        lifts[arm] = lift
        top1 = st.mean([1.0 if rk[0] == a else 0.0 for rk, a, _, _, _ in recs])
        print(f"    {arm:<18}{thetas[arm]:>7.2f}{st.mean(l):>11.4f}{lift:>+11.4f}{top1:>8.3f}")

    real, plac, swp = lifts.get("hypothesis"), lifts.get("placebo"), lifts.get("swap")
    print(f"\n    BANDS ({BANDS})")
    # THE BANDS ARE UNDEFINED WHEN THE REAL LIFT IS NOT POSITIVE, and applying
    # them anyway inverts them. With real = -0.0988, half of it is -0.0494, so a
    # swap at -0.1612 satisfies `swap < 0.5*real` and prints EFFECT -- for an arm
    # that LOST to the bar. Phase CP's bands assume a positive lift; they are a
    # way of asking whether a real effect is person-specific, not a way of
    # ranking two failures.
    if real is None or real <= 0:
        print(f"      hypothesis lift {real:+.4f} is not positive -- THE BANDS DO NOT APPLY.")
        print(f"      There is no effect to attribute, so there is nothing for a control")
        print(f"      to be compared against. placebo {plac:+.4f}, swap {swp:+.4f} reported raw.")
    else:
        for name, v in (("placebo", plac), ("swap", swp)):
            if v is None:
                continue
            if v >= real:
                verdict = "CONTROL MOVED WITH IT -- not a finding"
            elif v >= 0.5 * real:
                verdict = "WEAK -- report as weak, never as an effect"
            else:
                verdict = "EFFECT"
            print(f"      hypothesis {real:+.4f} vs {name} {v:+.4f}  ->  {verdict}")

    hrecs = base["hypothesis"]["recs"]
    ids = {r[4] for r in hrecs}
    bar_aligned = [b for b, r in zip(barvals, rows) if r["cp_id"] in ids]
    p = clustered_perm(base["hypothesis"]["ll"], bar_aligned, [r[3] for r in hrecs])
    print(f"\n    clustered permutation p (hypothesis vs bar) = {p:.4f}")
    out[label] = {"n": len(rows), "people": len({r["person_uid"] for r in rows}),
                  "bar": bar_name, "lifts": lifts, "p_clustered": p,
                  "theta": thetas}


def main():
    d = json.load(open("m1_casino.json", encoding="utf-8"))
    rows = d["records"]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    # THETA per arm, fitted on DEV (standing rule 9)
    thetas = {arm: fit(recs_for(dev, arm)) for arm in
              ["context_neutral", "obvious", "hypothesis", "placebo", "swap"]}
    # THE BAR, frozen on dev: whichever of obvious/majority scores better there
    obv = st.mean(ll(recs_for(dev, "obvious"), thetas["obvious"]))
    maj = st.mean(majority_ll(dev, dev))
    bar_name = "obvious" if obv >= maj else "majority"
    print(f"BAR FROZEN ON DEV: obvious {obv:.4f} vs majority {maj:.4f} -> {bar_name}")
    print(f"THETA per arm (dev): {thetas}")
    tv = [r["perm_rank_move"] for r in rows if r.get("perm_rank_move") is not None]
    if tv:
        print(f"order sensitivity at k=8, {d['orders']} renderings: mean rank displacement "
              f"median {st.median(tv):.2f} places, mean {st.mean(tv):.2f}")

    out = {}
    report(test, dev, "TEST (M1 partition)", thetas, bar_name, out)
    silent = [r for r in test if r["silent"]]
    if silent:
        report(silent, dev, "TEST, SILENT SUBGROUP", thetas, bar_name, out)
    out["_meta"] = {"model": d["model"], "orders": d["orders"], "calls": d["calls"],
                    "bar": bar_name, "perm_rank_move_median": st.median(tv) if tv else None}
    json.dump(out, open("m1_results.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote m1_results.json")


if __name__ == "__main__":
    main()
