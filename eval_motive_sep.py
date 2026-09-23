"""Can the filter tell two groups' motives apart, REPEATEDLY?

This is the harness the STANDARD prompt is tuned against. It exists because a
single run is not evidence: measured on two runs of identical input, Wolf and
Lyubovsky SWAPPED which settlement mode dominated (person-confirms 56%->37%
against 16%->64%), while the chi-square stayed significant in both. A metric
that reads one run therefore rewards noise. So every number here is computed
per seed and the score is the part that survives across seeds.

Two cases, deliberately different in kind:
  color  bloomfield-0002..0039   Wolf (management) vs Lyubovsky/Rovani (eng)
  size   bloomfield-0329..0334   Field vs Engineers, as merged coalitions

Score = MARGIN OVER CONTROL x DIRECTION CONSISTENCY.

The margin, not the raw separation, is the whole metric. Measured on v1 seed 0:
Wolf vs the two engineers separates 0.51, which looks like a result -- and the
two engineers separate from EACH OTHER by 0.48. Raw separation is therefore
almost entirely a measure of how much any two traces of this corpus differ, and
a variant tuned on it would be tuned on trace-to-trace noise.
  separation   per seed, total-variation distance between the two groups'
               distributions over settlement modes. 0 = identical, 1 = disjoint.
  consistency  fraction of seeds whose largest-difference mode points the same
               way as the majority. A detector that separates strongly but
               inconsistently is measuring the sampling, not the people.

Usage:
  eval_motive_sep.py --case color --seeds 3          # run + score
  eval_motive_sep.py --case color --score-only       # score what is on disk
"""
import argparse
import collections
import glob
import json
import random
import re
import subprocess

# What would SETTLE it -- written against the register the STANDARD field
# produces (sentences about evidence), NOT against transcript vocabulary. That
# distinction matters: a lexicon fitted on who-says-what in the transcript
# scored 39% of standards at zero and matched generic domain nouns (estimate,
# rows, plant), which made the deliverable/measurable axis unusable here.
MODES = [
    ("instrument", re.compile(
        r"\bRGB\b|spectromet|statistic|confidence interval|percentage of|objective measure|"
        r"sensor|reconcil|correlation|significant difference|\d+\s*%|\bmetric|threshold|"
        r"algorithm|recount|measurement|calibrat|ground truth", re.I)),
    ("outside party", re.compile(
        r"\bcustomer|\bclient|grower|botanist|third-party|independent (expert|research|audit)|"
        r"stakeholder|external|packer|buyer", re.I)),
    # Widened for RECALL after 35% of one group's standards and 22% of the
    # other's fell in no bin -- differential loss at that rate manufactures
    # separation on its own. The additions are all plainly person-settled
    # ("an expert's definitive label", "an observation by a trusted source",
    # "Lyubovsky applies the standard the same way"), not borderline calls.
    ("a person confirms", re.compile(
        r"\b(confirms?|states?|describes?|acknowledges?|agrees?|approves?|demonstrates?|"
        r"provides?|accepts?|concurs?|corroborat|endorses?|validates?|clarifies|explains?|"
        r"applies|interpret|weigh in|consensus|team member|colleague|trusted source|"
        r"an expert|expert(?:'s)? (opinion|label|explanation|judgement|judgment)|"
        r"someone|others?)\b", re.I)),
    ("a document exists", re.compile(
        r"\bdocument|written|guideline|protocol|\bspec\b|definition[s]? (are|is) established|"
        r"\breport\b|standard for|spreadsheet|policy", re.I)),
]
MODE_NAMES = [n for n, _ in MODES]

# NEGATIVE CONTROLS. A prompt can raise the score by manufacturing difference
# -- "unlike the others, {t} wants ..." separates any two people you point it
# at. The consistency term does not catch that: a variant can be consistently
# wrong. So every variant is also scored on a pair that should NOT separate,
# and a variant is only an improvement if it raises the real score WITHOUT
# raising the control. This is the same discipline that killed the three
# earlier culture detectors, which all fired on noise for lack of a control.
CONTROLS = {
    "color": {"a": ["Lyubovsky"], "b": ["Rovani"]},   # two engineers, same side
    # The size case needs its own same-side control or it repeats the mistake
    # the color case exposed: management vs engineering scored 0.33 while two
    # ENGINEERS scored 0.42, so without this the number means nothing. Field is
    # split in half by person -- FieldA Rodriguez+Cisneros (26 turns), FieldB
    # Letelier+Littell (48) -- and both halves are traced as coalitions exactly
    # the way the real arms are, so the control is built the same way as the
    # thing it controls.
    "size": {"a": ["FieldA"], "b": ["FieldB"]},
}

# Extra coalitions traced only to serve as controls; launched alongside the
# real arms so they share seeds, prompt variant and span.
CONTROL_MERGE = {
    "size": "FieldA=Rodriguez,Cisneros;FieldB=Letelier,Littell",
}

CASES = {
    "color": {
        "sets": "bloomfield-0002,bloomfield-0006,bloomfield-0009,bloomfield-0015,"
                "bloomfield-0031,bloomfield-0039",
        "merge": None,
        "groups": {"management": ["Wolf"], "engineering": ["Lyubovsky", "Rovani"]},
    },
    "size": {
        "sets": "bloomfield-0329,bloomfield-0330,bloomfield-0331,bloomfield-0332,"
                "bloomfield-0333,bloomfield-0334",
        "merge": "Field=Rodriguez,Letelier,Cisneros,Littell;"
                 "Engineers=Lyubovsky,Rovani,Deskins,McLafferty",
        "groups": {"field": ["Field"], "engineering": ["Engineers"]},
    },
}


# v4 asks the model to name the source in one reserved word. When it does, read
# that word rather than re-deriving it from the prose -- the regexes below are
# themselves a source of run-to-run noise (39% of one run's standards fell in no
# bin), and a variant that removes the guessing step should be scored without it.
EXPLICIT = {"COLLEAGUE": "a person confirms", "CUSTOMER": "outside party",
            "MEASUREMENT": "instrument", "DOCUMENT": "a document exists"}


def classify(s):
    s = s or ""
    head = re.match(r"\s*\**\s*([A-Z]{6,12})\b", s)
    if head and head.group(1) in EXPLICIT:
        return EXPLICIT[head.group(1)]
    for n, rx in MODES:
        if rx.search(s):
            return n
    return None


def standards(run_id, min_steps=8, settle_s=90):
    """Every DISTINCT standard a run produced, or None if the run is not usable.

    A run still being written scores differently every time it is read --
    measured: the same command gave separation 0.51 then 0.48, and the control
    0.48 then 0.31, purely because one trace gained steps in between. So a file
    touched in the last `settle_s` seconds, or shorter than `min_steps`, is
    treated as absent rather than as a short run.
    Distinct, not per-step: a commitment that survives 20 steps would otherwise
    count 20 times and one long-lived hypothesis would decide the result.
    """
    import os
    import time
    fs = glob.glob(f"musing_out/runs/*/{run_id}-*.steps.jsonl")
    if not fs:
        return None
    if time.time() - os.path.getmtime(fs[0]) < settle_s:
        return None
    if sum(1 for _ in open(fs[0], encoding="utf-8")) < min_steps:
        return None
    seen = []
    known = set()
    for line in open(fs[0], encoding="utf-8"):
        for p in json.loads(line).get("particles", []):
            s = p.get("standard")
            if s and s not in known:
                known.add(s)
                seen.append(s)
    return seen


def dist(items):
    c = collections.Counter(x for x in (classify(s) for s in items) if x)
    n = sum(c.values())
    return ({m: c[m] / n for m in MODE_NAMES}, n) if n else (None, 0)


def boot_tv(a, b, iters=4000, seed=11):
    """TV between two POOLED groups, with a bootstrap interval.

    Pooling every seed into one distribution and resampling it is strictly
    better use of the same runs than scoring each seed and averaging: at ~18
    standards per run the per-seed TV is dominated by multinomial noise
    (median 0.20-0.25 between two draws from an IDENTICAL distribution, 95th
    percentile 0.40), which is the same order as the effect being measured.
    Pooling three seeds takes n to ~55 and the noise floor to ~0.15, and the
    interval says outright how much of what is left is real.
    """
    rnd = random.Random(seed)

    def d(xs):
        c = collections.Counter(x for x in (classify(s) for s in xs) if x)
        n = sum(c.values())
        return [c[m] / n for m in MODE_NAMES] if n else None

    da, db = d(a), d(b)
    if not da or not db:
        return None
    obs = 0.5 * sum(abs(x - y) for x, y in zip(da, db))
    lo = []
    for _ in range(iters):
        ra = [a[rnd.randrange(len(a))] for _ in a]
        rb = [b[rnd.randrange(len(b))] for _ in b]
        xa, xb = d(ra), d(rb)
        if xa and xb:
            lo.append(0.5 * sum(abs(x - y) for x, y in zip(xa, xb)))
    lo.sort()
    # n is the CLASSIFIED count, not the raw pooled count: the distribution is
    # estimated from what fell in a bin, and reporting the raw total overstates
    # the sample the interval is actually built on.
    nc = (sum(1 for x in a if classify(x)), sum(1 for x in b if classify(x)))
    return {"tv": obs, "n": nc,
            "lo": lo[int(0.025 * len(lo))], "hi": lo[int(0.975 * len(lo))],
            "da": dict(zip(MODE_NAMES, da)), "db": dict(zip(MODE_NAMES, db))}


def pooled(case, seeds, variant, members):
    out = []
    for sd in seeds:
        for m in members:
            st = standards(rid_for(case, variant, sd, m))
            if st:
                out += st
    return out


def report_pooled(case, seeds, variant):
    cfg = CASES[case]
    ga, gb = list(cfg["groups"])
    A = pooled(case, seeds, variant, cfg["groups"][ga])
    B = pooled(case, seeds, variant, cfg["groups"][gb])
    real = boot_tv(A, B)
    if not real:
        print("  not enough classifiable standards")
        return
    print(f"  POOLED over {len(seeds)} seeds")
    print(f"    {ga:<12} " + "  ".join(f"{k.split()[0][:5]}={real['da'][k]:.2f}" for k in MODE_NAMES)
          + f"   n={real['n'][0]}")
    print(f"    {gb:<12} " + "  ".join(f"{k.split()[0][:5]}={real['db'][k]:.2f}" for k in MODE_NAMES)
          + f"   n={real['n'][1]}")
    print(f"    real split      TV {real['tv']:.2f}  95% CI [{real['lo']:.2f}, {real['hi']:.2f}]")
    ctl = CONTROLS.get(case)
    if ctl:
        c = boot_tv(pooled(case, seeds, variant, ctl["a"]),
                    pooled(case, seeds, variant, ctl["b"]))
        if c:
            print(f"    same-side ctrl  TV {c['tv']:.2f}  95% CI [{c['lo']:.2f}, {c['hi']:.2f}]"
                  + ("   <-- overlaps the real split" if c["hi"] >= real["lo"] else "   CLEARS"))
    # same person, seeds split in half
    half = len(seeds) // 2
    if half:
        selfs = []
        for members in cfg["groups"].values():
            for m in members:
                x = pooled(case, seeds[:half], variant, [m])
                y = pooled(case, seeds[half:], variant, [m])
                r = boot_tv(x, y) if x and y else None
                if r:
                    selfs.append(r["tv"])
        if selfs:
            print(f"    same person     TV {sum(selfs)/len(selfs):.2f}  "
                  f"(noise floor, {len(selfs)} people)")


def score_case(case, seeds, variant="v1", quiet=False):
    cfg = CASES[case]
    per_seed = []
    for sd in seeds:
        gd = {}
        for g, members in cfg["groups"].items():
            pooled = []
            for m in members:
                rid = rid_for(case, variant, sd, m)
                st = standards(rid)
                if st is None:
                    return None, f"missing run {rid}"
                pooled += st
            d, n = dist(pooled)
            if d is None:
                return None, f"no classifiable standards for {g} seed {sd}"
            gd[g] = (d, n)
        a, b = list(cfg["groups"])
        # total variation: half the L1 distance between the two distributions
        tv = 0.5 * sum(abs(gd[a][0][m] - gd[b][0][m]) for m in MODE_NAMES)
        lead = max(MODE_NAMES, key=lambda m: abs(gd[a][0][m] - gd[b][0][m]))
        sign = 1 if gd[a][0][lead] > gd[b][0][lead] else -1
        per_seed.append({"seed": sd, "tv": tv, "lead": lead, "sign": sign,
                         "n": {g: gd[g][1] for g in gd}, "d": {g: gd[g][0] for g in gd}})
        if not quiet:
            print(f"    seed {sd}: separation {tv:.2f}   biggest gap on "
                  f"'{lead}' ({'+' if sign > 0 else '-'}{a})   "
                  f"n={gd[a][1]}/{gd[b][1]}")
    if not per_seed:
        return None, "no seeds"
    # consistency is over (mode, direction) jointly -- the same gap pointing the
    # same way. Agreeing on the mode while disagreeing on who leads is not
    # agreement about anything.
    keys = [(p["lead"], p["sign"]) for p in per_seed]
    top = collections.Counter(keys).most_common(1)[0]
    # With one seed consistency is 1.00 by construction and means nothing, so
    # it is reported as None rather than as a perfect score.
    cons = top[1] / len(per_seed) if len(per_seed) > 1 else None
    mtv = sum(p["tv"] for p in per_seed) / len(per_seed)
    return {"case": case, "seeds": len(per_seed), "mean_tv": mtv,
            "consistency": cons, "modal": top[0], "per_seed": per_seed}, None


TAG = ""   # set by --tag; distinguishes ARMS that share a case/variant/seed


def rid_for(case, variant, seed, member):
    """Run id for one arm member.

    TAG defaults to "" so every id is byte-identical to what this file has
    always produced -- the runs already on disk stay findable and comparable.
    A non-empty tag is what keeps a profiled arm from overwriting the
    unprofiled run it is supposed to be measured against.
    """
    return f"{case[:2]}{variant}{TAG}s{seed}_{member}"


def launch(case, seed, variant="v1", extra=None, controls=True):
    cfg = CASES[case]
    out = []
    for members in cfg["groups"].values():
        out += members
    ctl_members = []
    if controls and case in CONTROL_MERGE:
        for k in ("a", "b"):
            ctl_members += CONTROLS[case][k]
    for m in out + ctl_members:
        merge = (CONTROL_MERGE[case] if m in ctl_members else cfg["merge"])
        rid = rid_for(case, variant, seed, m)
        cmd = [".venv/bin/python", "run_musing.py", "--corpus", "bloomfield",
               "--target", m, "--set-ids", cfg["sets"], "--chronological",
               "--n-hypotheses", "8", "--tracing-model", "gemini-2.5-flash",
               "--goal-seeding", "--extract-anchors", "--use-anchor",
               "--anchored-perturbation", "--enable-split", "--enable-expiry",
               "--seed", str(seed), "--standard-prompt", variant,
               "--run-id", rid]
        if merge:
            cmd += ["--merge-speakers", merge]
        if extra:
            cmd += extra
        print(f"  {rid} ...", flush=True)
        with open(f"musing_out/{rid}.log", "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)


def self_control(case, seeds, variant="v1"):
    """The SAME person, two different seeds. Should separate by ~0.

    This is the tightest control available and it costs nothing: it reuses the
    runs already made. Where the same-side control (two engineers) still
    compares two different people, this compares one person with himself, so
    anything it registers is pure run-to-run variance. If the real split does
    not clear THIS, the metric is measuring sampling and nothing else.
    """
    out = []
    for members in CASES[case]["groups"].values():
        for m in members:
            ds = []
            for sd in seeds:
                st = standards(rid_for(case, variant, sd, m))
                if st is None:
                    continue
                d, n = dist(st)
                if d:
                    ds.append(d)
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    out.append(0.5 * sum(abs(ds[i][k] - ds[j][k]) for k in MODE_NAMES))
    return (sum(out) / len(out), len(out)) if out else (None, 0)


def score_control(case, seeds, variant="v1"):
    ctl = CONTROLS.get(case)
    if not ctl:
        return None
    tvs = []
    for sd in seeds:
        gd = {}
        for g in ("a", "b"):
            pooled = []
            for m in ctl[g]:
                st = standards(rid_for(case, variant, sd, m))
                if st is None:
                    return None
                pooled += st
            d, n = dist(pooled)
            if d is None:
                return None
            gd[g] = d
        tvs.append(0.5 * sum(abs(gd["a"][m] - gd["b"][m]) for m in MODE_NAMES))
    return sum(tvs) / len(tvs) if tvs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="color", choices=list(CASES) + ["both"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variant", default="v1", help="STANDARD_PROMPTS variant to run/score")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tag", default="",
                    help="distinguish an ARM sharing this case/variant/seed, e.g. 'p' for "
                         "a --character-profile arm. Empty keeps the historical run ids.")
    ap.add_argument("--extra", default=None,
                    help="extra flags passed through to run_musing.py, space separated, "
                         "e.g. '--character-profile'")
    a = ap.parse_args()
    globals()["TAG"] = a.tag
    extra = a.extra.split() if a.extra else None
    cases = list(CASES) if a.case == "both" else [a.case]
    seeds = list(range(a.seeds))
    total = []
    for c in cases:
        if not a.score_only:
            for s in seeds:
                launch(c, s, a.variant, extra=extra)
        print(f"\n{c} [{a.variant}]:")
        r, err = score_case(c, seeds, a.variant)
        if err:
            print(f"    {err}")
            continue
        ctl = score_control(c, seeds, a.variant)
        cons = r["consistency"]
        cs = "n/a (1 seed)" if cons is None else f"{cons:.2f}"
        print(f"  real split separates {r['mean_tv']:.2f}   "
              f"control (same side) {('%.2f' % ctl) if ctl is not None else '--'}")
        if ctl is None:
            print("  no control available -- score not computed")
            continue
        sc, npairs = self_control(c, seeds, a.variant)
        if sc is not None:
            print(f"  same PERSON, different seed: {sc:.2f} over {npairs} pairs"
                  + ("   <-- noise floor exceeds the real split"
                     if sc >= r["mean_tv"] else ""))
        margin = r["mean_tv"] - ctl
        score = max(0.0, margin) * (cons if cons is not None else 1.0)
        print(f"  MARGIN {margin:+.2f} x consistency {cs} = SCORE {score:.3f}"
              f"   (modal gap: {r['modal'][0]})")
        if margin <= 0.05:
            print("  -> the split is not distinguishable from two people on the SAME side")
        total.append(score)
        report_pooled(c, seeds, a.variant)
    if total:
        print(f"\nOVERALL {sum(total)/len(total):.3f}")


if __name__ == "__main__":
    main()
