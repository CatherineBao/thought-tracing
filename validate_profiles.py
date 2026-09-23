"""Is the external system's read of a person right? Checked against evidence it never saw.

THE PROBLEM THIS SOLVES. Once a profile is injected into hypothesis generation,
the hypotheses are downstream of it -- measured on this corpus, handed someone
else's record a target lands 0.08 from that record, inside the same-seed noise
floor. So "the traces agree with our read of Catherine" is circular and means
nothing. A profile cannot be validated by output it produced.

WHAT THIS DOES INSTEAD. Every profile makes a falsifiable claim: `settles_expect`,
the settlement mode its `settles` clause predicts. That claim is scored against
runs traced WITHOUT the profile -- record-only traces, which had no access to
it. The profile predicts; the transcript judges; the two never touch.

`settles_expect` is validator-only and never reaches a prompt (test_profiles
enforces it). If it did, this would be circular in exactly the way the whole
file exists to avoid.

READING THE OUTPUT.
  share      how much of this person's record-derived settlement mass landed in
             the mode their profile predicted.
  lift       that share against the same mode's share across everyone else. A
             profile that predicts the mode everybody uses has told you nothing,
             so lift, not share, is the score.
  verdict    SUPPORTED / UNSUPPORTED / CONTRADICTED.

A CONTRADICTED profile is the useful output: the external system believes
something about this person that their own conversations do not bear out, and
because the check never saw the profile, that is real evidence rather than an
echo. Feed it back to whatever produced the read, or gate the profile out with
run_musing's --confidence-gate.

  python validate_profiles.py                      # default record-only runs
  python validate_profiles.py --runs cov4s{seed}_{person} --seeds 0,1,2
"""
import argparse

import eval_motive_sep as e
import profiles as P

# A share this far below the cohort baseline is not merely unsupported -- the
# record is actively pointing somewhere else.
CONTRADICTED_LIFT = -0.05
SUPPORTED_LIFT = 0.05


def record_modes(run_pattern, person, seeds):
    """This person's settlement distribution from runs that never saw a profile."""
    pooled = []
    for sd in seeds:
        st = e.standards(run_pattern.format(seed=sd, person=person))
        if st:
            pooled += st
    if not pooled:
        return None, 0
    return e.dist(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="bloomfield")
    ap.add_argument("--runs", default="cov4s{seed}_{person}",
                    help="run-id template for RECORD-ONLY traces (no --character-profile). "
                         "Using profiled runs here would make the check circular.")
    ap.add_argument("--people", default="Wolf,Lyubovsky,Rovani")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    profs = P.load_profiles(a.corpus)
    if not profs:
        raise SystemExit(f"no profiles for {a.corpus}")
    people = [p.strip() for p in a.people.split(",") if p.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    rows, dists = [], {}
    for person in people:
        d, n = record_modes(a.runs, person, seeds)
        if d is None:
            print(f"[!] no record-only runs for {person} ({a.runs})")
            continue
        dists[person] = (d, n)

    for person, (d, n) in dists.items():
        rec = P.people(profs).get(person) or {}
        claim = rec.get("settles_expect")
        conf = P.confidence_of(profs, person)
        if not claim:
            continue
        share = d.get(claim, 0.0)
        # cohort baseline for the SAME mode, this person excluded
        others = [dd.get(claim, 0.0) for p2, (dd, _) in dists.items() if p2 != person]
        base = sum(others) / len(others) if others else 0.0
        lift = share - base
        verdict = ("SUPPORTED" if lift >= SUPPORTED_LIFT else
                   "CONTRADICTED" if lift <= CONTRADICTED_LIFT else "unsupported")
        rows.append((person, conf, claim, share, base, lift, verdict, n))

    print(f"\n{'person':<12}{'conf':>6}  {'profile predicts':<18}"
          f"{'record':>8}{'cohort':>8}{'lift':>8}   verdict")
    for person, conf, claim, share, base, lift, verdict, n in rows:
        cs = f"{conf:.2f}" if conf is not None else "  --"
        print(f"{person:<12}{cs:>6}  {claim:<18}{share:>8.2f}{base:>8.2f}"
              f"{lift:>+8.2f}   {verdict}")

    print(f"\nScored against {a.runs} -- runs traced with NO profile, so the "
          f"evidence is independent of the claim.")
    bad = [r for r in rows if r[6] == "CONTRADICTED"]
    if bad:
        print("\nCONTRADICTED -- the record points elsewhere than the external read:")
        for person, conf, claim, share, base, lift, verdict, n in bad:
            d = dists[person][0]
            actual = max(e.MODE_NAMES, key=lambda m: d[m])
            print(f"  {person}: profile says '{claim}' ({conf:.2f} confidence), "
                  f"record says '{actual}' ({d[actual]:.2f})")
        print("  -> feed back to whatever produced the read, or suppress it with "
              "--confidence-gate.")


if __name__ == "__main__":
    main()
