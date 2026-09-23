"""How much of the population is weight-identical, and why it matters.

    python audit_ties.py
    python audit_ties.py --runs v6_ rrctrl_

THE FINDING THIS EXISTS FOR. A block of particles carrying EXACTLY the same
weight is invisible to every rule that ranks by weight -- expiry, resampling,
the perturbation triggers and any choice of which slot a mint takes over. It is
not a tie in the ordinary sense: these are particles the evidence has never
separated, and the filter cannot prefer any of them to any other.

Measured over every logged run: the weakest half is exactly tied on 55% of
surprise steps, and the largest weight-identical block is a median 17% of the
population -- rising to a median 43% on the tied steps themselves, which is what
makes those steps tied. The block GROWS as a run proceeds, from a median 13% in
a run's first half to 20% in its second, in 115 of 187 runs. That bounds a whole
class of intervention. Three separate rules for
choosing which weak particle a mint replaces were built, measured and removed
after this: each could only act on the minority of steps where the weak half was
separable at all, which is why none of them moved anything downstream.

TWO CAUSES RULED OUT, because both are the obvious guess and both are wrong:

  the eps floor      would clamp weak particles to one value, but
                     mass_moved_by_floor is 0 on 90% of tied steps
  resample uniformity resampling resets every weight to 1/n, but a resample
                     fired on NONE of them, and the whole population is uniform
                     on only 15%

What is left is the accumulator. w_t ~ w_{t-1}^alpha * L_t^beta is a monotone
map of the prior, so it can never separate two particles that arrive equal --
only the likelihood can, and only when it differs between them. Particles start
equal at initialization, and a step that fires surprise sets the likelihood to
[1.0]*n BY DESIGN, so every surprise step actively refuses to separate anything
while the branch it feeds spends a slot chosen by weight. Whatever is tied stays
tied, and the block grows.
"""

import argparse
import glob
import os
import statistics as st

import musing_layout as ml
from audit_margins import load


def blocks(w, tol=1e-12):
    """Largest set of particles sharing one weight, as a share of the population."""
    if not w:
        return 0.0
    counts = {}
    for x in w:
        key = round(float(x), 12)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=ml.OUT_DIR)
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.out_dir, "**", "*.steps.jsonl"),
                             recursive=True))
    if a.runs:
        files = [f for f in files
                 if any(os.path.basename(f).startswith(p) for p in a.runs)]

    hdr = (f"{'run':<26}{'steps':>6}{'pop':>6}{'biggest tied block':>20}"
           f"{'weak half tied':>16}")
    print(hdr); print("-" * len(hdr))
    all_block, all_tied, all_first, all_last = [], [], [], []
    for f in files:
        recs = [r for r in load(f) if r.get("weights_pre")]
        if len(recs) < 10:
            continue
        blk, tied, pops = [], [], []
        for r in recs:
            w = [float(x) for x in r["weights_pre"]]
            n = len(w)
            pops.append(n)
            blk.append(blocks(w))
            if r.get("surprise"):
                nrep = max(1, n // 4)
                pool = sorted(w)[:max(2 * nrep, nrep + 1)]
                tied.append(1.0 if max(pool) - min(pool) <= 0 else 0.0)
        name = os.path.basename(f).split("-")[0][:25]
        print(f"{name:<26}{len(recs):>6}{st.median(pops):>6.0f}"
              f"{st.median(blk):>19.0%}"
              f"{(st.mean(tied) if tied else float('nan')):>15.0%}")
        all_block += blk
        all_tied += tied
        half = len(blk) // 2
        all_first.append(st.median(blk[:half]))
        all_last.append(st.median(blk[half:]))

    if all_block:
        print(f"\npooled: biggest weight-identical block is a median "
              f"{st.median(all_block):.0%} of the population")
        print(f"        weak half exactly tied on {st.mean(all_tied):.0%} of "
              f"surprise steps")
        print(f"        first half of a run {st.median(all_first):.0%} -> "
              f"second half {st.median(all_last):.0%}  "
              f"({sum(1 for x, y in zip(all_first, all_last) if y > x)}"
              f"/{len(all_first)} runs grow it)")


if __name__ == "__main__":
    main()
