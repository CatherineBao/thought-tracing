"""Revival and mint health, per run and pooled across arms.

    python audit_revival.py v6_bb_Rodriguez rrctrl_s0
    python audit_revival.py --arm ctrl=rrctrl_s0,rrctrl_s1 --arm x=run_a,run_b

The per-run table is for reading one run. The --arm mode is for comparing two,
and it exists because the table is NOT safe to compare with: it reports a per-run
median, and mint birth mass spans 0.12 to 2.18 of fair share WITHIN a single run
on 4-12 mints. Reading a two-seed ordering of those medians as an effect is how
three interventions in a row were each briefly believed to work. Pooling the
underlying events and permuting the labels uses the same runs and the same API
spend and gives an honest p.

  revive      revivals, and how many were a SECOND return of the same anchor.
              Across the four runs that first logged it, 8 of 25 revivals were
              repeat returns -- the retire/revive oscillation the null gate in
              revive_retired is documented as preventing.
  birth       median post-step weight of a revived particle as a FRACTION OF
              FAIR SHARE. Never the raw weight: the population decays from 8 to
              4-6 over a run, so 1/n moves by a factor of two and two arms can
              post the same raw median at opposite ends of parity.
  mint-birth  the same for non-revived mints. Within-run control.
  rm-ess      median root-mass ESS over N.
  re-fire     longest run of consecutive steps that fired perturbation.
  survive     of the roots minted or revived at step t, the share still alive
              five steps later.
"""

import argparse
import glob
import json
import os
import statistics as st

import musing_layout as ml


def load(run_id, out_dir):
    hits = [f for f in glob.glob(os.path.join(out_dir, "**", "*.steps.jsonl"),
                                 recursive=True)
            if os.path.basename(f).startswith(run_id + "-")
            or os.path.basename(f) == run_id + ".steps.jsonl"]
    if not hits:
        return None
    return [json.loads(l) for l in open(hits[0], encoding="utf-8") if l.strip()]


def alive_at(recs, i, root):
    return any(p.get("root_id") == root for p in (recs[i].get("particles") or []))


def measure(recs):
    n = len(recs)
    revivals, oscillations = 0, 0
    birth_rev, birth_mint = [], []
    rm, refire, cur = [], 0, 0
    survive_hit, survive_tot = 0, 0
    argmax, churn = None, 0

    for i, r in enumerate(recs):
        parts = r.get("particles") or []
        if r.get("root_mass_ess") is not None:
            rm.append(float(r["root_mass_ess"]))
        fired = "perturb" in (r.get("operators_fired") or [])
        cur = cur + 1 if fired else 0
        refire = max(refire, cur)

        rev_idx = set()
        for v in (r.get("revived") or []):
            revivals += 1
            if (v.get("revivals") or 1) > 1:
                oscillations += 1
            if v.get("index") is not None:
                rev_idx.add(v["index"])

        minted = set(r.get("minted_roots") or [])
        fair = 1.0 / len(parts) if parts else 0.0
        for j, p in enumerate(parts):
            if not fair:
                break
            if j in rev_idx:
                birth_rev.append(float(p.get("weight") or 0.0) / fair)
            elif p.get("root_id") in minted and p.get("perturb_accepted"):
                birth_mint.append(float(p.get("weight") or 0.0) / fair)

        # five-step survival of everything the operator introduced this step
        if i + 5 < n:
            new_roots = {parts[j].get("root_id") for j in rev_idx} | minted
            for root in new_roots:
                if root:
                    survive_tot += 1
                    survive_hit += 1 if alive_at(recs, i + 5, root) else 0

        if parts:
            top = max(parts, key=lambda p: float(p.get("weight") or 0.0))
            if argmax is not None and top.get("root_id") != argmax:
                churn += 1
            argmax = top.get("root_id")

    md = lambda v: st.median(v) if v else float("nan")
    return {
        "steps": n,
        "surprise": sum(1 for r in recs if r.get("surprise")),
        "revivals": revivals,
        "oscillations": oscillations,
        "birth_rev": md(birth_rev),
        "n_rev": len(birth_rev),
        "birth_mint": md(birth_mint),
        "n_mint": len(birth_mint),
        "rm_ess": md(rm),
        "refire": refire,
        "survive": (survive_hit / survive_tot) if survive_tot else float("nan"),
        "churn": churn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out-dir", default=ml.OUT_DIR)
    ap.add_argument("--arm", action="append", default=None,
                    help="name=run1,run2 -- pool events across the arm's runs and "
                         "compare against the FIRST --arm given. Repeatable.")
    a = ap.parse_args()
    if a.arm:
        arms = {}
        for spec in a.arm:
            name, _, rids = spec.partition("=")
            arms[name] = [x for x in rids.split(",") if x]
        pooled(arms, a.out_dir)
        if not a.runs:
            return

    hdr = (f"{'run':<14}{'steps':>6}{'surp':>6}{'reviv':>6}{'osc':>5}"
           f"{'birth/fair':>12}{'mint/fair':>12}{'rm-ess':>8}{'re-fire':>9}"
           f"{'survive':>9}{'churn':>7}")
    print(hdr); print("-" * len(hdr))
    for rid in a.runs:
        recs = load(rid, a.out_dir)
        if not recs:
            print(f"{rid:<14}  -- not found")
            continue
        m = measure(recs)
        print(f"{rid:<14}{m['steps']:>6}{m['surprise']:>6}{m['revivals']:>6}"
              f"{m['oscillations']:>5}{m['birth_rev']:>8.2f}"
              f"({m['n_rev']:>2}){m['birth_mint']:>9.2f}({m['n_mint']:>2})"
              f"{m['rm_ess']:>8.3f}{m['refire']:>9}{m['survive']:>9.2f}"
              f"{m['churn']:>7}")



# --------------------------------------------------------------------------
# Pooled, event-level arm comparison.
#
# The table above reports a per-run MEDIAN and leaves two numbers an arm to be
# read by eye. That threw away most of the data and led to reading a two-seed
# ordering as a mechanism: mint birth mass spans 0.12 to 2.18 of fair share
# WITHIN one run on 4-12 mints, so its per-run median is not a level. Pooling
# the underlying events and comparing distributions uses the same runs and the
# same API spend, and says how much of the apparent difference survives.
#
# Unpaired permutation, not a t-test: these are bounded, skewed, small samples.
# It is also NOT a paired test and cannot be -- two arms diverge after the first
# step they disagree on, so there is no step-to-step correspondence to pair. For
# anything whose effect is immediate rather than downstream, audit_slots.py
# pairs it properly against the same populations and is the stronger instrument.
# --------------------------------------------------------------------------

def events(recs):
    """Per-event samples, not per-run summaries."""
    mint_birth, survival = [], []
    n = len(recs)
    for i, r in enumerate(recs):
        parts = r.get("particles") or []
        if not parts:
            continue
        fair = 1.0 / len(parts)
        rev = {v["index"] for v in (r.get("revived") or []) if v.get("index") is not None}
        minted = set(r.get("minted_roots") or [])
        for j, p in enumerate(parts):
            if j not in rev and p.get("root_id") in minted and p.get("perturb_accepted"):
                mint_birth.append(float(p.get("weight") or 0.0) / fair)
        if i + 5 < n:
            new = {parts[j].get("root_id") for j in rev} | minted
            for root in new:
                if root:
                    survival.append(1.0 if alive_at(recs, i + 5, root) else 0.0)
    return {"mint_birth": mint_birth, "survival": survival}


def perm_diff(a, b, trials=20000, seed=0):
    """Two-sided p for a difference in means, by label shuffling."""
    import numpy as _np
    a, b = _np.array(a, dtype=float), _np.array(b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    both = _np.concatenate([a, b])
    rng = _np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        rng.shuffle(both)
        if abs(both[:len(a)].mean() - both[len(a):].mean()) >= obs:
            hits += 1
    return hits / trials


def pooled(arms, out_dir):
    """arms: {name: [run_id, ...]}. The first is the control."""
    data = {}
    for name, rids in arms.items():
        acc = {"mint_birth": [], "survival": []}
        for rid in rids:
            recs = load(rid, out_dir)
            if not recs:
                print(f"!! {rid} not found")
                continue
            for k, v in events(recs).items():
                acc[k].extend(v)
        data[name] = acc

    ctrl = list(arms)[0]
    for metric in ("mint_birth", "survival"):
        print(f"\n{metric}  (pooled events, control = {ctrl})")
        print(f"  {'arm':<12}{'n':>5}{'mean':>9}{'median':>9}{'vs ctrl':>10}{'p':>8}")
        for name in arms:
            v = data[name][metric]
            if not v:
                continue
            d = "-" if name == ctrl else f"{st.mean(v) - st.mean(data[ctrl][metric]):>+.3f}"
            p = "-" if name == ctrl else f"{perm_diff(v, data[ctrl][metric]):.3f}"
            print(f"  {name:<12}{len(v):>5}{st.mean(v):>9.3f}{st.median(v):>9.3f}"
                  f"{d:>10}{p:>8}")


if __name__ == "__main__":
    main()
