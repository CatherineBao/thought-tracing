"""Does --revival-rebirth earn its mass move? The exp_4 metrics, per arm.

    python audit_revival.py rrctrl_s0 rrarm_s0 rrctrl_s1 rrarm_s1

exp_4 tested the fair-share reset on EVERY accepted mint and lost: birth mass
rose 0.047 -> 0.113 as intended, but the mass came from the established
hypotheses, so root-mass ESS fell, the collapse trigger fired 9 -> 13 times with
3 -> 6 consecutive re-fires, mint survival fell 59% -> 32% and argmax churn
8 -> 5. --revival-rebirth is the same reset restricted to REVIVED particles, so
it has to be judged on the same columns -- a birth-mass rise that buys another
root-mass collapse is the exp_4 result again at lower volume.

  revive      revivals, and how many were a SECOND return of the same anchor.
              A rising oscillation count is the failure the null gate exists to
              prevent, and it is the specific risk of making revival cheaper.
  birth       median post-step weight of a revived particle, AS A FRACTION OF
              FAIR SHARE. The raw weight is not comparable across arms: this
              population decays from 8 to 4-5 over a run, so 1/n moves by a
              factor of two within a single trace and two arms can post the same
              raw median while one is at fair share and the other is well under
              it. 1.00 means born at parity, which is what the flag forces.
  mint-birth  the same for non-revived mints, which the flag does NOT touch.
              It is the within-run control: if it moves too, the difference is
              the sampler, not the arm.
  rm-ess      median root-mass ESS over N. exp_4's actual failure.
  re-fire     longest run of consecutive steps that fired perturbation. exp_4
              went 3 -> 6; chasing a collapse the operator itself caused.
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
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-dir", default=ml.OUT_DIR)
    a = ap.parse_args()

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


if __name__ == "__main__":
    main()
