"""Set the resample and 3e triggers from observed distributions -- OFFLINE.

Both are currently inherited constants (resample 1/3, 3e root-mass 0.5), and
every inherited constant checked so far has landed in the high-density region of
this data: N/2 for resampling, cosine 0.90 for merge, 3x-mean for split.

Two rules this script enforces:

  OFFLINE, not live. The resample trigger acts on an ESS trajectory that drifts
  systematically as the accumulator warms (~21 steps at alpha=0.85). A percentile
  over a live, still-warming series sets the cut ~3x too high and fires exactly
  when resampling is least warranted (measured: p10 live-early 0.49-0.53 vs
  post-warm-up 0.152). Merge escapes this because its percentile is over
  within-step pairwise similarities, which are stationary.

  POST-WARM-UP basis. Percentiles are computed over steps >= WARMUP only.

Placement is chosen for LOW DENSITY, not for a target firing rate: a cut inside
the mass makes crossings near-ties, which is what made resample count vary 1-2
on identical contexts.
"""
import argparse
import glob
import musing_layout as ml
import statistics

import trace_log as t

WARMUP = 21   # ~3 time constants at alpha = 0.85


def series(pattern, field, warmup=WARMUP, min_steps=28):
    out = []
    for f in ml.find(pattern):
        st = t.read_steps(f)
        if len(st) < min_steps:
            continue
        for s in st[warmup:]:
            n = len(s.particles) or 1
            if field == "ess" and s.posterior_ess_pre:
                out.append(s.posterior_ess_pre / n)
            elif field == "rootmass" and s.root_mass_ess is not None:
                out.append(s.root_mass_ess)
    return sorted(out)


def density_report(vals, label, candidates):
    if not vals:
        print(f"  {label}: NO DATA")
        return None
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    print(f"  {label}  n={len(vals)}")
    print(f"    p05={q(.05):.3f} p10={q(.10):.3f} p25={q(.25):.3f} "
          f"median={statistics.median(vals):.3f} p75={q(.75):.3f}")
    print(f"    {'cut':>8} | {'below':>7} | {'within +/-0.05':>15} | placement")
    best = None
    for c in candidates:
        below = sum(1 for v in vals if v < c)
        near = sum(1 for v in vals if abs(v - c) < 0.05)
        inside = q(.25) <= c <= q(.75)
        verdict = "INSIDE the mass" if inside else ("low-density OK" if near <= max(1, len(vals)//40) else "near the mass")
        print(f"    {c:8.3f} | {below:3d} ({100*below/len(vals):4.1f}%) | {near:15d} | {verdict}")
        if not inside and (best is None or near < best[1]):
            best = (c, near)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n8", default="p3e_n2_*.steps.jsonl")
    ap.add_argument("--n12", default="p4n12_*.steps.jsonl")
    a = ap.parse_args()

    print(f"OFFLINE trigger calibration (post-warm-up, step >= {WARMUP})\n")
    print("=== RESAMPLE trigger: normalized pre-operator ESS ===")
    for lab, pat in (("N=8 ", a.n8), ("N=12", a.n12)):
        vals = series(pat, "ess")
        best = density_report(vals, lab, [0.50, 0.40, 0.333, 0.30, 0.25, 0.20])
        if best:
            print(f"    -> lowest-density candidate: {best[0]:.3f} "
                  f"(divisor {1/best[0]:.2f})\n")
        else:
            print()
    print("=== 3e trigger: root-mass ESS ===")
    for lab, pat in (("N=8 ", a.n8), ("N=12", a.n12)):
        vals = series(pat, "rootmass")
        best = density_report(vals, lab, [0.60, 0.50, 0.40, 0.333, 0.25])
        if best:
            print(f"    -> lowest-density candidate: {best[0]:.3f}\n")
        else:
            print()
    print("NOTE: 3e's trigger is a conjunction (mass AND collapse), so its cut only")
    print("      gates events where duplicate roots already exist. Placement still")
    print("      matters -- it decides which collapses count as worth repairing.")


if __name__ == "__main__":
    main()
