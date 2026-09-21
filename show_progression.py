"""Render how the hypothesis population changed over a run.

Non-linear progression, as defined: strengthen, weaken, merge, split, revise.
Shows the commitments themselves, not just the metrics.
"""
import argparse, glob, sys
import trace_log as t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--width", type=int, default=62)
    a = ap.parse_args()
    fs = sorted(glob.glob(f"musing_out/{a.run}*.steps.jsonl"))
    if not fs:
        sys.exit(f"no steps file for {a.run}")
    st = t.read_steps(fs[0])

    # stable colour-free label per root, in order of first appearance
    label, nxt = {}, 0
    for s in st:
        for p in s.particles:
            r = p.root_id or p.lineage_id
            if r not in label:
                label[r] = chr(ord("A") + nxt) if nxt < 26 else f"z{nxt}"
                nxt += 1

    print(f"\n{'='*96}\nHYPOTHESIS POPULATION OVER {len(st)} STEPS\n{'='*96}")
    print("  each letter is an ancestral COMMITMENT (root). bar = share of belief mass.\n")
    prev = set()
    for s in st:
        alive = {}
        for p in s.particles:
            r = label[p.root_id or p.lineage_id]
            alive[r] = alive.get(r, 0.0) + (p.weight or 0.0)
        born = set(alive) - prev
        died = prev - set(alive)
        ops = ",".join(s.operators_fired)
        bar = ""
        for r in sorted(alive, key=lambda k: -alive[k]):
            n = max(1, round(alive[r] * 40))
            bar += (r * n)
        marks = []
        if born: marks.append(f"+{''.join(sorted(born))}")
        if died: marks.append(f"-{''.join(sorted(died))}")
        print(f"  {s.step_idx:3d} |{bar:<42}| {ops:<22} {' '.join(marks)}")
        prev = set(alive)

    print(f"\n{'='*96}\nCOMMITMENTS\n{'='*96}")
    seen = {}
    for s in st:
        for p in s.particles:
            r = label[p.root_id or p.lineage_id]
            if r not in seen and p.anchor:
                seen[r] = (s.step_idx, p.anchor)
    for r in sorted(seen):
        born, anc = seen[r]
        print(f"  {r}  (born step {born:2d})  {anc[:a.width]}")

    print(f"\n{'='*96}\nTRAJECTORIES  (ancestral mass per step)\n{'='*96}")
    mass = t.ancestral_mass(st)
    for rid, ser in mass.items():
        r = label.get(rid, "?")
        pts = "".join(" ▁▂▃▄▅▆▇█"[min(8, int((v or 0) * 12))] for v in ser)
        vals = [v for v in ser if v is not None]
        if not vals:
            continue
        tag = "grew" if vals[-1] > vals[0] * 1.2 else ("faded" if vals[-1] < vals[0] * 0.8 else "flat")
        print(f"  {r} |{pts}| {vals[0]:.2f} -> {vals[-1]:.2f}  {tag}")

    summ = t.nonlinearity_summary(st)
    print(f"\n{'='*96}\nNON-LINEAR PROGRESSION\n{'='*96}")
    for k in ("reversals_median_long_lived", "argmax_churn", "split_count",
              "merge_count", "perturb_count", "resample_count", "distinct_roots"):
        print(f"  {k:28s}: {summ.get(k)}")


if __name__ == "__main__":
    main()
