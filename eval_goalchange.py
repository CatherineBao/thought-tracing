"""Does the trace track a goal change that was not proposable at step 0?

The discriminating test between the two routes. Never-collapse preserves the
initial commitments but cannot represent one nobody proposed at initialization.
Collapse-then-repair mints new commitments mid-trajectory, so it can.

Ground truth (established from the transcript, both sides quoted):
  EARLY  convince J-Oppenheimer not to appeal the charges
  LATE   be confirmed as Commerce Secretary
  mutually exclusive: YES

Scored on the FINAL belief, which should reflect the LATE goal.
"""
import argparse, glob, json, os, re
import run_musing as rm

EARLY = "convince J-Oppenheimer not to appeal the charges against him"
LATE = "be confirmed as Commerce Secretary"


def load(path):
    d = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return d.get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    tr = rm.build_tracer(rm.make_args(run_id="gc", output_dir="musing_out"))
    sys_p = (
        f"Strauss's goal CHANGES during this transcript.\n"
        f"  EARLY goal: {EARLY}\n  LATE goal: {LATE}\n"
        f"They are mutually exclusive.\n\n"
        f"Given a model's inferred beliefs, score:\n"
        f"TRACKS_LATE: <0-10>  (10 = clearly attributes the LATE goal to Strauss)\n"
        f"LATE_EVIDENCE: <quote or NONE>\n"
        f"STUCK_EARLY: <0-10>  (10 = still attributes only the EARLY goal at the end)\n"
        f"EARLY_EVIDENCE: <quote or NONE>\n"
        f"Score above 0 only with a quote.")
    print(f"{'run':>16} | {'TRACKS_LATE':>11} | {'STUCK_EARLY':>11} | evidence")
    agg = {}
    for pref in a.runs:
        rows = []
        for f in sorted(glob.glob(f"musing_out/tracer-*runid-{pref}*.jsonl")):
            b = load(f)
            if not isinstance(b, str) or len(b) < 200:
                continue
            raw = tr.tracer_model.interact(
                f"<INFERRED BELIEFS FOR Strauss>\n{b[-7000:]}\n</INFERRED BELIEFS>",
                system_prompt=sys_p, temperature=0, max_tokens=1024, stage="gc")
            g = lambda k: (int(m.group(1)) if (m := re.search(rf"{k}\s*:\s*(\d+)", raw, re.I)) else 0)
            ev = re.search(r"LATE_EVIDENCE\s*:\s*(.+)", raw, re.I)
            tl, se = g("TRACKS_LATE"), g("STUCK_EARLY")
            rows.append((tl, se))
            print(f"{os.path.basename(f)[-16:]:>16} | {tl:11d} | {se:11d} | "
                  f"{(ev.group(1).strip()[:44] if ev else '')}")
        agg[pref] = rows
    print("\n=== SUMMARY ===")
    for pref, rows in agg.items():
        if not rows:
            print(f"  {pref:14s}: no runs"); continue
        tl = sum(r[0] for r in rows)/len(rows); se = sum(r[1] for r in rows)/len(rows)
        print(f"  {pref:14s}: n={len(rows)}  tracks_late {tl:.1f}  stuck_early {se:.1f}")


if __name__ == "__main__":
    main()
