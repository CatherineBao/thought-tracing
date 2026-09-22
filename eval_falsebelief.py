"""False-belief evaluation on Odyssey disguise scenes.

The false-attribution metric saturated: baseline scored 0 even with 15,756
chars of unknowable content, because upstream perception-tracking already
bounds what a target can know. It measured a component neither arm changes.

This tests the thing the method actually exists for -- a belief that diverges
from reality. Odyssey carries `disguised_as` on 41 turns as ground truth:
Minerva speaking as Mentor, Ulysses as the beggar. Characters present hold a
FALSE belief about who is speaking, and the correct trace must report the
false belief, not the truth.

Scoring is deliberately two-sided, because either error is a failure:
  holds_false_belief  does the trace attribute the DISGUISED identity to the
                      target? (correct ToM)
  leaks_truth         does it credit the target with knowing the TRUE identity?
                      (the classic false-belief failure)

A trace can fail by leaking the truth OR by being too vague to commit, so a
third score catches vacuity.
"""
import argparse
import glob
import musing_layout as ml
import json
import os
import re

import run_musing as rm

DATA = "data/musing/odyssey_dialogue.json"


def scenes_with_disguise(min_turns=12):
    with open(DATA, encoding="utf-8") as fh:
        d = json.load(fh)
    out = []
    for s in d:
        dis = [t for t in s["turns"] if t.get("disguised_as")]
        # keep real disguises: a distinct persona, not self or null
        real = [t for t in dis
                if t["disguised_as"] not in (None, "null", "", t["speaker"])]
        if real and s["n_turns"] >= min_turns:
            personas = sorted({(t["speaker"], t["disguised_as"]) for t in real})
            targets = [sp for sp in s["speakers"]
                       if sp not in {t["speaker"] for t in real}]
            if targets:
                out.append({"set": s, "personas": personas, "targets": targets,
                            "n_disguised": len(real)})
    out.sort(key=lambda x: -x["n_disguised"])
    return out


def judge(tracer, target, true_id, persona, belief):
    sys_p = (
        f"In this scene {true_id} is speaking while DISGUISED as {persona}. "
        f"{target} does not know this and should believe they are speaking with "
        f"{persona}.\n\n"
        f"You are given a model's inferred beliefs for {target}. Score it:\n\n"
        f"HOLDS_FALSE_BELIEF: <0-10>  (10 = clearly attributes to {target} the belief "
        f"that they are dealing with {persona}; 0 = no such attribution)\n"
        f"LEAKS_TRUTH: <0-10>  (10 = credits {target} with knowing it is really "
        f"{true_id}; 0 = never does)\n"
        f"EVIDENCE: <quote the phrase for whichever score is higher, or NONE>\n"
        f"COMMITTAL: <0-10>  (10 = takes a clear position on who {target} thinks they "
        f"are dealing with; 0 = vague, hedged, or silent on identity)\n\n"
        f"Score LEAKS_TRUTH above 0 only if you can quote the phrase.")
    raw = tracer.tracer_model.interact(
        f"<INFERRED BELIEFS FOR {target}>\n{belief[:7000]}\n</INFERRED BELIEFS>",
        system_prompt=sys_p, temperature=0, max_tokens=1024, stage="falsebelief")
    g = lambda k: (int(m.group(1)) if (m := re.search(rf"{k}\s*:\s*(\d+)", raw, re.I)) else None)
    fb, lt, cm = g("HOLDS_FALSE_BELIEF"), g("LEAKS_TRUTH"), g("COMMITTAL")
    ev = re.search(r"EVIDENCE\s*:\s*(.+)", raw, re.I)
    ev_txt = ev.group(1).strip() if ev else "NONE"
    if fb is None or lt is None:
        return None
    if lt > 0 and ev_txt.upper().startswith("NONE"):
        lt = 0
    return {"false_belief": fb, "leaks_truth": lt, "committal": cm or 0,
            "evidence": ev_txt[:110]}


def run_meta(prefix, out_dir="musing_out"):
    """Read what the run ACTUALLY traced, rather than re-deriving it.

    Four separate evaluations in this project produced invalid numbers because
    the eval re-applied selection logic and picked a different context or target
    than the runner did -- judging one target's beliefs against another's ground
    truth. The run records its own target_agent and context_id; read those.
    """
    path = ml.runs_jsonl(out_dir)
    if not os.path.exists(path):
        return None
    meta = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("run_id", "")).startswith(prefix):
                meta = r
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scene-index", type=int, default=0)
    a = ap.parse_args()

    sc = scenes_with_disguise()[a.scene_index]
    true_id, persona = sc["personas"][0]
    # target comes from the RUN, not from re-derived selection
    meta = run_meta(a.runs[0])
    target = (meta or {}).get("target_agent")
    if not target:
        raise SystemExit("cannot read target_agent from runs.jsonl -- refusing to guess")
    if target not in sc["targets"] and target not in sc["set"]["speakers"]:
        raise SystemExit(f"run target {target!r} is not a speaker in {sc['set']['set_id']}")
    print(f"  [target read from runs.jsonl: {target}]")
    print(f"scene {sc['set']['set_id']}: {true_id} disguised as {persona} "
          f"({sc['n_disguised']} turns)")
    print(f"  target holding the false belief: {target}\n")

    tracer = rm.build_tracer(rm.make_args(run_id="fb", output_dir="musing_out"))
    results = {}
    for pref in a.runs:
        scores = []
        for f in sorted(ml.find(f"tracer-*runid-{pref}*.jsonl")):
            d = {}
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            pass
            belief = d.get("text", "")
            if not isinstance(belief, str) or len(belief) < 200:
                continue
            r = judge(tracer, target, true_id, persona, belief)
            if r:
                scores.append(r)
                print(f"  {pref:10s} {os.path.basename(f)[-26:]:26s} "
                      f"FB={r['false_belief']:2d} LEAK={r['leaks_truth']:2d} "
                      f"COM={r['committal']:2d}  {r['evidence'][:46]}")
        results[pref] = scores
    print("\n=== SUMMARY (higher FB better, lower LEAK better) ===")
    for pref, sc_ in results.items():
        if not sc_:
            print(f"  {pref:12s}: no scorable runs")
            continue
        f = lambda k: sum(s[k] for s in sc_) / len(sc_)
        print(f"  {pref:12s}: n={len(sc_)}  false_belief {f('false_belief'):.2f}  "
              f"leaks_truth {f('leaks_truth'):.2f}  committal {f('committal'):.2f}")


if __name__ == "__main__":
    main()
