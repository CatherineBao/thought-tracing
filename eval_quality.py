"""Quality evaluation for traced beliefs, grounded in corpus structure.

Every result so far is a diversity or concentration metric -- roots, ESS,
separation. None of them say whether the hypotheses are BETTER. A filter that
holds 8 commitments instead of collapsing to 3 has achieved nothing if the 8
are worse.

Gold contexts carry ground truth that makes this checkable without human
labels: the target was demonstrably ABSENT for `missed_set_ids`. So the content
of those sets is what the target CANNOT know, and the traced belief must not
attribute it to them.

Two scores per run:
  false attribution  belief credits the target with knowledge only available in
                     the missed sets. Lower is better. This is the ToM failure
                     the whole method exists to avoid.
  grounding          belief reflects what the target DID witness. Higher is
                     better. Guards against scoring well by saying nothing.

Judged by an LLM against the two texts, with the judgement forced to cite the
specific claim -- an unsupported verdict is discarded rather than counted.
"""
import argparse
import glob
import musing_layout as ml
import json
import os
import re

import run_musing as rm


def load_summary(path):
    # the dump is JSONL -- one record appended per traced context, so json.load
    # fails on any file with more than one. Take the last complete record.
    d = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
    # 'aggregated' is a bool flag in the dump, not the summary -- the belief
    # text is under 'text'. Guard the type rather than trusting the key name.
    for k in ("text", "summary", "aggregated"):
        v = d.get(k)
        if isinstance(v, str) and len(v) > 200:
            return v
    return ""


def judge(tracer, target, witnessed, missed, belief):
    sys_p = (
        f"You audit a model's inferred beliefs about {target} for a specific failure: "
        f"crediting {target} with knowledge they could not have had.\n\n"
        f"You are given (a) what {target} WITNESSED, (b) content {target} was ABSENT for, "
        f"and (c) the inferred belief.\n\n"
        f"Answer exactly:\n"
        f"FALSE_ATTRIBUTION: <0-10>  (10 = belief heavily credits {target} with the absent "
        f"content; 0 = none of it)\n"
        f"EVIDENCE: <quote the specific phrase, or NONE>\n"
        f"GROUNDING: <0-10>  (10 = belief is well supported by what {target} witnessed; "
        f"0 = untethered or vacuous)\n\n"
        f"Score FALSE_ATTRIBUTION above 0 ONLY if you can quote the phrase. A belief that is "
        f"merely plausible, or that speculates without asserting, is not false attribution.")
    prompt = (f"<{target} WITNESSED>\n{witnessed[:6000]}\n</{target} WITNESSED>\n\n"
              f"<{target} WAS ABSENT FOR>\n{missed[:6000]}\n</{target} WAS ABSENT FOR>\n\n"
              f"<INFERRED BELIEF>\n{belief[:6000]}\n</INFERRED BELIEF>")
    raw = tracer.tracer_model.interact(prompt, system_prompt=sys_p, temperature=0,
                                       max_tokens=1024, stage="quality")
    fa = re.search(r"FALSE_ATTRIBUTION\s*:\s*(\d+)", raw, re.I)
    ev = re.search(r"EVIDENCE\s*:\s*(.+)", raw, re.I)
    gr = re.search(r"GROUNDING\s*:\s*(\d+)", raw, re.I)
    if not fa or not gr:
        return None
    ev_txt = (ev.group(1).strip() if ev else "NONE")
    score = int(fa.group(1))
    # a positive score without a citation is discarded, not counted
    if score > 0 and ev_txt.upper().startswith("NONE"):
        score = 0
    return {"false_attribution": score, "grounding": int(gr.group(1)), "evidence": ev_txt[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="oppenheimer")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="must match the --max-chars the runs used")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="must match the --min-chars the runs used")
    ap.add_argument("--min-target-turns", type=int, default=0,
                    help="minimum turns the TARGET speaks. High-unknowable contexts have\n                         an absent target, who therefore speaks less and yields a short\n                         trajectory -- 5 steps in one case. Both must be constrained.")
    ap.add_argument("--min-missed-chars", type=int, default=0,
                    help="must match the --min-missed-chars the runs used")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run-id prefixes to compare, e.g. p2clean p3e_n2")
    a = ap.parse_args()

    corpus = rm.load_corpus(a.corpus)
    # MUST replicate the driver's selection exactly. Judging a belief traced on
    # one gold context against another context's missed sets makes the score
    # meaningless -- the "unknowable" content would be from a different gap.
    gold = rm.select_gold(corpus)
    gold = [g for g in gold if a.min_chars <= (g.get("stitched_chars") or 0) <= a.max_chars]
    if a.min_missed_chars:
        def _missed(ctx):
            return sum(len(corpus["by_id"][s]["full_context"])
                       for s in ctx["missed_set_ids"] if s in corpus["by_id"])
        gold = [g for g in gold if _missed(g) >= a.min_missed_chars]
        gold.sort(key=lambda g: _missed(g))
    if a.min_target_turns:
        def _tturns(ctx):
            n = 0
            for sid in ctx["set_ids"]:
                st = corpus["by_id"].get(sid)
                if st:
                    n += sum(1 for t in st["turns"] if t["speaker"] == ctx["target_agent"])
            return n
        gold = [g for g in gold if _tturns(g) >= a.min_target_turns]
    if not gold:
        raise SystemExit("no gold context matches the driver's selection")
    ctx = gold[0]
    missed = "\n\n".join(corpus["by_id"][s]["full_context"]
                         for s in ctx["missed_set_ids"] if s in corpus["by_id"])
    witnessed = "\n\n".join(corpus["by_id"][s]["full_context"]
                            for s in ctx["set_ids"]
                            if s in corpus["by_id"] and s not in ctx["missed_set_ids"])

    tracer = rm.build_tracer(rm.make_args(run_id="quality", output_dir="musing_out"))
    print(f"context: {ctx['context_id']}")
    print(f"  target absent for {ctx['missed_set_ids']} ({len(missed)} chars of unknowable content)\n")

    results = {}
    for pref in a.runs:
        scores = []
        for f in sorted(ml.find(f"tracer-*runid-{pref}*.jsonl")):
            belief = load_summary(f)
            if len(belief) < 200:
                continue
            r = judge(tracer, ctx["target_agent"], witnessed, missed, belief)
            if r:
                scores.append(r)
                print(f"  {pref:12s} {os.path.basename(f)[-28:]:28s} "
                      f"FA={r['false_attribution']:2d} GR={r['grounding']:2d}  {r['evidence'][:52]}")
        results[pref] = scores

    print("\n=== SUMMARY (lower FA better, higher GR better) ===")
    for pref, sc in results.items():
        if not sc:
            print(f"  {pref:14s}: no scorable runs")
            continue
        fa = [s["false_attribution"] for s in sc]
        gr = [s["grounding"] for s in sc]
        print(f"  {pref:14s}: n={len(sc)}  false_attribution {sum(fa)/len(fa):.2f} "
              f"(range {min(fa)}-{max(fa)})   grounding {sum(gr)/len(gr):.2f} "
              f"(range {min(gr)}-{max(gr)})")


if __name__ == "__main__":
    main()
