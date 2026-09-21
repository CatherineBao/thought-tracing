"""Probe the 3e coherence gate with KNOWN-incoherent candidates.

Acceptance sits at 24/25 = 0.96 in the 3e N/2 arm. That is not pinned at 1.0 --
the gate rejected one -- but it is high enough that the gate might be
near-rubber-stamping, and Phase 4's split will reuse the same accept/reject
machinery with a different prompt.

A rate alone cannot distinguish "the proposals were genuinely coherent" from
"the gate accepts anything". This feeds it candidates that CONTRADICT the
observed record by construction, alongside genuinely coherent ones, and
measures whether it separates them.
"""
import argparse
import json
import re

import run_musing as rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    c = json.load(open("musing_out/probe_context.json"))
    args = rm.make_args(run_id="gate", output_dir="musing_out", n_hypotheses=a.n,
                        use_anchor=True, anchored_perturbation=True)
    tr = rm.build_tracer(args)
    tr.target_agent = c["target_agent"]
    tr.assumption = c["assumption"]
    ctx = tr.interleave_context_and_perception(c["contexts"], c["perceptions"])
    ta = c["target_agent"]

    # by construction: 3 coherent, 3 contradicting the observed record
    cands = [
        ("COHERENT",   "secure the project timeline",
         f"{ta} believes the schedule is slipping and wants it recovered."),
        ("COHERENT",   "protect his own standing",
         f"{ta} believes his authority is being tested and wants it preserved."),
        ("COHERENT",   "get a candid assessment",
         f"{ta} believes he is being told a sanitised version and wants the truth."),
        ("CONTRADICTS", "leave the project entirely",
         f"{ta} believes he has already resigned and is no longer involved in any of this."),
        ("CONTRADICTS", "celebrate the completed project",
         f"{ta} believes the work finished successfully years ago and this is a retrospective."),
        ("CONTRADICTS", "avoid ever meeting Oppenheimer",
         f"{ta} believes he has never met Oppenheimer and is speaking to a stranger."),
    ]

    gate_sys = (
        f"Decide whether each proposed account CONTRADICTS what {ta} has "
        f"demonstrably observed.\n\n"
        f"Reject ONLY for contradiction with the observed record. Do NOT reject an account "
        f"for being surprising, unlikely, or unflattering -- a genuinely different account "
        f"is the point.\n\nAnswer one line each:\n1: COHERENT or CONTRADICTS\n...")
    block = "\n".join(f"{i+1}. COMMITMENT: {cm} | BELIEF: {b}"
                      for i, (_, cm, b) in enumerate(cands))
    raw = tr.tracer_model.interact(
        f"<observed record>\n{ctx}\n</observed record>\n\n<proposed accounts>\n{block}\n"
        f"</proposed accounts>", system_prompt=gate_sys, temperature=0,
        max_tokens=1024, stage="gate_probe")
    verdicts = dict((int(n), v.upper()) for n, v in re.findall(
        r"^\s*\**\s*(\d+)\s*\**\s*[:.\)]\s*\**\s*(COHERENT|CONTRADICTS)", raw, re.I | re.M))

    print(f"gate probe: {len(cands)} candidates, 3 coherent / 3 contradicting by construction\n")
    tp = fp = tn = fn = unparsed = 0
    for i, (truth, cm, _) in enumerate(cands, start=1):
        got = verdicts.get(i)
        if got is None:
            unparsed += 1
            mark = "UNPARSED"
        else:
            ok = (got == truth)
            mark = "correct" if ok else "*** WRONG ***"
            if truth == "CONTRADICTS":
                tp += (got == "CONTRADICTS"); fn += (got == "COHERENT")
            else:
                tn += (got == "COHERENT"); fp += (got == "CONTRADICTS")
        print(f"  {i}. [{truth:>11}] -> {str(got):>11}  {mark}   {cm[:44]}")
    total = tp + tn + fp + fn
    print(f"\n  caught contradictions : {tp}/3")
    print(f"  passed coherent ones  : {tn}/3")
    print(f"  false rejections      : {fp}")
    print(f"  MISSED contradictions : {fn}   <- rubber-stamping if high")
    print(f"  unparsed              : {unparsed}")
    if total:
        print(f"  accuracy              : {(tp+tn)/total:.2f}")
    print()
    if fn >= 2:
        print("  VERDICT: gate is near-rubber-stamping. Split must NOT reuse it unchanged.")
    elif tp == 3 and tn == 3:
        print("  VERDICT: gate discriminates cleanly. 0.96 acceptance reflects good proposals,")
        print("           not a permissive gate. Safe for split to reuse.")
    else:
        print("  VERDICT: partial discrimination -- tighten before split reuses it.")


if __name__ == "__main__":
    main()
