"""Measure the FORM of what a run produced, not whether it is true.

Two form defects were found by hand and are cheap to regress against, so they
live here as a guard rather than as a one-off analysis:

  BELIEF FORM. Propagation used to ask, in full, "What did {target} believe?",
  with the current context holding what somebody ELSE had just said. Given that
  question, "Katara believes that Aang is genuinely asking about the situation"
  is a correct answer, and the population filled with readings of the
  interlocutor. Baseline before the fix: 49-78% of top beliefs were about
  another character, 29-37% stated any aim.

  ANCHOR FORM. A commitment is what the target is trying to achieve. Once the
  positional-likelihood bug was fixed and split fired for the first time, its
  unconstrained prompt returned the next conversational move instead --
  23 of 42 anchors in one run were requests to another character.

Neither is a truth measure. A run can score perfectly here and still be wrong
about the episode; that is what the transcript audit is for. But a run that
scores badly here cannot be right for the right reason, which makes this the
cheaper gate to run first.

Usage:
  audit_form.py RUN [RUN ...]          # run ids, e.g. ep_Katara fix_katara v2_katara
"""
import argparse
import json
import re

import musing_layout as ml

# "X believes that <another character> ..." -- a reading of the interlocutor.
# Deliberately anchored at the start: a belief that OPENS on someone else is the
# failure. One that states an aim and then mentions another character is fine.
OTHER = re.compile(
    r"^\w[\w' -]* (?:believes?|believed|thinks?|thought|perceives?|perceived|"
    r"understands?|understood|sees?|saw|interprets?|interpreted) "
    r"(?:that )?(?:Aang|Sokka|Zuko|Katara|Toph|Suki|Azula|Hakoda|Rha|Yon Rha|"
    r"the commander|his|her|their|they|he|she)\b", re.I)

# Any explicit statement of aim. Generous on purpose: the point is to catch
# texts that name NO goal at all, not to grade the goal.
WANT = re.compile(
    r"\b(wants?|wanted|aims? to|intends?|intended|is trying to|was trying to|"
    r"seeks?|sought|needs? to|hopes? to|in order to|so that|goal is|"
    r"is determined to)\b", re.I)

QUESTIONY = re.compile(r"^\s*(ask|prompt|inquire|question|tell|request|demand|urge)\b", re.I)


def load(run_id):
    hits = ml.find(f"{run_id}-*.steps.jsonl") or ml.find(f"{run_id}*.steps.jsonl")
    if not hits:
        return None, None
    path = sorted(hits)[-1]
    with open(path, encoding="utf-8") as fh:
        return path, [json.loads(l) for l in fh if l.strip()]


def top_of(step):
    ps = step.get("particles") or []
    return max(ps, key=lambda x: x.get("weight") or 0) if ps else None


def audit(steps):
    tops = [t for t in (top_of(s) for s in steps) if t]
    texts = [(t.get("text") or "") for t in tops]
    top_anchors = {t.get("anchor") for t in tops if t.get("anchor")}
    all_anchors = {p.get("anchor") for s in steps for p in s["particles"] if p.get("anchor")}

    def clean(a):
        return len(a.split()) <= 8 and not QUESTIONY.match(a) and not a.rstrip().endswith("?")

    # An AIM can arrive two ways: as prose ("Zuko wants to earn Katara's
    # trust") or as the bare clause the anchor already holds ("Gain Katara's
    # full confidence."). Both state the goal; only the first matches WANT, so
    # counting WANT alone under-reports a run whose aims are terse. Treat a
    # text that OPENS on its own anchor as stating its aim.
    def states_aim(text, anchor):
        if WANT.search(text):
            return True
        if not anchor:
            return False
        a = anchor.strip().rstrip(".").lower()
        return bool(a) and text.strip().lower().startswith(a)

    aims = [states_aim(t.get("text") or "", t.get("anchor")) for t in tops]
    n = len(texts) or 1
    return {
        "steps": len(steps),
        "belief_other": sum(1 for t in texts if OTHER.match(t)) / n,
        "belief_aim": sum(aims) / n,
        "anchors_all": len(all_anchors),
        "anchors_top": len(top_anchors),
        "anchors_all_clean": sum(1 for a in all_anchors if clean(a)),
        "anchors_top_clean": sum(1 for a in top_anchors if clean(a)),
        "anchors_all_questiony": sum(1 for a in all_anchors if QUESTIONY.match(a)),
        "median_anchor_words": sorted(len(a.split()) for a in all_anchors)[len(all_anchors) // 2]
        if all_anchors else 0,
        "max_anchor_words": max((len(a.split()) for a in all_anchors), default=0),
        "top_anchors": sorted(top_anchors, key=len),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show-anchors", action="store_true")
    a = ap.parse_args()

    rows = []
    for r in a.runs:
        path, steps = load(r)
        if not steps:
            print(f"  {r}: no steps file found"); continue
        rows.append((r, audit(steps)))

    print(f"{'run':<16}{'steps':>6}{'belief: other':>15}{'belief: aim':>13}"
          f"{'anchors':>9}{'clean':>7}{'?-shaped':>10}{'med w':>7}{'max w':>7}")
    print("-" * 90)
    for r, d in rows:
        print(f"{r[:15]:<16}{d['steps']:>6}{d['belief_other']:>14.0%}{d['belief_aim']:>13.0%}"
              f"{d['anchors_all']:>9}{d['anchors_all_clean']:>7}"
              f"{d['anchors_all_questiony']:>10}{d['median_anchor_words']:>7}"
              f"{d['max_anchor_words']:>7}")
    print("\nbelief: other = top belief opens as a reading of another character (lower is better)")
    print("belief: aim   = top belief states any goal at all            (higher is better)")
    print("clean         = anchors <=8 words, not a question or a request to someone")

    if a.show_anchors:
        for r, d in rows:
            print(f"\n--- {r}: {d['anchors_top']} anchors that held top weight ---")
            for x in d["top_anchors"]:
                print(f"    {'ok ' if len(x.split()) <= 8 and not QUESTIONY.match(x) else 'BAD'} {x[:96]}")


if __name__ == "__main__":
    main()
