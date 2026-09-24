"""CaSiNo -> a `data/musing` corpus, choice points on the EARLY CUT, and an answer key.

WHAT CASINO IS FOR, STATED SO IT IS NOT OVER-READ. Its priorities are assigned by
the task and negotiators state them out loud, so it tests INFERENCE OF STATED
INCENTIVES, not recovery of motives people miss. It is Milestone 1 calibration.
Measured: median ONE structured choice point per person, 1,025 of ~1,080
proposers submit exactly one deal, 97 concession points corpus-wide. There is no
sequence here, so no M2 number may be quoted from this corpus (PREREG).

THE EARLY CUT IS THE WHOLE DESIGN. An explicit prose allocation precedes the
`Submit-Deal` in 1,158/1,181 = 0.981 of cases -- the two people have just spelled
the deal out and the submission formalises it. Forecasting from the full prefix
is reading comprehension, `context_neutral` would sit at ceiling and the headroom
gate would fail. So the prefix is cut at THE FIRST EXPLICIT ALLOCATION PROPOSAL
BY EITHER PARTY, which is selection on the INPUT.

The cut is what makes the corpus usable at all: the silent subgroup -- people who
stated no priority in their own prefix -- is 0.182 under the full cut and 0.629
under the early cut. It turns CaSiNo's one genuine test from a footnote into a
dataset.

THE CUT RULE READS ONLY THE INPUT. It is a frozen regex over the prose and never
consults the submitted deal. That matters enough that test_hindsight covers it:
the 0.629 figure depends on it entirely.

THE LABEL is which issue the proposer claims the largest share of, read from
`task_data.issue2youget`. TIES GET THEIR OWN LABELS -- 33.5% of proposals claim
two issues equally, and a deal claiming two equally is a different claim from one
that picks a favourite. Excluding them would be selection on the outcome and
would delete a third of the corpus; breaking them would invent a preference the
proposal does not express.

THE ANSWER KEY IS SCORING-ONLY. `value2issue` is the hidden priority order and
`value2reason` the person's own stated reasons; neither may reach a prompt.
`value2reason` is keyed High/Medium/Low, so putting it in a prompt would hand
over the ground truth -- which is why the stated-reason particle is built from
the CHAT PREFIX instead.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics as st
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "casino", "casino.json")
SOURCE = "https://raw.githubusercontent.com/kushalchawla/CaSiNo/main/data/casino.json"
MUSING = os.path.join(HERE, "data", "musing")
CP_DIR = os.path.join(HERE, "musing_out", "choice_points")
KEY_DIR = os.path.join(HERE, "data", "casino", "answer_keys")

ISSUES = ("Food", "Water", "Firewood")
SPECIAL = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

# The seven labels: three singles, three pairs, one all-three. Plus OTHER, which
# is always offered and always scored -- a point whose actual action is not among
# the alternatives is never discarded, because that is selection on the outcome.
LABELS = ("Food", "Water", "Firewood",
          "Firewood+Food", "Firewood+Water", "Food+Water",
          "Firewood+Food+Water", "OTHER")

_NUM = r"(?:\d+|one|two|three|1|2|3|all|both)"
ALLOC_RE = re.compile(rf"\b{_NUM}\b[^.?!]{{0,40}}\b(food|water|firewood|wood)\b"
                      rf"|\b(food|water|firewood|wood)\b[^.?!]{{0,25}}\b{_NUM}\b", re.I)
NEED_RE = re.compile(r"\b(need|needs|needed|want|wants|priority|prioritize|important|"
                     r"most|essential|crucial|require|requires|low on|short on|out of|"
                     r"running low|care about|matter|matters)\b", re.I)
ISSUE_RE = re.compile(r"\b(food|water|firewood|wood)\b", re.I)

# CaSiNo carries no timestamps. Dates are synthesised from dialogue_id so that a
# contiguous id range is a contiguous date range, which is what make_splits
# assumes. Order is real (id order); the calendar is not, and nothing reads it as
# one.
EPOCH_YEAR = 2020


def fetch(path=RAW):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(SOURCE, path)
    return path


def speaker_of(agent_id):
    """Short, unique, prefix-free tokens -- the chat labeller matches line prefixes."""
    return "A" if agent_id == "mturk_agent_1" else "B"


def synthetic_date(dialogue_id):
    day = dialogue_id % 28 + 1
    month = (dialogue_id // 28) % 12 + 1
    year = EPOCH_YEAR + dialogue_id // (28 * 12)
    return f"{year:04d}-{month:02d}-{day:02d}"


def first_allocation_turn(logs):
    """Index of the first explicit allocation proposal by EITHER party.

    INPUT-SIDE ONLY: a quantity adjacent to an issue word in the prose, or a
    `Submit-Deal`. It never reads `task_data`, so the cut cannot know what was
    proposed -- only that a proposal happened.
    """
    for i, c in enumerate(logs):
        t = (c.get("text") or "").strip()
        if t == "Submit-Deal":
            return i
        if t in SPECIAL:
            continue
        if ALLOC_RE.search(t):
            return i
    return None


def states_priority(text):
    return bool(NEED_RE.search(text or "") and ISSUE_RE.search(text or ""))


def claim_label(issue2youget):
    """Which issue the proposer claims most of. Ties are their own label."""
    try:
        v = {k: int(issue2youget[k]) for k in ISSUES}
    except (KeyError, TypeError, ValueError):
        return None
    top = max(v.values())
    return "+".join(sorted(k for k in ISSUES if v[k] == top))


def convert(raw):
    """Return (sets, choice_points, answer_keys, report)."""
    sets, points, keys = [], [], {}
    rejects = collections.Counter()
    prose_first = 0

    for dl in raw:
        did = dl.get("dialogue_id")
        if did is None:
            rejects["no dialogue_id"] += 1
            continue
        logs = dl.get("chat_logs") or []
        info = dl.get("participant_info") or {}
        turns = [{"speaker": speaker_of(c.get("id")),
                  "text": " ".join((c.get("text") or "").split())}
                 for c in logs if (c.get("text") or "").strip() not in SPECIAL]
        if not turns:
            rejects["no prose turns"] += 1
            continue

        set_id = f"casino-{did:04d}"
        sets.append({
            "set_id": set_id, "corpus": "casino", "conv_id": did,
            "channel": "campsite", "timeline": "campsite",
            "title": f"campsite negotiation {did}", "date": synthetic_date(did),
            # Both participants hear everything, so there is no attendance
            # asymmetry and no gap. CaSiNo's asymmetry is the PRIVATE PRIORITY
            # ORDER, which is richer -- see data/musing/README.txt on the roles.
            "tier": "silver",
            "speakers": ["A", "B"], "n_turns": len(turns),
            "full_context": "\n".join(f"{t['speaker']}: {t['text']}" for t in turns),
            "turns": turns,
        })

        cut = first_allocation_turn(logs)
        if cut is not None and (logs[cut].get("text") or "").strip() != "Submit-Deal":
            prose_first += 1

        seen_person = set()
        for i, c in enumerate(logs):
            if (c.get("text") or "").strip() != "Submit-Deal":
                continue
            agent = c.get("id")
            if agent in seen_person:
                continue            # first submission per person; the rest are concessions
            lab = claim_label((c.get("task_data") or {}).get("issue2youget") or {})
            if lab is None:
                rejects["submit-deal without a usable allocation"] += 1
                continue
            seen_person.add(agent)
            me = speaker_of(agent)
            at = cut if (cut is not None and cut < i) else i
            prefix = [{"speaker": speaker_of(x.get("id")),
                       "text": " ".join((x.get("text") or "").split())}
                      for x in logs[:at]
                      if (x.get("text") or "").strip() not in SPECIAL]
            silent = not any(states_priority(t["text"]) for t in prefix
                             if t["speaker"] == me)
            points.append({
                "cp_id": f"{set_id}-{me}", "corpus": "casino", "set_id": set_id,
                # `person` is the speaker token the prompt uses; `person_uid` is
                # the identity the per-person checks group on. They differ here
                # because every dialogue calls its participants A and B, and a
                # participant appears in exactly one dialogue.
                "person": me, "person_uid": f"{set_id}:{me}",
                "kind": "allocation", "at_turn": at,
                "situation": f"{me} now proposes how to split the packages.",
                "alternatives": [{"action": l, "gist": _gist(l), "gives_up": _gives_up(l)}
                                 for l in LABELS],
                "n_alternatives": len(LABELS),
                "actual": lab if lab in LABELS else "OTHER",
                "prefix_turns": prefix, "silent": silent,
                "turns_since_previous_choice_point": None,   # one point per person here
                "date": synthetic_date(did),
            })
        # ANSWER KEY -- scoring only, never a prompt
        keys[set_id] = {
            "schema": "casino.answer_key/1", "set_id": set_id,
            "participants": {
                speaker_of(a): {
                    "value2issue": (v.get("value2issue") or {}),
                    "value2reason": (v.get("value2reason") or {}),
                    "outcomes": (v.get("outcomes") or {}),
                } for a, v in info.items()},
        }

    report = {
        "dialogues": len(raw), "sets": len(sets), "choice_points": len(points),
        "people_with_a_point": len({(p["set_id"], p["person"]) for p in points}),
        "prose_allocation_before_submission": prose_first,
        "silent_subgroup": sum(1 for p in points if p["silent"]),
        "label_counts": dict(collections.Counter(p["actual"] for p in points).most_common()),
        "prefix_turns": {
            "mean": round(st.mean([len(p["prefix_turns"]) for p in points]), 2),
            "median": st.median([len(p["prefix_turns"]) for p in points]),
        } if points else None,
        "rejects": dict(rejects),
    }
    return sets, points, keys, report


def _gist(label):
    if label == "OTHER":
        return "something not listed here"
    parts = label.split("+")
    return ("claims the largest share of " + " and ".join(p.lower() for p in parts)
            if len(parts) < 3 else "splits everything evenly")


def _gives_up(label):
    if label == "OTHER":
        return "unknown"
    rest = [i for i in ISSUES if i not in label.split("+")]
    return ("concedes " + " and ".join(r.lower() for r in rest)) if rest else "concedes nothing"


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return None


def write(sets, points, keys, out_musing=MUSING, out_cp=CP_DIR, out_keys=KEY_DIR):
    os.makedirs(out_musing, exist_ok=True)
    os.makedirs(out_cp, exist_ok=True)
    os.makedirs(out_keys, exist_ok=True)
    with open(os.path.join(out_musing, "casino_dialogue.json"), "w", encoding="utf-8") as fh:
        json.dump(sets, fh)
    with open(os.path.join(out_musing, "casino_speakers.json"), "w", encoding="utf-8") as fh:
        json.dump({"A": "Participant A", "B": "Participant B"}, fh, indent=1)
    # No attendance gaps by construction; written so the loader finds the file.
    with open(os.path.join(out_musing, "casino_gaps.json"), "w", encoding="utf-8") as fh:
        json.dump([], fh)
    with open(os.path.join(out_cp, "casino.jsonl"), "w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(out_keys, "casino_answer_key.json"), "w", encoding="utf-8") as fh:
        json.dump({"git_commit": git_head(), "keys": keys}, fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=MUSING)
    ap.add_argument("--structured-only", action="store_true",
                    help="the honest denominator: no free-text offer parsing")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    raw = json.load(open(fetch(), encoding="utf-8"))
    sets, points, keys, report = convert(raw)
    if not a.dry_run:
        write(sets, points, keys, out_musing=a.out)
    if a.report or a.dry_run:
        n = report["choice_points"]
        print(json.dumps(report, indent=2))
        if n:
            print(f"\nprose allocation precedes the submission: "
                  f"{report['prose_allocation_before_submission']}/{report['sets']} = "
                  f"{report['prose_allocation_before_submission']/report['sets']:.3f}")
            print(f"silent subgroup (early cut): {report['silent_subgroup']}/{n} = "
                  f"{report['silent_subgroup']/n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
