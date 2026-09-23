"""Find the moments where somebody could have done otherwise, and what they did.

    python choice_points.py --corpora bloomfield,boeing,oppenheimer
    python choice_points.py --spot-check 40        # writes a hand-review sheet
    python choice_points.py --score-spot-check     # reads the filled sheet back

WHY CHOICE POINTS AND NOT TURNS. The filter's evidence unit is "the observed
next action", and PRECHECKS says what that costs: the honest summary of what
somebody's messages DO is "define numerical thresholds", which describes the
message and not the reason for sending it. Most turns carry no decision --
they are acknowledgements, logistics, a pasted number. A motive is only
testable where the person could have done otherwise and did one thing rather
than another, so those moments are the only ones worth forecasting, and
everything downstream is built on this file rather than on turns.

THE UNIT IS A CHANNEL STREAM, NOT A SET. bloomfield's sets average 5.4 turns --
too short to contain a choice AND the context that makes it one. So sets are
concatenated per channel in date order and cut into windows. The time split is
applied per CHOICE POINT by the set its action lands in, not per window: a
choice made after the cut is held out even when the conversation leading to it
started before it, which is what a forecast actually faces.

THE VALIDITY THREAT, NAMED. The extractor sees the whole window INCLUDING what
the person did, so it can write an alternative set that makes the actual choice
the obvious one -- longer, better-argued, or simply the only one that fits the
situation line. Nothing downstream can tell that apart from a good forecast.
Three things are done about it, and none of them is trust:

  1. Alternatives are normalised to a fixed seven-label taxonomy plus a short
     gist, so a model cannot spend more words on the one that happened.
  2. `situation` is written from BEFORE the choice and is checked for outcome
     leakage by a separate pass.
  3. choice_forecast.py runs a BLIND arm that sees only the alternative labels
     -- no transcript, no person, no situation. If blind beats chance, the
     alternative set is the signal and every other number in the report is
     void. That arm is the reason this file can be believed at all.

WHAT HAND-CHECKING IS FOR, AND WHAT IT IS NOT. `--spot-check` writes a sheet
asking one question per sampled point: was this a real choice with realistic
alternatives. It does not ask whether any motive is insightful. A human is the
only available check on whether the choice is real; a human is a bad check on
whether an explanation is deep, which is how the last five designs were scored
into uninterpretability.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import random
import re
import subprocess
import sys

from choice_llm import CachedModel, FakeModel, extract_json

HERE = os.path.dirname(os.path.abspath(__file__))
MUSING = os.path.join(HERE, "data", "musing")
OUT_DIR = os.path.join(HERE, "musing_out", "choice_points")
SPLITS = os.path.join(HERE, "splits.json")

# The seven a person can do with a disagreement. Fixed, because an open label
# set makes the actual action describable in a way the alternatives are not,
# and that difference alone is forecastable.
ACTIONS = ("CONCEDE", "HOLD", "ESCALATE", "DROP", "IGNORE", "RE_RAISE", "TRADE")

ACTION_GLOSS = {
    "CONCEDE": "give the point to the other side, or adopt their position",
    "HOLD": "restate or defend their own position without moving",
    "ESCALATE": "widen it -- pull in a third party, a manager, a meeting, a formal process",
    "DROP": "let it go and move on to something else, without conceding",
    "IGNORE": "say nothing to it at all while continuing to speak about other things",
    "RE_RAISE": "bring back a point that was previously settled, dropped or passed over",
    "TRADE": "give one thing to get another -- an explicit or implicit exchange",
}

SYSTEM = (
    "You read work conversations and find DECISION POINTS. A decision point is a "
    "moment where one named person could realistically have done at least two "
    "different things about a disagreement, an ask, or an open question, and did "
    "one of them.\n\n"
    "It is NOT a decision point when the person is acknowledging, relaying a "
    "number, answering a direct factual question, doing logistics, or when only "
    "one response was realistically available.\n\n"
    "The seven things a person can do:\n"
    + "\n".join(f"  {a} -- {ACTION_GLOSS[a]}" for a in ACTIONS)
    + "\n\nYou will be shown numbered turns. Return JSON only."
)

TEMPLATE = """Conversation from #{channel} ({corpus}).

{turns}

Participants: {people}

Find every decision point in these turns. For each one return an object:

  "person":      exactly one participant name as written above
  "at_turn":     the turn number where that person ACTS on the choice
  "situation":   <= 25 words describing what is in front of them, written from
                 BEFORE they act. It must not say or imply what they did.
  "alternatives": 3 to 5 objects, each
                   "action":   one of {actions}
                   "gist":     <= 12 words, what that would look like here
                   "gives_up": <= 12 words, what taking it costs them
  "actual":      the action label from `alternatives` that they actually took
  "why_real":    <= 20 words on why more than one option was genuinely open

Rules that matter:
  - Every alternative must be one somebody in their position could actually have
    taken here. Do not pad the list with options that were never available.
  - Spend the SAME amount of care on every alternative. Do not write the one
    that happened more fully than the others.
  - `situation` describes the position, never the outcome.
  - Return [] if these turns contain no real decision point. An empty list is a
    correct answer and is preferred to a manufactured one.

Return a JSON array and nothing else."""


# --------------------------------------------------------------------------
# corpus -> per-channel chronological turn stream
# --------------------------------------------------------------------------

def load_corpus(name):
    with open(os.path.join(MUSING, f"{name}_dialogue.json"), encoding="utf-8") as fh:
        return json.load(fh)


def channel_streams(corpus):
    """Turns per channel, in corpus order, each tagged with its set.

    Corpus order is date order -- make_splits verifies it -- so concatenating a
    channel's sets in file order reconstructs the thing people actually read.
    """
    streams = collections.OrderedDict()
    for s in load_corpus(corpus):
        key = s.get("channel") or s.get("timeline") or corpus
        for i, t in enumerate(s["turns"]):
            streams.setdefault(key, []).append({
                "speaker": t.get("speaker"), "text": t.get("text") or "",
                "sent": t.get("sent"), "message_id": t.get("message_id"),
                "set_id": s["set_id"], "date": s.get("date"),
                "title": s.get("title"), "turn_in_set": i,
            })
    return streams


def windows(stream, size=40, stride=32):
    """Overlapping windows. Overlap so a choice near a boundary is still seen
    with the turns that made it one; duplicates are removed afterwards on
    (person, at_turn)."""
    out = []
    i = 0
    while i < len(stream):
        chunk = stream[i:i + size]
        if len(chunk) >= 8:
            out.append((i, chunk))
        if i + size >= len(stream):
            break
        i += stride
    return out


def render(chunk, offset):
    lines = []
    for j, t in enumerate(chunk):
        body = re.sub(r"\s+", " ", t["text"]).strip()
        if len(body) > 600:
            body = body[:600] + " ..."
        lines.append(f"[{offset + j}] {t['speaker']}: {body}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def load_sides():
    """set_id -> 'dev' | 'test', from the frozen split."""
    if not os.path.exists(SPLITS):
        raise SystemExit("splits.json is missing; run `python make_splits.py` first")
    with open(SPLITS, encoding="utf-8") as fh:
        splits = json.load(fh)
    side = {}
    for corpus in (splits.get("corpora") or {}).values():
        for block in corpus.get("blocks", []):
            for sid in block.get("dev", []):
                side[sid] = "dev"
            for sid in block.get("test", []):
                side[sid] = "test"
    return side


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def validate(raw, chunk, offset, people):
    """Turn one model object into a choice point, or say why it is not one.

    Every rejection is counted and reported. A silently dropped malformed
    record would make the extraction look cleaner than it is, and the reject
    profile is the cheapest signal that a prompt change went wrong.
    """
    if not isinstance(raw, dict):
        return None, "not an object"
    person = (raw.get("person") or "").strip()
    if person not in people:
        return None, "person not a participant"
    try:
        at = int(raw.get("at_turn"))
    except (TypeError, ValueError):
        return None, "at_turn not an integer"
    if not offset <= at < offset + len(chunk):
        return None, "at_turn outside the window"
    if chunk[at - offset]["speaker"] != person:
        return None, "at_turn is not that person's turn"
    alts = raw.get("alternatives")
    if not isinstance(alts, list) or not 2 <= len(alts) <= 6:
        return None, "alternatives not a list of 2-6"
    clean, seen = [], set()
    for a in alts:
        if not isinstance(a, dict):
            return None, "alternative not an object"
        act = str(a.get("action") or "").strip().upper().replace("-", "_")
        if act not in ACTIONS or act in seen:
            return None, "bad or duplicate action label"
        seen.add(act)
        clean.append({"action": act,
                      "gist": " ".join(str(a.get("gist") or "").split())[:120],
                      "gives_up": " ".join(str(a.get("gives_up") or "").split())[:120]})
    actual = str(raw.get("actual") or "").strip().upper().replace("-", "_")
    if actual not in seen:
        return None, "actual not among the alternatives"
    if len(clean) < 3:
        return None, "fewer than 3 alternatives"
    t = chunk[at - offset]
    return {
        "person": person, "at_turn": at, "set_id": t["set_id"], "date": t["date"],
        "title": t["title"],
        "situation": " ".join(str(raw.get("situation") or "").split())[:300],
        "why_real": " ".join(str(raw.get("why_real") or "").split())[:240],
        "alternatives": clean, "actual": actual,
        "n_alternatives": len(clean),
    }, None


def extract(corpus, model, size=40, stride=32, limit=None, min_person_turns=3):
    sides = load_sides()
    streams = channel_streams(corpus)
    rejects = collections.Counter()
    prompts, meta = [], []

    for channel, stream in streams.items():
        # A person with almost no turns in a channel has no behaviour to
        # forecast, and their "choices" are the ones most likely to be invented.
        counts = collections.Counter(t["speaker"] for t in stream)
        people = sorted(p for p, n in counts.items() if p and n >= min_person_turns)
        if len(people) < 2:
            continue
        for offset, chunk in windows(stream, size, stride):
            here = sorted({t["speaker"] for t in chunk} & set(people))
            if len(here) < 2:
                continue
            prompts.append(TEMPLATE.format(
                channel=channel, corpus=corpus, turns=render(chunk, offset),
                people=", ".join(here), actions=" / ".join(ACTIONS)))
            meta.append((channel, offset, chunk, set(here)))

    if limit:
        prompts, meta = prompts[:limit], meta[:limit]
    if not prompts:
        return [], rejects

    raws = model.batch_interact(prompts, system_prompts=[SYSTEM] * len(prompts),
                                temperature=0, max_tokens=4096)

    found, seen = [], set()
    for raw, (channel, offset, chunk, people) in zip(raws, meta):
        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            rejects["unparseable reply"] += 1
            continue
        for obj in parsed:
            cp, why = validate(obj, chunk, offset, people)
            if cp is None:
                rejects[why] += 1
                continue
            key = (corpus, channel, cp["person"], cp["at_turn"])
            if key in seen:            # windows overlap on purpose
                rejects["duplicate across windows"] += 1
                continue
            seen.add(key)
            cp.update({
                "cp_id": f"{corpus}:{channel}:{cp['person']}:{cp['at_turn']}",
                "corpus": corpus, "channel": channel,
                "side": sides.get(cp["set_id"], "unassigned"),
                "action_text": chunk[cp["at_turn"] - offset]["text"][:800],
            })
            found.append(cp)
    found.sort(key=lambda c: (c["channel"], c["at_turn"]))
    return found, rejects


# --------------------------------------------------------------------------
# store + context
# --------------------------------------------------------------------------

def store_path(corpus, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"{corpus}.jsonl")


def save(corpus, points, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    with open(store_path(corpus, out_dir), "w", encoding="utf-8") as fh:
        for cp in points:
            fh.write(json.dumps(cp, ensure_ascii=False) + "\n")


def load(corpus=None, out_dir=OUT_DIR, side=None):
    paths = ([store_path(corpus, out_dir)] if corpus
             else sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))))
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cp = json.loads(line)
                    if side is None or cp.get("side") == side:
                        out.append(cp)
    return out


class Context:
    """The turns a forecaster is allowed to see, and nothing after them."""

    def __init__(self, corpora):
        self.streams = {c: channel_streams(c) for c in corpora}

    def prefix(self, cp, max_turns=60):
        stream = self.streams[cp["corpus"]][cp["channel"]]
        start = max(0, cp["at_turn"] - max_turns)
        return render(stream[start:cp["at_turn"]], start)

    def turn(self, cp):
        return self.streams[cp["corpus"]][cp["channel"]][cp["at_turn"]]


# --------------------------------------------------------------------------
# hand review
# --------------------------------------------------------------------------

SHEET_HEADER = """# Choice-point spot check

{n} points sampled at random from {total} (seed {seed}).

ONE question per point, and it is not about insight:

    Was this a real choice -- could this person, at this moment, realistically
    have taken at least two of the listed alternatives?

Write `VERDICT: real` or `VERDICT: not-real` on the line provided. Add
`NOTE:` freely. Do not judge whether any explanation is interesting; that is
what the forecast scoring is for, and a human judging it is how the last five
designs became uninterpretable.

Score the filled sheet with `python choice_points.py --score-spot-check`.

---
"""


def sheet(points, ctx, n, seed, total, context_turns=12):
    rng = random.Random(seed)
    sample = rng.sample(points, min(n, len(points)))
    out = [SHEET_HEADER.format(n=len(sample), total=total, seed=seed)]
    for i, cp in enumerate(sample, 1):
        out.append(f"## {i}. `{cp['cp_id']}`  ({cp['side']}, {cp['date']}, {cp['title']})\n")
        out.append("```")
        out.append(ctx.prefix(cp, context_turns) or "(no prior turns in this channel)")
        out.append("```")
        out.append(f"\n**Person:** {cp['person']}  \n**Situation:** {cp['situation']}  \n"
                   f"**Why it was open:** {cp['why_real']}\n")
        out.append("| | action | would look like | gives up |")
        out.append("|---|---|---|---|")
        for a in cp["alternatives"]:
            mark = "**<-- taken**" if a["action"] == cp["actual"] else ""
            out.append(f"| {mark} | `{a['action']}` | {a['gist']} | {a['gives_up']} |")
        body = re.sub(r"\s+", " ", cp["action_text"]).strip()
        out.append(f"\n**What they actually wrote:** {body[:400]}\n")
        out.append("VERDICT: \nNOTE: \n\n---\n")
    return "\n".join(out)


def score_sheet(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    ids = re.findall(r"^## \d+\. `([^`]+)`", text, flags=re.M)
    verdicts = re.findall(r"^VERDICT:\s*(\S+)?\s*$", text, flags=re.M)
    rows = list(zip(ids, verdicts + [None] * (len(ids) - len(verdicts))))
    filled = [(i, (v or "").lower()) for i, v in rows if v]
    good = [i for i, v in filled if v.startswith("real")]
    bad = [i for i, v in filled if v.startswith("not")]
    other = [(i, v) for i, v in filled if not (v.startswith("real") or v.startswith("not"))]
    return {"sampled": len(ids), "filled": len(filled), "real": len(good),
            "not_real": len(bad), "unrecognised": other,
            "precision": (len(good) / len(filled)) if filled else None,
            "not_real_ids": bad}


# --------------------------------------------------------------------------

def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def report(corpus, points, rejects):
    sides = collections.Counter(c["side"] for c in points)
    acts = collections.Counter(c["actual"] for c in points)
    people = collections.Counter(c["person"] for c in points)
    print(f"\n{corpus}: {len(points)} choice points")
    print(f"  side      {dict(sides)}")
    print(f"  action    {dict(acts.most_common())}")
    print(f"  people    {len(people)} ({', '.join(f'{p}:{n}' for p, n in people.most_common(6))})")
    if rejects:
        print(f"  rejected  {sum(rejects.values())}: "
              + ", ".join(f"{k} x{v}" for k, v in rejects.most_common(6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="bloomfield,boeing,oppenheimer")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="windows per corpus")
    ap.add_argument("--fake", action="store_true", help="pipeline only; produces no evidence")
    ap.add_argument("--spot-check", type=int, default=0, metavar="N",
                    help="write a hand-review sheet for N sampled points")
    ap.add_argument("--spot-check-out", default=os.path.join(HERE, "SPOTCHECK.md"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-spot-check", action="store_true")
    a = ap.parse_args()

    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]

    if a.score_spot_check:
        res = score_sheet(a.spot_check_out)
        print(json.dumps(res, indent=1))
        if res["filled"] < res["sampled"]:
            print(f"\n{res['sampled'] - res['filled']} of {res['sampled']} still blank")
        if res["precision"] is not None:
            print(f"\nreal-choice precision: {res['precision']:.0%} "
                  f"({res['real']}/{res['filled']})")
        return

    if a.spot_check:
        points = [cp for c in corpora for cp in load(c, a.out_dir)]
        if not points:
            raise SystemExit("no choice points on disk; extract first")
        ctx = Context(corpora)
        with open(a.spot_check_out, "w", encoding="utf-8") as fh:
            fh.write(sheet(points, ctx, a.spot_check, a.seed, len(points)))
        print(f"wrote {a.spot_check_out}: {min(a.spot_check, len(points))} of "
              f"{len(points)} points for hand review")
        return

    model = FakeModel() if a.fake else CachedModel(a.model)
    summary = {}
    for corpus in corpora:
        points, rejects = extract(corpus, model, a.window, a.stride, a.limit)
        save(corpus, points, a.out_dir)
        report(corpus, points, rejects)
        summary[corpus] = {"points": len(points), "rejected": sum(rejects.values()),
                           "reject_profile": dict(rejects)}

    meta = {"at": dt.date.today().isoformat(), "git_commit": git_head(),
            "model": model.model_name, "window": a.window, "stride": a.stride,
            "corpora": summary, **model.stats()}
    with open(os.path.join(a.out_dir, "_extraction.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"\n{model.stats()}")
    if a.fake:
        print("FAKE MODEL -- pipeline exercised, nothing here is evidence")


if __name__ == "__main__":
    main()
