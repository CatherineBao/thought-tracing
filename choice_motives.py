"""Propose candidate motives from the EARLIER choices, thirteen ways.

    python choice_motives.py --corpora bloomfield,boeing,oppenheimer
    python choice_motives.py --fake            # pipeline only, no evidence

WHAT A MOTIVE HAS TO BE HERE. Something that makes a prediction. PRECHECKS
records what happens without that constraint: once split started firing, 23 of
42 anchors came back as "Ask Zuko to leverage Fire Nation intelligence
networks" -- the next thing to say, not an aim. A motive that cannot be wrong
about a future choice is not a hypothesis, it is a summary, and the forecast
leg would score it as noise while it reads as insight.

THE TWELVE, AND WHY THEY ARE POINTED AT A LEDGER. methods.py exists because
every hypothesis source in the filter is a READER: it sees one transcript and
reports what the transcript says, which works when the motive is spoken and
fails when it is not. Here each method is handed the person's earlier CHOICES
rather than their prose -- situation, what was open, what they took -- so the
thing being read is a pattern of decisions, and a method that can only
paraphrase has nothing to paraphrase.

THE THIRTEENTH IS NOT ONE OF THEM. The revealed-preference pass sees no
situation text, no message text and no stated reason -- only, per choice, what
each forgone option would have cost and which one was taken. It is the one
generator that cannot be reading what the person SAID about themselves, which
is the failure mode `presentation` is named after and the one the other twelve
share. Its prompt is built by a different function on purpose: routing it
through the method machinery would give it the ledger's prose back.

EARLIER ONLY. Every generator sees the dev side of splits.json and nothing
else. A motive fitted on a choice it is later scored against is not a forecast,
and the split is loaded here rather than passed in so no caller can widen it.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess

import methods as M
from choice_llm import CachedModel, FakeModel, extract_json
from choice_points import ACTIONS, load as load_points

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "musing_out", "choice_points", "_motives.jsonl")

# A generator is only given a person with enough earlier choices to show a
# pattern, and only one who still has held-out choices to be scored on. Both
# floors are here rather than at the call site so every arm sees the same cast.
MIN_DEV = 4
MIN_TEST = 3

FORM = (
    "A MOTIVE is a standing aim: what this person is trying to achieve or "
    "protect, across occasions.\n"
    "  - <= 25 words, one sentence, third person, no name needed.\n"
    "  - It must PREDICT. Someone holding it should be able to say what this "
    "person would do at a choice they have not seen.\n"
    "  - It is never a move ('ask X for Y'), never a restatement of what "
    "happened, never tied to one thread, date or number.\n"
    "  - It may be wrong. A motive nothing could contradict is worthless here.\n"
)

SYSTEM = (
    "You propose motives for a named person from a ledger of choices they made. "
    "You are not summarising the ledger; you are proposing the aim that would "
    "make those choices the obvious ones, including the aim they would not "
    "state out loud.\n\n" + FORM + "\nReturn JSON only."
)

REVEALED_SYSTEM = (
    "You infer what somebody values from what they gave up, and from nothing "
    "else.\n\n"
    "You are shown a ledger of choices. For each one you see what each "
    "available option would have COST them, and which option they took. You do "
    "NOT see what they said, what the situation was, or how they explained "
    "themselves. Deliberately: people's stated reasons and their revealed "
    "trade-offs come apart, and this pass exists to read only the second.\n\n"
    "Work out what they consistently sacrificed and what they consistently "
    "protected, then state the aim that ordering implies.\n\n" + FORM
    + "\nReturn JSON only."
)


def ledger(points, with_text=True):
    """The dev choices as a compact ledger.

    `with_text=False` strips situation, gist and date and keeps only the cost
    of each option and which was taken -- the revealed-preference view. The two
    renderings come from one function so the two arms provably see the same
    CHOICES and differ only in what is shown about them.
    """
    lines = []
    for i, cp in enumerate(sorted(points, key=lambda c: (c.get("date") or "", c["at_turn"])), 1):
        head = f"{i}." if not with_text else f"{i}. [{cp.get('date') or '?'}] {cp['situation']}"
        lines.append(head)
        for a in cp["alternatives"]:
            took = "TOOK" if a["action"] == cp["actual"] else "    "
            if with_text:
                lines.append(f"     {took}  {a['action']:<9} {a['gist']}  "
                             f"(gives up: {a['gives_up']})")
            else:
                lines.append(f"     {took}  cost if taken: {a['gives_up']}")
        lines.append("")
    return "\n".join(lines)


def method_prompt(person, points, method_key, corpus, channels):
    frag = M.method_rule(method_key, "seed", person) if method_key else ""
    head = (f"{person} works in {corpus} ({', '.join(sorted(channels))}). Below are "
            f"{len(points)} choices they made, earliest first. For each you see the "
            f"situation, every option that was open, what each option would have cost, "
            f"and which they took.\n\n{ledger(points)}\n")
    body = (f"{frag}\n" if frag else "")
    tail = ('Propose ONE motive for ' + person + '. Return:\n'
            '  {"motive": "<= 25 words", "because": "<= 20 words, what in the ledger '
            'forces it", "predicts": "<= 20 words, a choice it would get right that '
            'the obvious reading would get wrong"}\n\nJSON only.')
    return head + body + tail


def revealed_prompt(person, points):
    return (f"{len(points)} choices by one person, earliest first. For each option you "
            f"see only what it would have cost them, and which one they took.\n\n"
            f"{ledger(points, with_text=False)}\n"
            'Name what they consistently gave up and what they consistently protected, '
            'then state the aim that ordering implies. Return:\n'
            '  {"gave_up": "<= 15 words", "protected": "<= 15 words", '
            '"motive": "<= 25 words", "because": "<= 20 words", '
            '"predicts": "<= 20 words"}\n\nJSON only.')


def cast(corpora, out_dir=None):
    """People with enough earlier choices to fit and enough later ones to score."""
    dev = collections.defaultdict(list)
    test = collections.Counter()
    chans = collections.defaultdict(set)
    for corpus in corpora:
        for cp in load_points(corpus):
            key = (cp["corpus"], cp["person"])
            chans[key].add(cp["channel"])
            if cp["side"] == "dev":
                dev[key].append(cp)
            elif cp["side"] == "test":
                test[key] += 1
    return {k: v for k, v in dev.items()
            if len(v) >= MIN_DEV and test[k] >= MIN_TEST}, chans, test


def generate(corpora, model, method_keys=None, out=OUT):
    keys = method_keys or list(M.METHODS)
    people, chans, test = cast(corpora)
    prompts, systems, tags = [], [], []
    for (corpus, person), points in sorted(people.items()):
        for k in keys:
            prompts.append(method_prompt(person, points, k, corpus, chans[(corpus, person)]))
            systems.append(SYSTEM)
            tags.append((corpus, person, k, len(points)))
        prompts.append(revealed_prompt(person, points))
        systems.append(REVEALED_SYSTEM)
        tags.append((corpus, person, "revealed", len(points)))

    if not prompts:
        # No cast is a result, not a crash: it means no person cleared MIN_DEV
        # and MIN_TEST, which is a fact about the extraction and is reported.
        return [], {"people": 0, "generators": len(keys) + 1, "unparsed": {},
                    "why": f"no person has >= {MIN_DEV} dev and >= {MIN_TEST} test choices"}

    raws = model.batch_interact(prompts, system_prompts=systems,
                                temperature=0, max_tokens=800)

    hyps, bad = [], collections.Counter()
    for (corpus, person, key, n_dev), raw in zip(tags, raws):
        obj = extract_json(raw)
        if isinstance(obj, list):
            obj = obj[0] if obj and isinstance(obj[0], dict) else None
        if not isinstance(obj, dict) or not str(obj.get("motive") or "").strip():
            bad[key] += 1
            continue
        motive = " ".join(str(obj["motive"]).split())
        hyps.append({
            "hyp_id": f"{corpus}:{person}:{key}",
            "corpus": corpus, "person": person, "method": key,
            "family": M.family_of(key) if key in M.METHODS else "revealed",
            "motive": motive[:400],
            "because": " ".join(str(obj.get("because") or "").split())[:300],
            "predicts": " ".join(str(obj.get("predicts") or "").split())[:300],
            "gave_up": " ".join(str(obj.get("gave_up") or "").split())[:200] or None,
            "protected": " ".join(str(obj.get("protected") or "").split())[:200] or None,
            "n_dev_choices": n_dev,
            "n_test_choices": test[(corpus, person)],
            "words": len(motive.split()),
        })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for h in hyps:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    return hyps, {"people": len(people), "generators": len(keys) + 1,
                  "unparsed": dict(bad)}


def load_motives(path=OUT):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="bloomfield,boeing,oppenheimer")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--methods", default=None, help="comma list; default all twelve")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]
    model = FakeModel() if a.fake else CachedModel(a.model)
    keys = M.parse_methods(a.methods) if a.methods else None
    hyps, info = generate(corpora, model, keys, a.out)

    if info.get("why"):
        print(f"no motives generated: {info['why']}")
    by_person = collections.Counter((h["corpus"], h["person"]) for h in hyps)
    print(f"{len(hyps)} motives over {len(by_person)} people "
          f"({info['generators']} generators each)")
    for (c, p), n in by_person.most_common():
        print(f"  {c:<12} {p:<22} {n:>3}")
    if info.get("unparsed"):
        print(f"  unparsed: {info['unparsed']}")
    print(f"\n{model.stats()}")

    meta = {"at": dt.date.today().isoformat(), "git_commit": git_head(),
            "model": model.model_name, **info, **model.stats()}
    with open(a.out.replace(".jsonl", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    if a.fake:
        print("FAKE MODEL -- nothing here is evidence")


if __name__ == "__main__":
    main()
