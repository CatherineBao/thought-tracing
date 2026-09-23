"""Freeze the dev/test split, and prove the two sides share no text.

    python make_splits.py            # writes splits.json
    python make_splits.py --check    # re-derive and diff against the frozen file

WHY A SPLIT AT ALL. PRECHECKS records three interventions in a row that were
each briefly believed to work, every one of them read off the same data they
were tuned on. A frozen split is the cheapest structural defence: nothing tuned
on dev is ever re-tuned after looking at test, and `--check` makes drift in the
file a failing command rather than a thing somebody notices later.

TWO SECTIONS, TWO RULES, BECAUSE THEY CARRY DIFFERENT RISK.

  corpora/  data/musing sets, cut into CONTIGUOUS DATE SPANS. These runs carry
            no misalignment label -- churn, rank movement, tie blocks, the
            swapped-person control -- so the only thing a cut has to buy is
            that a number tuned on one stretch of a channel is checked on a
            different stretch. set_ids are date-ordered within a corpus, so a
            contiguous date span is a contiguous id range and a reader can
            verify the cut by eye.

  eval/     altered_real_v1, assigned WHOLE ORG AT A TIME. A finer cut was
            measured and rejected: the noisy variant of one bloomfield case
            reaches 204 of the corpus's 402 sets, and noisy exyn/lead_time and
            noisy exyn/shipping_terms share 358 messages. Any within-org cut is
            straddled by the windows themselves, so org is the smallest unit
            that can be disjoint. The measured numbers are written into
            `checks` rather than asserted here.

WHICH ORG GOES WHERE IS NOT A COIN FLIP. bloomfield is the one eval org this
repo has already worked on -- 140 of 248 logged runs, five detector designs,
every finding in PRECHECKS. Prior exposure on the TEST side buys optimism, so
bloomfield is dev. agzen and exyn both have corpora in data/musing and NEITHER
has a single logged run against it, so either could be test; exyn is test
because it carries two real cases to agzen's one, and the test side is the one
short of real cases.

WHAT THE SPLIT CANNOT FIX. There are ten organic pairs in the whole vintage,
four of them on the test side. Four pairs cannot establish anything on their
own and are pre-registered as a confirmatory read, not a measurement; the
powered test is the 38 altered test cases. Said here so the number is on the
record before anybody reports it.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MUSING = os.path.join(HERE, "data", "musing")
VINTAGE = os.path.join(HERE, "data", "misalign_evals", "altered_real_v1")
OUT = os.path.join(HERE, "splits.json")

# Corpora cut by contiguous date span, and the key each one is blocked on
# before cutting. bloomfield is blocked on `title` because its two threads are
# two different disputes that happen to share a corpus; the other two are cut
# whole, because neither has a thread structure a span could follow.
CORPORA = {
    "bloomfield": "title",
    "boeing": None,
    "oppenheimer": None,
}

# Org -> side. See the module docstring for why each one sits where it does.
ORG_SIDE = {
    "bloomfield": "dev",   # exposed: 140 logged runs, all of PRECHECKS
    "agzen": "dev",        # unexposed, but dev needs volume and one real case
    "aigen": "dev",        # unexposed, altered only
    "exyn": "test",        # unexposed, two real cases
    "flovision": "test",
    "flovision_solutions": "test",  # same traffic as flovision; must share a side
    "musing_ai": "test",
}


# --------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------

def load_corpus(name):
    with open(os.path.join(MUSING, f"{name}_dialogue.json"), encoding="utf-8") as fh:
        return json.load(fh)


def cut_block(sets):
    """Split one date-ordered block in two at a DATE boundary.

    The cut is the earliest date that puts at least half the sets behind it, so
    no single day is ever split across the two sides -- a day is the finest
    grain the corpus timestamps a set at, and splitting one would put two halves
    of one conversation on opposite sides of the wall.

    Returns (dev_ids, test_ids, cut_date, undated_ids). Undated sets are NOT
    assigned: boeing has four, and there is no honest way to place a set on a
    timeline it is not on. They are reported instead.
    """
    dated = sorted([s for s in sets if s.get("date")], key=lambda s: (s["date"], s["set_id"]))
    undated = sorted(s["set_id"] for s in sets if not s.get("date"))
    if not dated:
        # oppenheimer: narrative order IS the only order. Cut on cumulative
        # turns rather than set count, so the two halves carry comparable
        # material -- its sets run from 1 turn to 310.
        ordered = sorted(sets, key=lambda s: s["set_id"])
        total = sum(s["n_turns"] for s in ordered)
        run, cut = 0, len(ordered)
        for i, s in enumerate(ordered):
            run += s["n_turns"]
            if run * 2 >= total:
                cut = i + 1
                break
        # Narrative order places every set, so nothing is left unassigned --
        # `undated` is the whole block here and would otherwise be reported as
        # 55 sets this function had just placed.
        return ([s["set_id"] for s in ordered[:cut]],
                [s["set_id"] for s in ordered[cut:]], None, [])

    half = len(dated) / 2.0
    cut_date = dated[-1]["date"]
    for i, s in enumerate(dated):
        if i + 1 >= half and (i + 1 == len(dated) or dated[i + 1]["date"] != s["date"]):
            cut_date = dated[i + 1]["date"] if i + 1 < len(dated) else s["date"]
            break
    dev = [s["set_id"] for s in dated if s["date"] < cut_date]
    test = [s["set_id"] for s in dated if s["date"] >= cut_date]
    return dev, test, cut_date, undated


def corpus_section():
    out = {}
    for name, block_key in CORPORA.items():
        sets = load_corpus(name)
        groups = collections.OrderedDict()
        if block_key:
            for s in sets:
                groups.setdefault(s[block_key], []).append(s)
        else:
            groups[name] = sets

        blocks = []
        for gname, gsets in groups.items():
            dev, test, cut, undated = cut_block(gsets)
            ids = sorted(s["set_id"] for s in gsets)
            blocks.append({
                "block": gname,
                "set_id_range": [ids[0], ids[-1]],
                "cut": cut,
                "cut_rule": ("earliest date with >=50% of sets before it"
                             if cut else "cumulative-turn midpoint of narrative order"),
                "dev": dev,
                "test": test,
                "unassigned": undated,
                "unassigned_reason": "no date on the set; cannot be placed on a timeline"
                                     if undated else None,
                "counts": {"dev": len(dev), "test": len(test), "unassigned": len(undated)},
            })
        out[name] = {
            "unit": "set_id",
            "order": "date" if any(b["cut"] for b in blocks) else "set_id (narrative)",
            "blocked_on": block_key,
            "scope": ("UNLABELLED mechanics only -- churn, rank movement, tie blocks, "
                      "scorer probes, swapped-person controls. These sets carry no "
                      "misalignment label, so text shared with an eval case cannot "
                      "inflate a detection score. Do NOT score a detector here."),
            "blocks": blocks,
        }
    return out


# --------------------------------------------------------------------------
# eval vintage
# --------------------------------------------------------------------------

def case_dirs(variant="minimal"):
    """Every case, as (branch, org, case, path-to-its-minimal-dir).

    Only `minimal` carries answer_key.json -- noisy reuses it, and its label map
    already names both variants' files -- so the case list is enumerated there
    and the noisy paths are built from it. Walking noisy for keys finds nothing
    and silently drops every noisy arm, which is exactly what it did.
    """
    root = os.path.join(VINTAGE, variant)
    for key in sorted(glob.glob(os.path.join(root, "**", "answer_key.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(key), root).split(os.sep)
        branch = "altered" if rel[0] == "altered" else "real"
        org, case = (rel[1], rel[2]) if branch == "altered" else (rel[0], rel[1])
        yield branch, org, case, os.path.dirname(key)


def arms(case_path, variant, org, case, branch, pos_dir=None):
    """The two arms of a case: every evidence file, and its negative twin.

    Read off the answer key's own `label` map rather than reconstructed from
    filenames, so a rename upstream shows up as a missing file here instead of
    as a silently mislabelled arm.
    """
    key = json.load(open(os.path.join(case_path, "answer_key.json"), encoding="utf-8"))
    pos_dir = pos_dir or case_path
    neg_dir = os.path.join(VINTAGE, variant, "negative",
                           *( ["altered", org, case] if branch == "altered" else [org, case]))
    out = []
    for fname, label in key["label"].items():
        if not fname.startswith(variant + "_"):
            continue
        path = os.path.join(pos_dir if label == "positive" else neg_dir, fname)
        out.append({"file": os.path.relpath(path, HERE), "label": label,
                    "present": os.path.exists(path)})
    return key, sorted(out, key=lambda a: a["file"])


def eval_section():
    cases = {}
    for branch, org, case, minimal_dir in case_dirs("minimal"):
        cid = f"{branch}/{org}/{case}"
        rec = None
        for variant in ("minimal", "noisy"):
            pos_dir = (minimal_dir if variant == "minimal"
                       else os.path.join(VINTAGE, variant,
                                         *(["altered", org, case] if branch == "altered"
                                           else [org, case])))
            key, arm = arms(minimal_dir, variant, org, case, branch, pos_dir)
            if rec is None:
                rec = cases.setdefault(cid, {
                    "branch": branch, "org": org, "case": case,
                    "side": ORG_SIDE[org],
                    "method": key.get("method"),
                    "question": key.get("question"),
                    "as_of": key.get("as_of"),
                    "term": key.get("term"),
                    "real_text_share": key.get("real_message_share"),
                    "variants": {},
                })
            rec["variants"][variant] = arm
    return cases


# --------------------------------------------------------------------------
# the checks that justify the org-level rule
# --------------------------------------------------------------------------

# Some messages carry message_id "". They are different messages sharing one
# empty key, and counting that key as an id made every org that has one look
# like it shared traffic with every other -- which is how the first run of this
# check reported a crossing. Blank ids are excluded and counted instead.
BLANK_IDS = collections.Counter()


def message_ids(paths):
    out = set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for m in json.load(fh).get("messages", []):
                mid = m.get("message_id")
                if mid:
                    out.add(mid)
                else:
                    BLANK_IDS[os.path.relpath(p, HERE)] += 1
    return out


def crossing_check(cases):
    """Does any message appear on both sides of the wall? Measured, not assumed.

    This is the whole justification for cutting by org, so it is recomputed
    every time the split is written and the number goes into the file.
    """
    side_ids = {"dev": set(), "test": set()}
    per_case = {}
    for cid, rec in cases.items():
        ids = set()
        for variant, arm in rec["variants"].items():
            ids |= message_ids([os.path.join(HERE, a["file"]) for a in arm if a["present"]])
        per_case[cid] = ids
        side_ids[rec["side"]] |= ids
    shared = side_ids["dev"] & side_ids["test"]

    # and the within-org overlap that forced the rule
    pairs = []
    orgs = collections.defaultdict(list)
    for cid, rec in cases.items():
        orgs[rec["org"]].append(cid)
    for org, cids in sorted(orgs.items()):
        for i, a in enumerate(sorted(cids)):
            for b in sorted(cids)[i + 1:]:
                ov = len(per_case[a] & per_case[b])
                if ov:
                    smaller = min(len(per_case[a]), len(per_case[b])) or 1
                    pairs.append({"a": a, "b": b, "shared_messages": ov,
                                  "share_of_smaller": round(ov / smaller, 4)})
    pairs.sort(key=lambda p: -p["shared_messages"])
    return {
        "dev_messages": len(side_ids["dev"]),
        "test_messages": len(side_ids["test"]),
        "shared_across_sides": len(shared),
        "within_org_overlapping_case_pairs": len(pairs),
        "worst_within_org_overlaps": pairs[:8],
        "messages_with_blank_id": sum(BLANK_IDS.values()),
        "files_with_blank_ids": len(BLANK_IDS),
        "note": ("shared_across_sides MUST be 0. Messages with a blank message_id are "
                 "excluded from it -- they are distinct messages sharing one empty "
                 "key, and counting it made every org holding one look shared. "
                 "within_org overlaps are why the eval split is cut by org: these "
                 "case pairs could not be put on opposite sides without sharing text."),
    }


def corpus_eval_overlap(cases):
    """How far the data/musing corpora and the eval windows are the same text.

    They are the same messages: every minimal bloomfield evidence id is also a
    corpus turn. This is recorded so the `scope` field on the corpora section is
    a measured claim rather than a caution.
    """
    corpus_ids = {}
    for name in CORPORA:
        for s in load_corpus(name):
            for t in s["turns"]:
                mid = t.get("message_id")
                if mid:
                    corpus_ids.setdefault(name, {}).setdefault(mid, set()).add(s["set_id"])
    rows = []
    for cid, rec in sorted(cases.items()):
        if rec["branch"] != "real":
            continue
        for variant, arm in sorted(rec["variants"].items()):
            ids = message_ids([os.path.join(HERE, a["file"]) for a in arm if a["present"]])
            for cname, table in corpus_ids.items():
                hit = ids & set(table)
                if not hit:
                    continue
                touched = sorted({sid for m in hit for sid in table[m]})
                rows.append({"case": cid, "variant": variant, "corpus": cname,
                             "eval_messages": len(ids), "also_in_corpus": len(hit),
                             "corpus_sets_touched": len(touched)})
    return rows


# --------------------------------------------------------------------------

def build():
    corpora = corpus_section()
    cases = eval_section()
    side_counts = collections.Counter(
        (rec["branch"], rec["side"]) for rec in cases.values())
    pair_counts = collections.Counter()
    for rec in cases.values():
        n = sum(1 for a in rec["variants"].get("minimal", []) if a["label"] == "positive")
        pair_counts[(rec["branch"], rec["side"])] += n

    return {
        "_README": (
            "Frozen dev/test split. Nothing tuned on dev is re-tuned after looking "
            "at test. Two sections with two rules and two scopes -- read `scope` on "
            "each before using it. Regenerate and verify with `python make_splits.py "
            "--check`, which fails if this file no longer matches the data."),
        "frozen_at": dt.date.today().isoformat(),
        "git_commit": git_head(),
        "generator": "make_splits.py",
        "corpora": corpora,
        "eval": {
            "vintage": "altered_real_v1",
            "root": os.path.relpath(VINTAGE, HERE),
            "unit": "org",
            "unit_reason": ("a noisy window reaches across a whole org's traffic, so no "
                            "within-org cut is disjoint; see checks.crossing"),
            "org_side": ORG_SIDE,
            "side_reason": {
                "bloomfield": "dev -- the only eval org with prior exposure (140 logged runs, all of PRECHECKS)",
                "agzen": "dev -- unexposed; dev needs volume and one real case",
                "aigen": "dev -- unexposed, altered only",
                "exyn": "test -- unexposed, and carries two real cases where agzen carries one",
                "flovision": "test -- unexposed",
                "flovision_solutions": "test -- same traffic as flovision, must share its side",
                "musing_ai": "test -- unexposed",
            },
            "counts": {
                "cases": {f"{b}/{s}": n for (b, s), n in sorted(side_counts.items())},
                "pairs_per_variant": {f"{b}/{s}": n for (b, s), n in sorted(pair_counts.items())},
            },
            "power_note": (
                "4 real pairs on the test side cannot establish anything alone and are "
                "pre-registered as a confirmatory read. The powered test is the 38 "
                "altered test cases, x2 variants."),
            "cases": cases,
        },
        "checks": {},
    }


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and exit non-zero if the frozen file disagrees")
    a = ap.parse_args()

    payload = build()
    payload["checks"] = {
        "crossing": crossing_check(payload["eval"]["cases"]),
        "corpus_eval_overlap": corpus_eval_overlap(payload["eval"]["cases"]),
    }

    crossing = payload["checks"]["crossing"]
    if crossing["shared_across_sides"]:
        raise SystemExit(f"ABORT: {crossing['shared_across_sides']} messages appear on "
                         f"both sides of the split. The split is not disjoint.")

    if a.check:
        if not os.path.exists(a.out):
            raise SystemExit(f"{a.out} does not exist; run without --check first")
        frozen = json.load(open(a.out, encoding="utf-8"))
        drift = []
        for key in ("corpora", "eval"):
            if json.dumps(frozen.get(key), sort_keys=True) != json.dumps(payload[key], sort_keys=True):
                drift.append(key)
        if drift:
            raise SystemExit(f"SPLIT DRIFT in {', '.join(drift)} -- the frozen split no "
                             f"longer matches the data. Do not silently regenerate.")
        print(f"split matches {a.out}")
        return

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    print(f"wrote {a.out}")
    for name, sec in payload["corpora"].items():
        for b in sec["blocks"]:
            c = b["counts"]
            print(f"  {name}/{b['block']:<22} dev={c['dev']:>4} test={c['test']:>4} "
                  f"unassigned={c['unassigned']:>2}  cut={b['cut']}")
    ev = payload["eval"]["counts"]
    print(f"  eval cases:  {ev['cases']}")
    print(f"  eval pairs:  {ev['pairs_per_variant']}  (per variant; x2 variants)")
    print(f"  crossing:    dev={crossing['dev_messages']} test={crossing['test_messages']} "
          f"shared={crossing['shared_across_sides']}")


if __name__ == "__main__":
    main()
