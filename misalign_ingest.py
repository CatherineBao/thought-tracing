"""Turn a misalignment_detection eval vintage into corpora run_musing can trace.

WHAT THE EVAL ASKS. Each case is a PAIR: one question ("Are we aligned on the
material unit scale?"), one window of Slack messages, and two arms that answer
it differently. The altered arms are the same window byte for byte except for
one to four rewritten bodies -- verified here, not assumed: 232 of 232 altered
pairs carry identical message ids, senders and message counts, and differ in a
median of 2 bodies. So the pair is a near-perfect control. Anything a detector
scores differently between the arms it scored off those few sentences, and
anything it scores identically it was not reading the misalignment at all.

TWO STRATA AND TWO VARIANTS. `minimal` is the curated bearing evidence (median
24 messages); `noisy` is the same bearing messages inside ~1,000 turns of the
real channel traffic they came from, with byte-identical noise in both arms.
That is the axis this ingest exists to serve -- the same case, twice, differing
only in how much unrelated traffic it is buried in.

ONE CORPUS PER ARM, AND THE NAME IS OPAQUE. run_musing loads a corpus by name
and reads `<corpus>_profiles.json` beside it, and a profile has to be cut at
that case's `as_of`, so the arm is the smallest unit that can carry its own
prior. The name is a hash rather than `..._positive` because the run id built
from it reaches log files, trace dumps and eventually the judge's prompt: the
eval's own filenames say `minimal_evidence` against `minimal_negative` and its
paths say `negative/`, and copying either into a corpus name would put the
answer one string match away from every downstream consumer. The pair index
holds the mapping and the judge is never handed it.

WHAT IS STRIPPED. `eval_meta` is byte-identical across a pair, so it does not
leak polarity -- but it names `altered_message_ids`, `term` and a
`discriminator_frame`, which is a map straight to the sentences that carry the
case. It is dropped here and kept only in the index. `title` goes too: "acrylic,
as of 2024-07-01" names the term in a field the tracer would read.

SETS ARE THREADS. Taken from evals_to_dialogue.py, for its reason: a channel is
interleaved threads and a file is a query result, and neither is a thing anybody
read top to bottom. A thread is, and threading is what decides who saw what --
flatten it and you assert everyone read everything, which is the asymmetry a
belief tracer exists to find. Messages with no thread_id fall back to
channel-and-day, which is the nearest thing to a thread that the data supports.

SPEAKER TOKENS ARE ORG-WIDE, NOT PER ARM. A person has to carry the same token
in every case of their org, or one profile file per org cannot be reused and the
two arms of a pair cannot be compared turn by turn. Tokens come from the union of
every sender in the org, so adding a case never renames anyone already ingested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "musing")
INDEX = os.path.join(DATA_DIR, "misalign_index.json")

# Corpus names carry this prefix so a sweep can find them, and `data/` stays
# gitignored, which is where every other real-Slack corpus in this repo lives.
PREFIX = "mis"

# --------------------------------------------------------------------------
# speaker tokens
#
# short_names is the one piece of the export that must NOT be reimplemented:
# the chat labeller in tracer.py matches a line prefix with startswith, so two
# tokens where one prefixes the other silently reassign turns. musing-sms owns
# that function; import it when the checkout is there and fall back to a local
# copy of the same algorithm when it is not, so this file still runs standalone.
# --------------------------------------------------------------------------

_MUSING_SMS = os.path.expanduser("~/Documents/GitHub/musing-sms")


def _load_short_names():
    if os.path.isdir(_MUSING_SMS):
        sys.path.insert(0, _MUSING_SMS)
        try:
            from scripts.HITL.export_dialogue import short_names  # noqa
            return short_names
        except Exception:
            pass
        finally:
            if sys.path and sys.path[0] == _MUSING_SMS:
                sys.path.pop(0)
    return _fallback_short_names


def _ascii_token(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    folded = re.sub(r"[\"'‘’“”]", "", folded)
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9]+", "-", folded)).strip("-")


def _fallback_short_names(full_names):
    """Surname where it separates, whole name where it does not, counter after that.

    Only reached when the musing-sms checkout is absent. Walks the same ladder
    its short_names does -- surname, first-surname, whole name -- advancing only
    the names that still collide, then breaking any survivors with a counter.
    """
    names = sorted(set(str(n) for n in full_names))

    def forms(name):
        parts = [p for p in name.split() if p]
        out = []
        surname = _ascii_token(parts[-1]) if parts else ""
        if len(surname) > 1 and not surname.isdigit():
            out.append(surname)
            first = _ascii_token(parts[0]) if len(parts) > 1 else ""
            if first and first != surname:
                out.append(f"{first}-{surname}")
        out.append(_ascii_token(name) or "Speaker")
        return list(dict.fromkeys(out))

    ladder = {n: forms(n) for n in names}
    depth = dict.fromkeys(names, 0)
    for _ in range(4):
        token = {n: ladder[n][depth[n]] for n in names}
        clashing = {n for n in names for o in names
                    if o != n and _collides(token[n], token[o])}
        if not clashing:
            break
        if not any(depth[n] + 1 < len(ladder[n]) for n in clashing):
            break
        for n in clashing:
            if depth[n] + 1 < len(ladder[n]):
                depth[n] += 1

    assigned = {n: ladder[n][depth[n]] for n in names}
    used = Counter()
    for n in names:
        used[assigned[n]] += 1
        if used[assigned[n]] > 1:
            assigned[n] = f"{assigned[n]}-{used[assigned[n]]}"
    return assigned


def _collides(a: str, b: str) -> bool:
    """Equal, or one a prefix of the other -- both break a startswith labeller."""
    return a == b or a.startswith(b) or b.startswith(a)


short_names = _load_short_names()


def prefix_free(tokens) -> list:
    """Tokens a startswith labeller would confuse. Empty is the only safe result.

    Three ways to fail: a token repeated for two people, a token that is another
    token's prefix, and whitespace -- the labeller splits the line on the first
    colon and a token with a space in it can never match what it wrote.
    """
    tokens = list(tokens)
    dupes = {t for t, n in Counter(tokens).items() if n > 1}
    bad = set(dupes) | {t for t in tokens if " " in t or not t}
    for i, a in enumerate(tokens):
        for b in tokens[i + 1:]:
            if a != b and (a.startswith(b) or b.startswith(a)):
                bad.update((a, b))
    return sorted(bad)


def collapse(text: str) -> str:
    """One turn, one line -- the chat format is newline-delimited."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


# --------------------------------------------------------------------------
# reading the vintage
# --------------------------------------------------------------------------

_SNAPSHOT = re.compile(
    r"^(?P<variant>minimal|noisy)_(?P<kind>evidence|negative)_"
    r"(?P<pol>[an])(?P<ordinal>\d+)_(?P<suffix>.*)\.json$")


def snapshot_of(filename: str):
    """(variant, ordinal, suffix) -- the thing that identifies one PAIR.

    A case directory is not the pair. `bloomfield/blueberry_size` holds three
    snapshots of the same topic at two different `as_of` dates plus a second
    term, and the eval pairs them by the ordinal and the suffix that follow the
    polarity marker:

        minimal_evidence_a3_2024-12-24_bin_width.json
        minimal_negative_n3_2024-12-24_bin_width.json

    Keying on the directory instead collapsed those three into one and silently
    dropped two thirds of the organic set. Returns None for a name that does not
    parse, which is the signal to skip rather than guess.
    """
    m = _SNAPSHOT.match(os.path.basename(filename))
    if not m:
        return None
    return m.group("variant"), int(m.group("ordinal")), m.group("suffix")


def walk_cases(root: str) -> dict:
    """(variant, org, case, snapshot) -> {'key': answer_key, 'docs': {polarity: ...}}.

    Keyed off `answer_key.json`'s own `label` map rather than off the filename
    or the path. Both of those do say the polarity, but they say it in two
    different spellings across the branches (`minimal_negative` under
    `negative/altered/`, `negative_evidence` under `synthetic/`), and the key is
    what the eval publishes as authoritative. If a file is not named in a label
    map it is not scored -- silently guessing its arm is how a pair gets built
    backwards.

    The filename is still parsed, but only to pair two files the key has already
    labelled, never to decide what the label is.
    """
    keys = {}
    for dirpath, _, filenames in os.walk(root):
        if "answer_key.json" not in filenames:
            continue
        with open(os.path.join(dirpath, "answer_key.json"), encoding="utf-8") as fh:
            keys[dirpath] = json.load(fh)

    cases, unpaired = {}, []
    for dirpath, key in sorted(keys.items()):
        labels = key.get("label") or {}
        # One key serves both variants and every snapshot under it, and the noisy
        # files it names live under a different prefix, so each is located by name
        # under the vintage root rather than assumed to be a sibling.
        for filename, polarity in sorted(labels.items()):
            if polarity not in ("positive", "negative"):
                continue
            snap = snapshot_of(filename)
            if snap is None:
                unpaired.append(filename)
                continue
            path = _locate(root, dirpath, filename)
            if path is None:
                continue
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            meta = doc.get("eval_meta") or {}
            variant, ordinal, suffix = snap
            variant = meta.get("variant") or variant
            org = meta.get("org") or key.get("case", "?").split("/")[0]
            case = meta.get("case") or key.get("case", "?").split("/")[-1]
            slot = cases.setdefault((variant, org, case, f"{ordinal}:{suffix}"),
                                    {"key": key, "docs": {}})
            slot["docs"][polarity] = {"path": path, "doc": doc, "filename": filename,
                                      "suffix": suffix, "ordinal": ordinal}
    if unpaired:
        print(f"  note: {len(unpaired)} labelled file(s) whose name does not parse "
              f"as a snapshot, e.g. {unpaired[0]}")
    return cases


_INDEX_BY_NAME = {}


def _locate(root: str, near: str, filename: str):
    """The named file, preferring a sibling of the answer key, then anywhere below root."""
    direct = os.path.join(near, filename)
    if os.path.exists(direct):
        return direct
    if not _INDEX_BY_NAME:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                _INDEX_BY_NAME.setdefault(fn, []).append(os.path.join(dirpath, fn))
    hits = _INDEX_BY_NAME.get(filename) or []
    # A negative arm and its positive never share a filename, so a single hit is
    # the normal case; with several, take the one sharing the most path with the key.
    if not hits:
        return None
    return max(hits, key=lambda p: len(os.path.commonprefix([p, near])))


# --------------------------------------------------------------------------
# building one arm
# --------------------------------------------------------------------------

def arm_id(org: str, case: str, snapshot: str, variant: str, polarity: str, salt: str) -> str:
    """An opaque, stable name for one arm.

    Deterministic so a re-ingest reuses the same corpus files and any run
    already on disk still matches, and salted so the digest of "positive" is not
    a constant anyone could memorise across vintages.
    """
    seed = f"{salt}|{org}|{case}|{snapshot}|{variant}|{polarity}"
    return f"{PREFIX}_{hashlib.sha1(seed.encode()).hexdigest()[:12]}"


def day_of(ts: str) -> str:
    return str(ts or "")[:10]


def to_sets(doc: dict, token_of: dict, corpus: str) -> list:
    """Messages -> dialogue sets, one per thread, chronological within and between.

    Sets are ordered by their first message so that stitching them in index order
    reproduces the window's real time order, which is what --set-ids relies on.
    """
    # A message with no sender has no speaker token, so it cannot be a turn in a
    # `Name: utterance` transcript. 62 of 255,748 across the vintage, all in
    # musing_ai noisy windows. Dropped rather than given a placeholder name,
    # which would invent a person and put them in the roster. The noise is
    # byte-identical across a pair, so both arms drop the same rows -- asserted
    # by the caller rather than trusted.
    msgs = sorted((m for m in (doc.get("messages") or [])
                   if str(m.get("sender") or "").strip()),
                  key=lambda m: (str(m.get("created_at") or ""), str(m.get("message_id") or "")))
    threads = defaultdict(list)
    for m in msgs:
        tid = m.get("thread_id") or f"{m.get('channel', '')}#{day_of(m.get('created_at'))}"
        threads[tid].append(m)

    sets = []
    for tid, group in threads.items():
        turns = []
        for m in group:
            text = collapse(m.get("body"))
            if not text:
                continue
            turns.append({
                "speaker": token_of[m.get("sender")],
                "text": text,
                "sent": m.get("created_at"),
                "message_id": m.get("message_id"),
            })
        if not turns:
            continue
        speakers = sorted({t["speaker"] for t in turns})
        sets.append({
            "thread_id": tid,
            "channel": group[0].get("channel") or "",
            "date": day_of(group[0].get("created_at")),
            "speakers": speakers,
            "n_turns": len(turns),
            "turns": turns,
            "full_context": "\n".join(f"{t['speaker']}: {t['text']}" for t in turns),
            "_first": str(group[0].get("created_at") or ""),
        })

    sets.sort(key=lambda s: (s["_first"], s["thread_id"]))
    out = []
    for i, s in enumerate(sets):
        s.pop("_first")
        # `tier` is what run_musing's silver selection reads. Everything here is
        # a window that every participant could see, so nothing is gold; the
        # tier is set for shape only and selection goes through --set-ids.
        s.update({
            "set_id": f"{corpus}-{i:04d}",
            "corpus": corpus,
            "conv_id": i,
            "timeline": s["channel"],
            "title": s["channel"].lstrip("#") or "thread",
            "tier": "silver" if len(s["speakers"]) > 1 and s["n_turns"] >= 4 else "general",
            "tier_reason": "ingested from misalignment_detection",
            "source": {"thread_id": s.pop("thread_id")},
        })
        out.append(s)
    return out


def cast_of(docs: dict, token_of: dict, k: int) -> list:
    """The k people the case is about: busiest senders in the MINIMAL window.

    Not the busiest in whichever window is being traced. In the noisy arm the
    busiest speaker is whoever talks most in ~1,000 turns of unrelated traffic,
    which is a fact about the channel and not about the case -- trace them and
    the noisy arm is not a harder version of the minimal one, it is a different
    question. Pinning the cast to the minimal window keeps the same people
    traced in both variants, so the only thing that moves is how much noise they
    are buried in.

    Senders are identical across a pair (checked: 232 of 232 altered pairs), so
    this returns the same cast for both arms whichever arm it reads. It reads
    both and unions them anyway, because a cast that depended on the arm would
    be a label leak and it should not be possible to introduce one by accident.
    """
    counts = Counter()
    for polarity in sorted(docs):
        for m in docs[polarity]["doc"].get("messages") or []:
            sender = str(m.get("sender") or "").strip()
            if sender:
                counts[token_of[sender]] += 1
    # most_common breaks ties by insertion order, which depends on which arm was
    # read first. Sort the token in as well so the cast is reproducible.
    return [tok for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def attendance_gaps(sets: list) -> list:
    """Per channel: who was present, absent, then present again, and what they missed.

    The same shape run_musing's gold selection expects. These windows are short
    and mostly single-channel, so this usually yields little -- it is written
    because load_corpus requires the file, and a real computation costs one pass.
    """
    by_channel = defaultdict(list)
    for s in sets:
        by_channel[s.get("channel") or ""].append(s)
    gaps = []
    for channel, group in by_channel.items():
        group = sorted(group, key=lambda s: s["set_id"])
        for speaker in sorted({sp for s in group for sp in s["speakers"]}):
            present = [i for i, s in enumerate(group) if speaker in s["speakers"]]
            for a, b in zip(present, present[1:]):
                missed = group[a + 1:b]
                if not missed:
                    continue
                gaps.append({
                    "speaker": speaker,
                    "channel": channel,
                    "last_present": group[a]["set_id"],
                    "rejoins_at": group[b]["set_id"],
                    "missed_set_ids": [s["set_id"] for s in missed],
                    "missed_turns": sum(s["n_turns"] for s in missed),
                    "stitched_chars": sum(len(s["full_context"])
                                          for s in [group[a]] + missed + [group[b]]),
                    "target_is_prominent": True,
                    "gold_rank": sum(s["n_turns"] for s in missed),
                })
    return gaps


def write_arm(out_dir: str, corpus: str, sets: list, names: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    def dump(suffix, payload):
        path = os.path.join(out_dir, f"{corpus}_{suffix}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    dump("dialogue", sets)
    dump("gaps", attendance_gaps(sets))
    dump("speakers", names)
    return {"sets": len(sets), "turns": sum(s["n_turns"] for s in sets)}


# --------------------------------------------------------------------------

def ingest(root: str, out_dir: str, salt: str, cast_size: int, variants, limit=None) -> dict:
    cases = walk_cases(root)

    # Org-wide tokens, computed over every sender in every case of the org so a
    # person's token never depends on which cases were ingested.
    org_senders = defaultdict(set)
    for (variant, org, case, snapshot), slot in cases.items():
        for polarity in slot["docs"]:
            for m in slot["docs"][polarity]["doc"].get("messages") or []:
                if m.get("sender"):
                    org_senders[org].add(m["sender"])
    org_tokens = {}
    for org, senders in org_senders.items():
        mapping = short_names(sorted(senders))
        bad = prefix_free(list(mapping.values()))
        if bad:
            raise SystemExit(f"{org}: speaker tokens are not prefix-free: {bad}")
        org_tokens[org] = mapping

    # Cast FIRST, from the minimal window of each pair, and shared with that
    # pair's noisy arms. See cast_of: taking it from the noisy window would
    # trace whoever talks most in 1,000 turns of unrelated traffic, and the
    # noisy arm would stop being a harder version of the same case.
    casts = {}
    for (variant, org, case, snapshot), slot in sorted(cases.items()):
        if variant != "minimal" or set(slot["docs"]) != {"positive", "negative"}:
            continue
        casts[(org, case, snapshot)] = cast_of(slot["docs"], org_tokens[org], cast_size)

    index, skipped = [], []
    wanted = [(k, v) for k, v in sorted(cases.items()) if k[0] in variants]
    for (variant, org, case, snapshot), slot in wanted:
        docs = slot["docs"]
        if set(docs) != {"positive", "negative"}:
            skipped.append({"variant": variant, "org": org, "case": case,
                            "snapshot": snapshot,
                            "why": f"incomplete pair: {sorted(docs)}"})
            continue
        # The ordinal pairs them; the suffix is the check that it paired the
        # right two. a3_bin_width against n1_size_definition would be a silent
        # mismatch of topic AND as_of, and both arms would still look well-formed.
        if docs["positive"]["suffix"] != docs["negative"]["suffix"]:
            skipped.append({"variant": variant, "org": org, "case": case,
                            "snapshot": snapshot,
                            "why": f"arms disagree on snapshot suffix: "
                                   f"{docs['positive']['suffix']} vs {docs['negative']['suffix']}"})
            continue
        token_of = org_tokens[org]
        names = {token_of[s]: s for s in sorted(org_senders[org])}
        # A noisy pair whose minimal sibling was not published (13 altered cases
        # sit in channels with too little prior traffic to go the other way, and
        # the reverse happens too) falls back to its own window, and says so.
        cast = casts.get((org, case, snapshot))
        cast_source = "minimal"
        if cast is None:
            cast, cast_source = cast_of(docs, token_of, cast_size), variant
        key = slot["key"]
        meta = docs["positive"]["doc"].get("eval_meta") or {}

        entry = {
            "variant": variant, "org": org, "case": case, "snapshot": snapshot,
            "pair_id": f"{org}/{case}#{snapshot}",
            "question": docs["positive"]["doc"].get("question"),
            "as_of": meta.get("as_of") or key.get("as_of"),
            "branch": meta.get("branch") or key.get("method"),
            "cast": cast,
            "cast_source": cast_source,
            "arms": {},
            # Held here and nowhere the tracer or the judge can reach.
            "gold": {
                "term": key.get("term"),
                "reading_a": key.get("reading_a"),
                "reading_b": key.get("reading_b"),
                "altered_message_ids": key.get("altered_message_ids") or [],
                "real_message_share": key.get("real_message_share"),
                "quality_score": key.get("quality_score"),
            },
        }
        # The two arms must agree on the question, or the pair is not a control.
        if docs["positive"]["doc"].get("question") != docs["negative"]["doc"].get("question"):
            skipped.append({"variant": variant, "org": org, "case": case,
                            "snapshot": snapshot,
                            "why": "arms ask different questions"})
            continue

        for polarity in ("positive", "negative"):
            corpus = arm_id(org, case, snapshot, variant, polarity, salt)
            sets = to_sets(docs[polarity]["doc"], token_of, corpus)
            if not sets:
                skipped.append({"variant": variant, "org": org, "case": case,
                                "snapshot": snapshot,
                                "why": f"{polarity} arm has no usable turns"})
                break
            counts = write_arm(out_dir, corpus, sets, names)
            entry["arms"][polarity] = {
                "corpus": corpus, "label": polarity,
                "source": os.path.relpath(docs[polarity]["path"], root), **counts,
            }
        else:
            # How equal the arms have to be depends on how they were built, and
            # the two branches were built differently.
            #
            #   altered/  the negative IS the positive's window with a few bodies
            #             rewritten, so turn count AND thread structure match
            #             exactly. Anything else means a row was dropped from one
            #             side only and length alone would answer the question.
            #
            #   <org>/    the negative is a DIFFERENT clean window from the same
            #             channels, length-matched but independently threaded.
            #             Requiring equal thread counts here rejected all ten
            #             organic pairs -- the whole point of the branch.
            pos, neg = entry["arms"]["positive"], entry["arms"]["negative"]
            same_shape = pos["turns"] == neg["turns"]
            if entry["branch"] == "altered":
                same_shape = same_shape and pos["sets"] == neg["sets"]
            if not same_shape:
                skipped.append({"variant": variant, "org": org, "case": case,
                                "snapshot": snapshot,
                                "why": f"arms differ in shape: {pos['turns']}t/{pos['sets']}s "
                                       f"vs {neg['turns']}t/{neg['sets']}s"})
                continue
            index.append(entry)
        if limit and len(index) >= limit:
            break

    payload = {
        "_README": ("Index for the misalignment_detection corpora in this directory. "
                    "POLARITY LIVES ONLY HERE. Corpus names are salted digests so no "
                    "run id, log path or trace dump spells the answer; scoring joins "
                    "back through `arms`. Do not hand this file to a detector."),
        "root": os.path.abspath(root),
        "salt": salt,
        "cast_size": cast_size,
        "cases": index,
        "skipped": skipped,
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local copy of the vintage prefix")
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--salt", default="altered_real_v1",
                    help="changes every opaque corpus name; keep it fixed to reuse runs")
    ap.add_argument("--cast-size", type=int, default=3,
                    help="people traced per case, busiest-first in the MINIMAL window")
    ap.add_argument("--variants", default="minimal,noisy")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    payload = ingest(a.root, a.out_dir, a.salt, a.cast_size,
                     set(a.variants.split(",")), a.limit)
    os.makedirs(os.path.dirname(a.index), exist_ok=True)
    with open(a.index, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    cases = payload["cases"]
    by = Counter((c["variant"], c["branch"]) for c in cases)
    print(f"{len(cases)} pairs -> {a.out_dir}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:<8} {str(k[1]):<14} {v}")
    turns = sum(arm["turns"] for c in cases for arm in c["arms"].values())
    print(f"  {turns:,} turns written across {2 * len(cases)} corpora")
    if payload["skipped"]:
        print(f"  skipped {len(payload['skipped'])}:")
        for s in payload["skipped"][:10]:
            print(f"    {s['variant']}/{s['org']}/{s['case']}: {s['why']}")
    print(f"index: {a.index}")


if __name__ == "__main__":
    main()
