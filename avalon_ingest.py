"""Avalon-NLU -> a `data/musing` corpus, two choice streams, and an answer key.

WHY AVALON. A role is assigned at setup, never stated in the transcript, and
drives everything the player does. That is the strongest hidden-motive label in
the project -- CaSiNo's priorities are task-assigned and usually spoken aloud,
Diplomacy's are structural, atla is fiction and may be memorised.

TWO STREAMS, AND ONLY ONE IS SCORED (PREREG).

  include/exclude   SCORED. One point per (leader, other player) at each
                    proposal: did the leader put them on the party. Base rate is
                    favourable by construction -- parties are 2 to 4 of 6 --
                    and it clears the gate at majority 0.520.

  party vote        EVIDENCE ONLY. Every player votes on every proposal, roughly
                    7-10 choices per player per game, and it is the only thing
                    giving Avalon a real per-player sequence. It FAILS the gate
                    as a scoring target at 0.708 yes. The gate protects the
                    reported metric; it does not forbid the filter from taking
                    evidence. Weights update on votes; the headline log-score
                    does not read them.

A PROPOSAL IS ONE OBSERVATION, NOT FIVE. The pairs in a proposal are tied --
the leader must pick exactly k players -- so they are revealed together
(`batch_id`) and contribute a tempered `(prod p)^(1/m)` likelihood. Without that,
148 proposals would be counted as 740 independent decisions.

SELF-PAIRS ARE EXCLUDED. Measured: leaders include themselves in 133/148 = 0.899
of proposals, so those points are near-trivial.

PORTFOLIOS RESET PER GAME. Roles are reassigned between games, so a portfolio
carried across one would carry a previous game's role into the next. The game is
the span.

THE SAME-PERSON AXIS DOES NOT EXIST HERE. `users` carries only index/name/role
and `name` is `player-1`..`player-6` in every one of the 20 games -- positional,
not a person. Whatever the paper says about participant count, the released data
has no mapping, so standing traits cannot be separated from the assigned role.

ROLE-REVEALING FIELDS GO IN THE ANSWER KEY. Per-message persuasion and deception
self-labels and the recorded beliefs about other players' roles both give the
game away -- a message tagged `deception` nearly announces an evil player. None
of it may reach a prompt, and every prefix stops at its own proposal, so the
assassination phase and post-game chat (which discuss roles openly) are never in
one.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "avalon")
MUSING = os.path.join(HERE, "data", "musing")
CP_DIR = os.path.join(HERE, "musing_out", "choice_points")
KEY_DIR = os.path.join(HERE, "data", "avalon", "answer_keys")

PROP_RE = re.compile(r"^(player-\d+) proposed a party:\s*(.+)$", re.I)
VOTE_RE = re.compile(r"^party vote outcome:\s*(.+)$", re.I)
OUTCOME_RE = re.compile(r"^(vote succeeded|vote failed|quest succeeded|quest failed)", re.I)

INCLUDE_LABELS = ("include", "exclude")
VOTE_LABELS = ("yes", "no")
EPOCH_YEAR = 2023


def short(name):
    """`player-4` -> `P4`. Single token, unique, prefix-free."""
    m = re.match(r"player-(\d+)", name or "")
    return f"P{m.group(1)}" if m else (name or "?")


def ordered_messages(game):
    """Messages in true order. The dict keys are numeric and ARE the order --
    (quest, turn) is not sufficient, since several messages share a turn."""
    return [game["messages"][k] for k in sorted(game["messages"], key=lambda x: int(x))]


def is_endgame(text):
    t = (text or "").lower()
    return "assassin" in t or "won for now" in t or "forces of good" in t


def convert_game(game, gid):
    msgs = ordered_messages(game)
    roles = {u["name"]: u["role"] for u in game["users"].values()}
    players = sorted(roles, key=lambda p: int(p.split("-")[1]))
    turns, points = [], []
    rejects = collections.Counter()

    # visible transcript: chat plus the system lines that are legitimate game
    # state. Endgame lines are dropped outright rather than merely excluded from
    # prefixes, so nothing downstream can reach them by accident.
    for m in msgs:
        if is_endgame(m.get("msg")):
            continue
        who = "GAME" if m["player"] == "system" else short(m["player"])
        turns.append({"speaker": who, "text": " ".join((m.get("msg") or "").split())})

    prop_no = 0
    for i, m in enumerate(msgs):
        if m["player"] != "system":
            continue
        text = (m.get("msg") or "").strip()
        mo = PROP_RE.match(text)
        if not mo:
            continue
        leader_full, party_full = mo.group(1), [x.strip() for x in mo.group(2).split(",")]
        if leader_full not in roles:
            rejects["proposal by an unknown player"] += 1
            continue
        prop_no += 1
        leader = short(leader_full)
        party = {short(p) for p in party_full if p in roles}
        batch_id = f"avalon-{gid}-prop{prop_no:02d}"

        # PREFIX: everything strictly before the proposal, endgame stripped.
        prefix = [{"speaker": ("GAME" if x["player"] == "system" else short(x["player"])),
                   "text": " ".join((x.get("msg") or "").split())}
                  for x in msgs[:i] if not is_endgame(x.get("msg"))]

        candidates = [short(p) for p in players if short(p) != leader]
        for cand in candidates:
            points.append({
                "cp_id": f"{batch_id}-{cand}", "corpus": "avalon",
                "set_id": f"avalon-{gid}", "person": leader,
                "person_uid": f"avalon-{gid}:{leader}",
                "kind": "include", "scored": True,
                "batch_id": batch_id, "batch_size": len(candidates),
                "at_turn": i, "subject": cand,
                "situation": f"{leader} is choosing a party of {len(party)}. "
                             f"Does {leader} put {cand} on it?",
                "alternatives": [{"action": l, "gist": f"{leader} {l}s {cand}",
                                  "gives_up": ""} for l in INCLUDE_LABELS],
                "n_alternatives": len(INCLUDE_LABELS),
                "actual": "include" if cand in party else "exclude",
                "prefix_turns": prefix, "date": f"{EPOCH_YEAR}-01-{(gid % 28) + 1:02d}",
            })

        # the vote that follows this proposal -- EVIDENCE ONLY, never scored
        for j in range(i + 1, len(msgs)):
            nxt = msgs[j]
            if nxt["player"] != "system":
                continue
            nt = (nxt.get("msg") or "").strip()
            if PROP_RE.match(nt):
                break                      # a re-proposal: this one never went to a vote
            mv = VOTE_RE.match(nt)
            if not mv:
                continue
            vprefix = [{"speaker": ("GAME" if x["player"] == "system" else short(x["player"])),
                        "text": " ".join((x.get("msg") or "").split())}
                       for x in msgs[:j] if not is_endgame(x.get("msg"))]
            for part in mv.group(1).split(","):
                if ":" not in part:
                    continue
                who, v = [x.strip() for x in part.split(":", 1)]
                if who not in roles or v.lower() not in VOTE_LABELS:
                    continue
                voter = short(who)
                points.append({
                    "cp_id": f"{batch_id}-vote-{voter}", "corpus": "avalon",
                    "set_id": f"avalon-{gid}", "person": voter,
                    "person_uid": f"avalon-{gid}:{voter}",
                    "kind": "party_vote", "scored": False,
                    "batch_id": f"{batch_id}-vote", "batch_size": 1,
                    "at_turn": j, "subject": None,
                    "situation": f"{leader} proposed a party of {len(party)}: "
                                 f"{', '.join(sorted(party))}. Does {voter} approve?",
                    "alternatives": [{"action": l, "gist": f"{voter} votes {l}",
                                      "gives_up": ""} for l in VOTE_LABELS],
                    "n_alternatives": len(VOTE_LABELS),
                    "actual": v.lower(), "prefix_turns": vprefix,
                    "date": f"{EPOCH_YEAR}-01-{(gid % 28) + 1:02d}",
                })
            break

    st_set = {
        "set_id": f"avalon-{gid}", "corpus": "avalon", "conv_id": gid,
        "channel": "table", "timeline": "table",
        "title": f"Avalon game {gid}", "date": f"{EPOCH_YEAR}-01-{(gid % 28) + 1:02d}",
        "tier": "silver", "speakers": [short(p) for p in players] + ["GAME"],
        "n_turns": len(turns),
        "full_context": "\n".join(f"{t['speaker']}: {t['text']}" for t in turns),
        "turns": turns,
    }
    # ANSWER KEY -- roles, beliefs and the self-labels, none of which may be prompted
    key = {
        "schema": "avalon.answer_key/1", "set_id": f"avalon-{gid}",
        "roles": {short(p): r for p, r in roles.items()},
        "beliefs": list((game.get("beliefs") or {}).values()),
        "message_labels": list((game.get("persuasion") or {}).values()),
    }
    return st_set, points, key, rejects


def convert(paths):
    sets, points, keys = [], [], {}
    rejects = collections.Counter()
    for gid, path in enumerate(sorted(paths)):
        game = json.load(open(path, encoding="utf-8"))
        s, p, k, rj = convert_game(game, gid)
        sets.append(s)
        points.extend(p)
        keys[s["set_id"]] = k
        rejects.update(rj)
    scored = [p for p in points if p["scored"]]
    votes = [p for p in points if not p["scored"]]
    per_player = collections.Counter(p["person_uid"] for p in votes)
    report = {
        "games": len(sets), "total_points": len(points),
        "scored_include_points": len(scored), "unscored_vote_points": len(votes),
        "proposals": len({p["batch_id"] for p in scored}),
        "mean_batch_size": round(st.mean([p["batch_size"] for p in scored]), 2) if scored else 0,
        "include_labels": dict(collections.Counter(p["actual"] for p in scored).most_common()),
        "vote_labels": dict(collections.Counter(p["actual"] for p in votes).most_common()),
        "votes_per_player_game": {
            "mean": round(st.mean(list(per_player.values())), 2),
            "median": st.median(list(per_player.values())),
            "max": max(per_player.values()),
        } if per_player else None,
        "rejects": dict(rejects),
    }
    return sets, points, keys, report


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return None


def write(sets, points, keys):
    os.makedirs(MUSING, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)
    os.makedirs(KEY_DIR, exist_ok=True)
    json.dump(sets, open(os.path.join(MUSING, "avalon_dialogue.json"), "w", encoding="utf-8"))
    speakers = {"GAME": "Game narrator"}
    speakers.update({f"P{i}": f"Player {i}" for i in range(1, 7)})
    json.dump(speakers, open(os.path.join(MUSING, "avalon_speakers.json"), "w",
                             encoding="utf-8"), indent=1)
    json.dump([], open(os.path.join(MUSING, "avalon_gaps.json"), "w", encoding="utf-8"))
    with open(os.path.join(CP_DIR, "avalon.jsonl"), "w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump({"git_commit": git_head(), "keys": keys},
              open(os.path.join(KEY_DIR, "avalon_answer_key.json"), "w", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    paths = glob.glob(os.path.join(RAW, "*.json"))
    if not paths:
        print(f"no games in {RAW}", file=sys.stderr)
        return 2
    sets, points, keys, report = convert(paths)
    if not a.dry_run:
        write(sets, points, keys)
    if a.report or a.dry_run:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
