"""Diplomacy -> a `data/musing` corpus, signed choice points, and an answer key.

THE CHOICE UNIT IS ONE POINT PER (PLAYER, MOVEMENT PHASE), and the label is
SIGNED: `attack:X`, `support:X`, or `NONE`.

Three framings were measured and all three are on the record (PREREG):

  (player, power) pair        neither 0.816 messaged / 0.759 in contact   FAIL
  primary target, unsigned    majority 0.3478 -- but `Germany` then means
                              BOTH attacking and supporting Germany, which are
                              opposite motives under one label
  primary target, SIGNED      majority 0.3478, entropy 0.8075, loo 0.3346  PASS

The pair unit manufactures a `neither` sink the way `HOLD` and `Accept` did: a
power has a handful of units and six neighbours and acts on two or three. The
signed unit was adopted on semantic grounds and the gate formula independently
agrees -- it fails the unsigned attack-only label at 0.4395 and passes signed.

ALIGNMENT IS (game_id, year, season), AND THE SEASON FIELD EXISTS. Messages
carry `seasons` alongside `years`, and moves files are keyed
`DiplomacyGame{N}_{year}_{season}`, so bucketing by year alone -- which would put
Fall messages before Spring orders and leak the Spring outcome -- is not needed
and is not done. Winter is an adjustment phase (BUILD/DISBAND only) and carries
no attack or support, so movement phases only.

SIMULTANEITY IS ACROSS PLAYERS, NOT JUST WITHIN ONE. All seven powers order at
once, so a prefix for phase t must exclude EVERY player's phase-t orders and the
adjudicated results of phase t -- not merely the subject's own. A nonce test over
a single actor never reaches this; the prefix is built from strictly-earlier
phases and `test_hindsight` checks the second-actor case.

FEASIBILITY (clause C2) USES THE REPAIRED MAP. Offering "attack Turkey" to a
player whose units are nowhere near Turkey is not a feasible option. The
canonical source (the `diplomacy` package) is AGPL-3.0 and is not vendored; the
adjacency is rebuilt from move geometry -- a MOVE A->B is an edge, a SUPPORT at S
for X->Y gives S-Y and X-Y, convoys give none -- taking mean degree from 12.6 to
7.5 against a real board's ~4.5, with seven spot checks passing. It stays
somewhat permissive, which INFLATES k, and since the gate limit falls as k rises
that is the safe direction.

TIE-BREAKING IS VISIBLE, NOT ALPHABETICAL. 16.4% of signed labels tie on order
count, and an alphabetical rule is label noise nobody could predict. The
registered order is: the target whose targeted province is a SUPPLY CENTRE, then
the target against which more units are committed, then alphabetical as a last
resort with the residual rate reported.
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
RAW = os.path.join(HERE, "data", "diplomacy")
MUSING = os.path.join(HERE, "data", "musing")
CP_DIR = os.path.join(HERE, "musing_out", "choice_points")
KEY_DIR = os.path.join(HERE, "data", "diplomacy", "answer_keys")

MOVEMENT = ("Spring", "Fall")
SEASON_ORDER = {"Spring": 0, "Fall": 1, "Winter": 2}

# The 34 standard supply centres. Public game rules, not derived from any
# package -- 22 home centres plus 12 neutrals. Used only for the tie-break.
SUPPLY_CENTRES = {
    "VIE", "BUD", "TRI", "LON", "EDI", "LVP", "PAR", "BRE", "MAR",
    "BER", "MUN", "KIE", "ROM", "VEN", "NAP", "MOS", "STP", "WAR", "SEV",
    "CON", "ANK", "SMY",
    "NWY", "SWE", "DEN", "HOL", "BEL", "SPA", "POR", "TUN", "SER", "RUM",
    "BUL", "GRE",
}

PHASE_RE = re.compile(r"DiplomacyGame(\d+)_(\d{4})_(\w+)")


# -- the map ---------------------------------------------------------------

def build_adjacency(phases):
    """Move geometry only. See the module docstring for why `from` is excluded
    on the supporter side: it is the SUPPORTED unit's province and need not be
    adjacent to the supporter."""
    adj = collections.defaultdict(set)

    def add(a, b):
        a, b = (a or "").upper(), (b or "").upper()
        if a and b and a != b:
            adj[a].add(b)
            adj[b].add(a)

    for orders in phases.values():
        for power, units in orders.items():
            for prov, od in units.items():
                t = (od.get("type") or "").upper()
                to, fr = od.get("to"), od.get("from")
                if t == "MOVE" and to:
                    add(prov, to)
                elif t == "SUPPORT" and to:
                    add(prov, to)
                    if fr:
                        add(fr, to)
    return {k: sorted(v) for k, v in adj.items()}


def load_phases(root=RAW):
    out = {}
    for p in glob.glob(os.path.join(root, "moves", "*.json")):
        m = PHASE_RE.match(os.path.basename(p)[:-5])
        if not m:
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out[(int(m.group(1)), m.group(2), m.group(3).capitalize())] = d.get("orders") or {}
    return out


def load_messages(root=RAW):
    """Every annotated message, flattened and keyed by phase."""
    rows = []
    for split in ("train", "validation", "test"):
        path = os.path.join(root, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            for i, text in enumerate(c["messages"]):
                rows.append({
                    "game": c["game_id"],
                    "year": str(c["years"][i]),
                    "season": str(c["seasons"][i]).capitalize(),
                    "speaker": c["speakers"][i].capitalize(),
                    "receiver": c["receivers"][i].capitalize(),
                    "text": " ".join((text or "").split()),
                    "abs_index": c["absolute_message_index"][i],
                    "sender_label": c["sender_labels"][i],
                    "receiver_label": c["receiver_labels"][i],
                    "split": split,
                })
    return rows


# -- the label -------------------------------------------------------------

def occupancy(orders):
    occ = {}
    for power, units in orders.items():
        for prov in units:
            occ[prov.upper()] = power
    return occ


def acts_toward(orders, occ, player):
    """(attacks, supports, targeted_province) counters for one player's phase."""
    atk, sup = collections.Counter(), collections.Counter()
    prov_of = collections.defaultdict(set)
    for prov, od in (orders.get(player) or {}).items():
        t = (od.get("type") or "").upper()
        if t == "MOVE":
            dest = (od.get("to") or "").upper()
            q = occ.get(dest)
            if q and q != player:
                atk[q] += 1
                prov_of[("attack", q)].add(dest)
        elif t == "SUPPORT":
            src = (od.get("from") or "").upper()
            q = occ.get(src)
            if q and q != player:
                sup[q] += 1
                prov_of[("support", q)].add((od.get("to") or "").upper())
    return atk, sup, prov_of


def primary_target(atk, sup, prov_of):
    """The signed label, with the registered visible tie-break.

    Order: most orders -> targeted province is a SUPPLY CENTRE -> more units
    committed -> alphabetical. Returns (label, tie_depth) where tie_depth says
    which rule settled it; 3 means alphabetical, i.e. arbitrary, and its rate is
    reported rather than hidden.
    """
    pool = {("attack", q): c for q, c in atk.items()}
    pool.update({("support", q): c for q, c in sup.items()})
    if not pool:
        return "NONE", 0
    top = max(pool.values())
    cands = [k for k, v in pool.items() if v == top]
    if len(cands) == 1:
        return f"{cands[0][0]}:{cands[0][1]}", 0
    sc = [k for k in cands if any(p in SUPPLY_CENTRES for p in prov_of.get(k, ()))]
    if len(sc) == 1:
        return f"{sc[0][0]}:{sc[0][1]}", 1
    cands = sc or cands
    by_units = collections.Counter({k: len(prov_of.get(k, ())) for k in cands})
    top_u = max(by_units.values())
    units = [k for k in cands if by_units[k] == top_u]
    if len(units) == 1:
        return f"{units[0][0]}:{units[0][1]}", 2
    k = sorted(units)[0]
    return f"{k[0]}:{k[1]}", 3


def contacts_of(occ, adj, player):
    mine = {p for p, v in occ.items() if v == player}
    out = set()
    for a in mine:
        for b in adj.get(a, ()):
            q = occ.get(b)
            if q and q != player:
                out.add(q)
    return sorted(out)


# -- conversion ------------------------------------------------------------

def convert(root=RAW):
    phases = load_phases(root)
    msgs = load_messages(root)
    adj = build_adjacency(phases)

    by_phase = collections.defaultdict(list)
    for m in msgs:
        by_phase[(m["game"], m["year"], m["season"])].append(m)
    by_game = collections.defaultdict(list)
    for m in msgs:
        by_game[m["game"]].append(m)

    sets, points, keys = [], [], {}
    rejects = collections.Counter()
    tie_depth = collections.Counter()
    ks = []

    for game in sorted(by_game):
        gm = sorted(by_game[game], key=lambda m: m["abs_index"])
        turns = [{"speaker": m["speaker"], "text": f"(to {m['receiver']}) {m['text']}"}
                 for m in gm]
        sets.append({
            "set_id": f"diplomacy-{game:02d}", "corpus": "diplomacy", "conv_id": game,
            "channel": "press", "timeline": "press",
            "title": f"Diplomacy game {game}",
            "date": f"{1901 + game // 12:04d}-{game % 12 + 1:02d}-01",
            "tier": "silver",
            "speakers": sorted({m["speaker"] for m in gm}),
            "n_turns": len(turns),
            "full_context": "\n".join(f"{t['speaker']}: {t['text']}" for t in turns),
            "turns": turns,
        })
        keys[f"diplomacy-{game:02d}"] = {
            "schema": "diplomacy.answer_key/1", "set_id": f"diplomacy-{game:02d}",
            # speaker_intention / receiver_perception / deception_quadrant all
            # reveal roles and intent; scoring only, never a prompt.
            "messages": [{"abs_index": m["abs_index"], "speaker": m["speaker"],
                          "receiver": m["receiver"], "year": m["year"],
                          "season": m["season"],
                          "speaker_intention": "Lie" if m["sender_label"] is False else "Truth",
                          "receiver_perception": (None if m["receiver_label"] == "NOANNOTATION"
                                                  else ("Lie" if m["receiver_label"] is False
                                                        else "Truth"))}
                         for m in gm],
        }

    for key in sorted(by_phase):
        game, year, season = key
        if season not in MOVEMENT:
            continue
        orders = phases.get(key)
        if not orders:
            rejects["messaged phase with no orders"] += 1
            continue
        occ = occupancy(orders)
        # who spoke to whom this phase -- the restriction that makes the unit work
        talkers = {m["speaker"] for m in by_phase[key] if orders.get(m["speaker"])}

        # PREFIX: strictly earlier phases only. All seven order simultaneously,
        # so nothing from THIS phase -- anyone's orders, anyone's messages, the
        # adjudicated results -- may appear.
        def earlier(m):
            return (int(m["year"]), SEASON_ORDER.get(m["season"], 9)) < \
                   (int(year), SEASON_ORDER.get(season, 9))
        prefix = [{"speaker": m["speaker"], "text": f"(to {m['receiver']}) {m['text']}"}
                  for m in sorted(by_game[game], key=lambda m: m["abs_index"])
                  if earlier(m)]

        for player in sorted(talkers):
            atk, sup, prov_of = acts_toward(orders, occ, player)
            label, depth = primary_target(atk, sup, prov_of)
            tie_depth[depth] += 1
            contacts = contacts_of(occ, adj, player)
            alts = [{"action": f"{d}:{q}",
                     "gist": f"{player} {d}s {q} this phase",
                     "gives_up": ""} for q in contacts for d in ("attack", "support")]
            alts.append({"action": "NONE", "gist": f"{player} acts against nobody", "gives_up": ""})
            offered = {a["action"] for a in alts}
            if label not in offered:
                # the map is permissive, not omniscient: a real action the
                # reconstruction cannot see becomes OTHER rather than a discard,
                # because dropping it would be selection on the outcome
                alts.append({"action": "OTHER", "gist": "something not listed", "gives_up": ""})
                label_out = "OTHER"
                rejects["actual target not in contacts"] += 1
            else:
                label_out = label
            ks.append(len(alts))
            points.append({
                "cp_id": f"diplomacy-{game:02d}-{year}{season[0]}-{player}",
                "corpus": "diplomacy", "set_id": f"diplomacy-{game:02d}",
                "person": player, "person_uid": f"diplomacy-{game:02d}:{player}",
                "kind": "primary_target", "scored": True,
                "batch_id": f"diplomacy-{game:02d}-{year}{season[0]}",
                "batch_size": len(talkers),
                "year": year, "season": season, "at_turn": len(prefix),
                "situation": f"{player} orders for {season} {year}. "
                             f"In contact with: {', '.join(contacts) or 'nobody'}.",
                "alternatives": alts, "n_alternatives": len(alts),
                "actual": label_out, "tie_depth": depth,
                "prefix_turns": prefix,
                "date": f"{year}-{1 if season == 'Spring' else 7:02d}-01",
            })

    report = {
        "games": len(sets), "messages": len(msgs), "phases_with_orders": len(phases),
        "choice_points": len(points),
        "people": len({p["person_uid"] for p in points}),
        "alternatives_offered": {"mean": round(st.mean(ks), 2), "median": st.median(ks),
                                 "max": max(ks)} if ks else None,
        "label_counts": dict(collections.Counter(p["actual"] for p in points).most_common(8)),
        "tie_break_depth": {"clear": tie_depth[0], "supply_centre": tie_depth[1],
                            "more_units": tie_depth[2], "alphabetical": tie_depth[3]},
        "adjacency": {"provinces": len(adj),
                      "mean_degree": round(st.mean([len(v) for v in adj.values()]), 2)},
        "rejects": dict(rejects),
    }
    return sets, points, keys, adj, report


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return None


def write(sets, points, keys, adj):
    for d in (MUSING, CP_DIR, KEY_DIR):
        os.makedirs(d, exist_ok=True)
    json.dump(sets, open(os.path.join(MUSING, "diplomacy_dialogue.json"), "w", encoding="utf-8"))
    json.dump({s: s for s in ["Austria", "England", "France", "Germany",
                              "Italy", "Russia", "Turkey"]},
              open(os.path.join(MUSING, "diplomacy_speakers.json"), "w", encoding="utf-8"), indent=1)
    json.dump([], open(os.path.join(MUSING, "diplomacy_gaps.json"), "w", encoding="utf-8"))
    with open(os.path.join(CP_DIR, "diplomacy.jsonl"), "w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump({"git_commit": git_head(), "keys": keys},
              open(os.path.join(KEY_DIR, "diplomacy_answer_key.json"), "w", encoding="utf-8"))
    json.dump(adj, open(os.path.join(RAW, "adjacency.json"), "w", encoding="utf-8"), indent=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    sets, points, keys, adj, report = convert()
    if not a.count_only:
        write(sets, points, keys, adj)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
