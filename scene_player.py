"""Play a scene back with both targets' belief boards moving beside it.

`stacked.py` and `evolution.py` draw a run as a chart -- one line per
commitment across the whole scene. That answers "what happened to this
hypothesis" but not "what was this character holding when they said THAT", and
the second question is the one the boards exist to answer. This emits a
self-contained HTML player instead: the transcript runs line by line, and each
target's ranked commitments re-order beside it.

    python scene_player.py v4_katara v4_zuko \
        --corpus atla --sets atla-0604,atla-0605,atla-0606,atla-0607,\
atla-0608,atla-0609,atla-0610,atla-0611 \
        --start-turn 20 --title "The Southern Raiders Trace" \
        --out musing_out/reports/southern_raiders_player.html

Takes exactly two runs, because the layout is a scene between two minds. Both
must be traces of the SAME scene span -- the player has no way to check that
and will happily place two unrelated runs on one transcript.

TWO THINGS HERE ARE DELIBERATELY NOT REUSED FROM stacked.py
-----------------------------------------------------------
ALIGNMENT. `stacked.scene_positions` matches a step to a turn through the last
`<response>` block in the likelihood prompt, which is the turn BEFORE the one
being scored. On this pair it put Zuko's steps 0 and 1 both on turn 0 and left
2 of Katara's 41 unmatched. A chart survives that -- the line still rises in
roughly the right place -- but a player does not, because the reader is
watching one named utterance go by. `align` matches on `scored_action`
instead, walking the transcript forward so a repeated line cannot cross-match
("What are you doing?" is said by three different characters in this scene),
and splitting on newlines so a step that merged consecutive turns claims all of
them. That places 40/40 and 37/37.

CAUSE. The steps file records what the population IS, not why it changed:
`operators_fired` names the operators that ran but not which commitment each
one hit. Lineage recovers it. Perturb, expire and split all keep the particle's
`lineage_id` and mint a new `root_id` under it (the anchor==root invariant), so
a new root sharing a lineage with a root that VANISHED this step is that root's
replacement or its child. Restricting to vanished roots is what makes it sound:
resampling duplicates particles, so one lineage can point at several prior
roots, and a prior root still sitting on the board plainly did not turn into
anything. Merges are read straight off `anchor_collapses`, which already names
the commitment kept and the one lost.

Everything is computed offline from the steps files. No LLM calls.
"""
import argparse
import json
import os
import re
import sys

import musing_layout as ml
import stacked as st

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_player.html")

norm = lambda s: re.sub(r"\s+", " ", s or "").strip()


def loose_keys(text):
    """Match keys that survive the drift between the transcript and the log.

    Three differences show up on Slack corpora, and each needs its own key:
      - <@U0677JT1HSL> mentions and &lt;/&gt; entities appear in the transcript
        and not in the logged scored_action;
      - the log sometimes drops a leading greeting, so "hi <@U...>, sry for the
        late reply" is logged as ", sry for the late reply" -- a PREFIX key
        shifts and misses, a suffix key still lands;
      - the log sometimes truncates, so a SUFFIX key misses and a prefix lands.
    Returning both ends means one of them hits. Measured on the Bloomfield pair:
    9/15 placed with an exact key, 11/15 with a prefix key, 14/15 with both.
    """
    t = re.sub(r"<@[A-Z0-9]+>", " ", text or "")
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"[^0-9a-z]+", " ", t.lower()).strip()
    if len(t) < 12:
        return [t] if t else []
    return [t[:60], t[-60:]]


# ------------------------------------------------------------------ transcript
def scene(corpus, set_ids):
    """The turns of one or more sets, stitched in order, with scene boundaries."""
    d = json.load(open(f"data/musing/{corpus}_dialogue.json"))
    sets = d if isinstance(d, list) else d.get("sets", d)
    by = {x.get("set_id"): x for x in sets}
    turns, scenes = [], []
    for sid in [x.strip() for x in set_ids.split(",") if x.strip()]:
        if sid not in by:
            sys.exit(f"set {sid} not found in {corpus}")
        scenes.append({"set_id": sid, "title": by[sid]["title"], "start": len(turns)})
        for tn in by[sid]["turns"]:
            turns.append({"speaker": tn["speaker"], "text": norm(tn["text"])})
    return turns, scenes


def align(steps, turns, index):
    """Each step's turn span, by walking the transcript forward.

    Forward-only because the same line recurs verbatim in a scene and a global
    lookup would bind a late step to an early turn. The fallback to a global
    search exists for a step whose utterance genuinely sits behind the cursor;
    it has not fired on the ATLA pair, and a run where it fires a lot is a sign
    the two runs are not tracing the same span.
    """
    out, cur = [], 0
    for s in steps:
        span = []
        for line in [l for l in (s.scored_action or "").split("\n") if l.strip()]:
            sp, sep, txt = line.partition(":")
            # A speaker prefix is a short bare name; anything longer is a line
            # that simply had none, in which case the whole line is the text.
            if not sep or len(sp) > 30 or sp.count(" ") > 2:
                sp, txt = "", line
            cands = index.get((sp.strip(), norm(txt)), [])
            for k in ([] if cands else loose_keys(txt)):
                cands = index.get(("~loose", k), [])
                if cands:
                    break
            k = next((j for j in cands if j >= cur), cands[0] if cands else None)
            if k is not None:
                span.append(k)
                cur = k + 1
        out.append(span)
    return out


# ------------------------------------------------------------------ the boards
def roots_of(step):
    """Root-level belief state.

    Mass is SUMMED over a root's particles, not read off one of them:
    resampling encodes the posterior in multiplicity, so a winner duplicated
    three times at 1/N each is holding 3/N and a per-particle read would show
    it at a third of its real strength.
    """
    agg = {}
    for p in step.particles:
        r = p.root_id or p.lineage_id or p.particle_id
        a = agg.setdefault(r, {"root": r, "mass": 0.0, "copies": 0, "best": -1.0,
                               "anchor": "", "text": "", "standard": "",
                               "lineages": set()})
        w = p.weight or 0.0
        a["mass"] += w
        a["copies"] += 1
        a["lineages"].add(p.lineage_id or p.particle_id)
        if w > a["best"]:
            a["best"], a["anchor"], a["text"] = w, p.anchor or "", p.text or ""
            a["standard"] = getattr(p, "standard", None) or ""
    return agg


def agent_name(run_id, steps):
    """The target's name, from the run's own metadata where possible.

    Parsing it out of the first scored_action fails whenever that message has
    no "Speaker: " prefix -- on the purple_threshold pair it produced the agent
    name "to image", taken from the start of Lyubovsky's first line. The run
    metadata records target_agent explicitly, so prefer it and keep the parse
    as a fallback.
    """
    try:
        for line in open(ml.meta_path("runs.jsonl"), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = d.get("meta", d)
            if str(d.get("run_id") or m.get("run_id") or "").startswith(run_id):
                if m.get("target_agent"):
                    return m["target_agent"]
    except (OSError, json.JSONDecodeError):
        pass
    first = (steps[0].scored_action or ":") if steps else ":"
    sp = first.partition(":")[0].strip()
    return sp if sp and len(sp) <= 24 and sp.count(" ") <= 1 else run_id.split("_")[-1]


def trace(run_id, turns, index):
    """One event per filter step: the whole board, plus what formed and dropped."""
    steps = st.load(run_id)
    spans = align(steps, turns, index)
    agent = agent_name(run_id, steps)
    prev, evs = {}, []

    for s, span in zip(steps, spans):
        cur = roots_of(s)
        order = sorted(cur.values(), key=lambda a: -a["mass"])
        prev_order = sorted(prev.values(), key=lambda a: -a["mass"])
        prev_rank = {a["root"]: k for k, a in enumerate(prev_order)}
        minted = set(s.minted_roots or [])
        collapses = s.anchor_collapses or []
        ops = s.operators_fired or []
        gone = [r for r in prev if r not in cur]
        prev_by_lin = {l: a for a in prev.values() for l in a["lineages"]}

        beliefs = [{"root": a["root"], "anchor": a["anchor"], "text": a["text"],
                    "standard": a["standard"],
                    "mass": round(a["mass"], 4), "copies": a["copies"], "rank": k,
                    "prevRank": prev_rank.get(a["root"]),
                    "new": a["root"] not in prev, "minted": a["root"] in minted}
                   for k, a in enumerate(order)]

        # what each new root came out of -- see the module docstring
        origin = {}
        for b in beliefs:
            if not b["new"]:
                continue
            src = {prev_by_lin[l]["root"] for l in cur[b["root"]]["lineages"]
                   if l in prev_by_lin and prev_by_lin[l]["root"] != b["root"]}
            pick = src & set(gone)
            origin[b["root"]] = next(iter(pick)) if len(pick) == 1 else None
        # one unexplained arrival against one unexplained exit is that swap
        loose_in = [b["root"] for b in beliefs if b["new"] and not origin.get(b["root"])]
        loose_out = [r for r in gone if r not in set(origin.values())]
        if len(loose_in) == 1 and len(loose_out) == 1 and \
                ("perturb" in ops or "expire" in ops):
            origin[loose_in[0]] = loose_out[0]

        kids = {}
        for r, o in origin.items():
            if o:
                kids.setdefault(o, []).append(r)
        name = {r: a["anchor"] for r, a in list(prev.items()) + list(cur.items())}

        # a dropped root's last value is in the PRE-operator population: it was
        # reweighted and only then removed, so its post-operator weight is gone
        pre = {}
        for q in (s.pre_operator_particles or []):
            rr = q.get("root_id")
            pre[rr] = pre.get(rr, 0.0) + float(q.get("weight") or 0.0)

        dropped = []
        for r in gone:
            a, ch = prev[r], kids.get(r, [])
            kept = next((c.get("kept") for c in collapses if c.get("lost") == a["anchor"]), None)
            if len(ch) > 1:
                cause, into = "split apart", " / ".join(name.get(c, "?") for c in ch)
            elif len(ch) == 1:
                cause, into = "rewritten as", name.get(ch[0], "?")
            elif kept:
                cause, into = "merged into", kept
            elif "expire" in ops:
                cause, into = "expired — weak too long", None
            elif "resample" in ops:
                cause, into = "resampled away", None
            else:
                cause, into = "dropped", None
            dropped.append({"root": r, "anchor": a["anchor"], "text": a["text"],
                            "standard": a["standard"],
                            "lastMass": round(pre.get(r, a["mass"]), 4),
                            "prevRank": prev_rank.get(r), "cause": cause, "into": into})

        formed = [b for b in beliefs if b["new"]]
        for b in formed:
            o = origin.get(b["root"])
            if o and len(kids.get(o, [])) > 1:
                b["cause"], b["from"] = "split off", name.get(o)
            elif o:
                b["cause"], b["from"] = "replaced", name.get(o)
            elif b["minted"]:
                b["cause"], b["from"] = "founded fresh", None
            else:
                b["cause"], b["from"] = "appeared", None

        evs.append({"agent": agent, "step": s.step_idx, "span": span,
                    "turn": span[0] if span else None,
                    "beliefs": beliefs, "formed": formed, "dropped": dropped,
                    "lead": order[0]["root"] if order else None,
                    "prevLead": prev_order[0]["root"] if prev_order else None,
                    "ops": ops,
                    "ess": round(s.root_mass_ess, 4) if s.root_mass_ess is not None else None,
                    "pop": s.population_post})
        prev = cur
    return agent, evs


def role_prior(run_id):
    """The SEAT/STAKE/PRESSURE the run was actually given, off its own log.

    Paraphrasing it into the page would put a caption next to the numbers that
    no run produced. If the log is not there the board simply shows no role.
    """
    path = f"musing_out/{run_id}.log"
    if not os.path.exists(path):
        return ""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read()[:4000])
    m = re.search(r"Role prior.*?\n(.*?)\n\s*\n", txt, re.S)
    if not m:
        return ""
    body = norm(" ".join(l.strip() for l in m.group(1).splitlines()))
    parts = re.split(r"\b(SEAT|STAKE|PRESSURE):\s*", body)
    d = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    out = []
    if d.get("STAKE"):
        out.append("Stake: " + d["STAKE"].rstrip(". ") + ".")
    if d.get("PRESSURE"):
        out.append("Pressure: " + d["PRESSURE"].rstrip(". ") + ".")
    return " ".join(out) or body


# ------------------------------------------------------------------- rendering
# ------------------------------------------------------- cultural divergence
# The misalignment this corpus is built around is not two parties disagreeing
# about a number -- it is two parties using one word for two different things.
# On the Bloomfield thread "size" means commercial caliber grading for export
# on one side and green-berry sizing for harvest timing on the other. Neither
# side is wrong and neither is arguing, which is exactly why it goes unnoticed.
#
# Scoring each commitment on which vocabulary it speaks makes that visible: a
# board leaning one way beside a board leaning the other IS the misalignment,
# and the moments where the two lean hardest apart are the ones worth reading.
#
# Lexicons are per-corpus and deliberately small; a commitment matching neither
# scores 0 and is drawn neutral rather than forced onto an axis it is not on.
# The first bloomfield axis is a TOPIC axis -- what the turn is about. It is
# blind to the dispute that actually runs through this corpus, which is not
# about topic but about EVIDENCE STANDARD: Wolf (CEO) judges a number by what
# the company ships to a named customer, the engineers judge it by what the
# pipeline can measure and reproduce. Measured: the sharpest turn in the whole
# Wolf trace --
#   "but we don't deliver data like based on that? why wouldn't we want to
#    draw a comparison against what we actually deliver (estimated counts)"
# -- scores 0.0 on the topic axis. Not one word matches, so the page draws the
# key moment neutral. These two word lists are not hand-picked: they are the
# top log-odds discriminators between Wolf and {Lyubovsky, Rovani, Berger,
# Ortega} over the whole corpus, with Slack URL fragments dropped.
LEXICONS = {
    # Reads the STANDARD field's own vocabulary rather than the transcript's.
    # With prompt variant v4 the model names the source in one reserved word,
    # so this is a direct read, not an inference: a commitment settled by a
    # colleague agreeing sits at one pole, one settled by a measurement at the
    # other. That axis IS the finding the page exists to show -- the filter
    # inferred a status motive for Wolf out of four questions, and what
    # actually separates these three is what each of them would accept as
    # proof.
    "bloomfield_settle": (
        ("people", re.compile(r"^\s*\**\s*(COLLEAGUE|CUSTOMER)\b|"
                              r"\b(confirms?|agrees?|acknowledges?|approves?|accepts?|states?|describes?|"
                              r"explains?|clarifies|demonstrates?|provides?|corroborat|"
                              r"weigh in|consensus|colleague|expert|customer|grower|client|"
                              r"stakeholder|trusted source)\b", re.I)),
        ("measurement", re.compile(r"^\s*\**\s*(MEASUREMENT|DOCUMENT)\b|"
                                   r"\b(RGB|spectromet|statistic|confidence interval|"
                                   r"percentage|objective measure|sensor|reconcil|correlation|"
                                   r"metric|threshold|algorithm|recount|ground truth|audit|\\breport\\b|"
                                   r"independently label|document|guideline|spec)\b", re.I)),
    ),
    "bloomfield_standard": (
        ("deliverable", re.compile(r"deliver|customers?|client|hortifrut|agrovision|"
                                   r"estimated|estimate|ship|block|rows?|rowside|plant|"
                                   r"spacing|ready|per bush|bush|what we (actually )?(deliver|sell)",
                                   re.I)),
        ("measurable", re.compile(r"dataset|datasets|labels?|labell?ing|annotat|experiments?|"
                                  r"batch|monorepo|github|model|metric|threshold|percentage|"
                                  r"visible count|ground truth|reproduc|pipeline|validat",
                                  re.I)),
    ),
    "bloomfield": (
        ("commercial", re.compile(r"caliber|calibre|market|export|commercial|sales|buyer|"
                                  r"premium|packing|price|client|customer|grade|grading|"
                                  r"jumbo|mediano|grande|peru", re.I)),
        ("harvest", re.compile(r"harvest|green berr|green stage|ripe|ripen|maturity|matur|"
                               r"picking|pick |labou?r|phenolog|yield|estimation|crop", re.I)),
    ),
}


# Prompt variant v4 makes the model name the settlement source in one reserved
# word. Where that word is present it is the answer, and counting the rest of
# the sentence only adds noise: "MEASUREMENT; two labellers AGREE on 50 images"
# scored 0.0 because "agree" pulled one way while the keyword pulled the other.
EXPLICIT_POLE = {"COLLEAGUE": -1.0, "CUSTOMER": -1.0,
                 "MEASUREMENT": 1.0, "DOCUMENT": 1.0}
_LEAD = re.compile(r"^\s*\**\s*([A-Z]{6,12})\b")


def culture(text, lex):
    """-1 = wholly the first lexicon, +1 = wholly the second, 0 = neither."""
    if not lex:
        return 0.0
    m = _LEAD.match(text or "")
    if m and m.group(1) in EXPLICIT_POLE:
        return EXPLICIT_POLE[m.group(1)]
    a = len(lex[0][1].findall(text or ""))
    b = len(lex[1][1].findall(text or ""))
    return 0.0 if not (a + b) else round((b - a) / (a + b), 3)


def pack(turns, scenes, runs, meta):
    """Normalise: one entry per commitment, steps carry only ids and numbers.

    The board text repeats verbatim on every step it survives, which is most of
    the payload. Keying it out once takes the Southern Raiders pair from 275KB
    to 56KB and makes the player's lookups trivial.
    """
    events = []
    names = []
    for i, (agent, evs) in enumerate(runs):
        names.append(agent)
        for e in evs:
            e["a"] = i
            events.append(e)
    # a step that scores no utterance logs the board after the scene ends and
    # carries nothing the step before it did not already show
    events = [e for e in events if e["turn"] is not None]
    events.sort(key=lambda e: (e["turn"], e["a"], e["step"]))

    lex = LEXICONS.get(meta.get("lexicon_key") or meta.get("corpus"))
    roots, packed = {}, []
    for e in events:
        for b in e["beliefs"] + e["dropped"]:
            # Score the STANDARD when the run has one. It is the field that
            # names the axis directly, so where it exists it is better
            # evidence of which side a commitment sits on than the aim and
            # the belief prose, which are about wanting rather than proving.
            roots.setdefault(b["root"], {"agent": names[e["a"]],
                                         "anchor": b["anchor"], "text": b["text"],
                                         "standard": b.get("standard") or "",
                                         "cult": culture(
                                             b.get("standard") or f"{b['anchor']} {b['text']}",
                                             lex)})
        packed.append({
            "a": e["a"], "s": e["step"], "t": e["turn"], "span": e["span"],
            "b": [[b["root"], b["mass"], b["copies"], b["prevRank"]] for b in e["beliefs"]],
            "f": [[b["root"], b["cause"], b.get("from")] for b in e["formed"]],
            "d": [[x["root"], x["cause"], x.get("into"), x["lastMass"], x["prevRank"]]
                  for x in e["dropped"]],
            # mass-weighted lean of the whole board, so a side is judged on
            # what it is actually holding rather than on its top line alone
            "cult": round(sum(b["mass"] * roots[b["root"]]["cult"] for b in e["beliefs"])
                          / (sum(b["mass"] for b in e["beliefs"]) or 1), 3),
            "lead": 1 if (e["prevLead"] and e["lead"] != e["prevLead"]) else 0,
            "ops": e["ops"], "ess": e["ess"],
        })
    # Score the TRANSCRIPT too, not only the boards. The commitments are the
    # filter's reading; the turns are what was actually said. Showing both lets
    # the reader check the filter against the record instead of taking it on
    # trust -- and it is the turn score, not the board score, that makes the
    # "what we actually deliver" line legible as one side of an axis.
    return {"meta": meta, "scenes": scenes, "roots": roots, "events": packed,
            "turns": [[t["speaker"], t["text"], culture(t["text"], lex)] for t in turns]}


def render(data, title, out):
    page = open(TEMPLATE).read()
    blob = json.dumps(data, separators=(",", ":"))
    # </script> inside a JSON string would close the tag it is sitting in
    page = page.replace("__TITLE__", title).replace("__DATA__", blob.replace("</", "<\\/"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(page)
    return len(page)


DEFAULT_BLURB = (
    "A particle filter reads the scene one utterance at a time and keeps a ranked set of "
    "<b>commitments</b> for each character. Press play: the boards re-order as the scene "
    "runs, and every commitment that <b>forms</b> or <b>drops</b> is named on the board "
    "that holds it. The line being scored stays at the top of the column; a dot in the "
    "margin marks the ones the filter scored.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+",
                    help="two or three run ids, all tracing the same scene span. Two are "
                         "placed left and right of the transcript; a third runs along the "
                         "bottom, because a third column leaves no board wide enough to read "
                         "a commitment in and the transcript is what the layout exists to keep "
                         "central.")
    ap.add_argument("--corpus", default="atla")
    ap.add_argument("--sets", required=True, help="comma-separated set ids, in order")
    ap.add_argument("--start-turn", type=int, default=0,
                    help="turn playback opens on; earlier turns stay readable above it")
    ap.add_argument("--title", default="Scene Trace")
    ap.add_argument("--lexicon", default=None,
                    help="which axis to score commitments on; defaults to the corpus "
                         "name. bloomfield_standard scores deliverable-vs-measurable "
                         "rather than topic")
    ap.add_argument("--blurb", default=None,
                    help='standing explanation under the title; "" removes it, '
                         'which a three-board layout needs to give the bottom board height')
    ap.add_argument("--tagline", default="two minds, traced live")
    ap.add_argument("--out", default="musing_out/reports/scene_player.html")
    a = ap.parse_args()
    if not 2 <= len(a.runs) <= 3:
        sys.exit(f"need two or three runs, got {len(a.runs)}")

    turns, scenes = scene(a.corpus, a.sets)
    index = {}
    for i, tn in enumerate(turns):
        index.setdefault((tn["speaker"], tn["text"]), []).append(i)
        # Fallback key: content only, mentions and entities stripped. Consulted
        # after the exact key so an unambiguous match still wins.
        for k in loose_keys(tn["text"]):
            index.setdefault(("~loose", k), []).append(i)

    runs = [trace(r, turns, index) for r in a.runs]
    placed = [sum(1 for e in evs if e["turn"] is not None) for _, evs in runs]
    for (agent, evs), n, rid in zip(runs, placed, a.runs):
        print(f"  {rid:<14} {agent:<10} {len(evs):>3} steps, {n} placed on the transcript")
        if n < len(evs) - 1:
            print(f"    WARNING: {len(evs) - n} steps did not match a turn -- "
                  f"is {rid} a trace of {a.sets}?")

    sid = [x.strip() for x in a.sets.split(",") if x.strip()]
    meta = {
        "title": a.title,
        "tagline": a.tagline,
        "corpus": a.corpus,
        "lexicon_key": a.lexicon or a.corpus,
        "lexicon": [n for n, _ in (LEXICONS.get(a.lexicon or a.corpus) or [])],
        "start": a.start_turn,
        "agents": [{"name": agent, "role": role_prior(rid), "run": rid}
                   for (agent, _), rid in zip(runs, a.runs)],
        "tags": [f"{rid} · {len(evs)} steps" for (_, evs), rid in zip(runs, a.runs)] +
                [f"{sid[0]} → {sid[-1].split('-')[-1]}",
                 f"{runs[0][1][0]['pop']} particles"],
        # Three boards plus the transcript leave no vertical room to spare, and the
        # bottom board pays for everything above it. --blurb "" drops the standing
        # explanation outright rather than shrinking it.
        "blurb": DEFAULT_BLURB if a.blurb is None else a.blurb,
    }

    data = pack(turns, scenes, runs, meta)
    n = render(data, a.title, a.out)
    ev = data["events"]
    print(f"  {len(turns)} turns · {len(ev)} events · {len(data['roots'])} commitments · "
          f"{sum(len(e['f']) for e in ev)} formed · {sum(len(e['d']) for e in ev)} dropped · "
          f"{sum(e['lead'] for e in ev)} lead changes")
    print(f"  wrote {a.out} ({n // 1024} KB)")


if __name__ == "__main__":
    main()
