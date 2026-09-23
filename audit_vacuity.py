"""Is a dominant commitment discriminating, or merely unfalsifiable?

THE GAP THIS MEASURES. split_candidates fires on heavy AND below-median
likelihood rank -- accumulated support that the present observation declines to
renew. A commitment that is too GENERAL fails that test in the wrong direction:
being compatible with whatever happens, it is ranked at or near the TOP every
step, so it is never once eligible to be split. Screened over the logged runs,
roots like "Protect someone from harm", "Understand the immediate threat" and
"To appear generous" hold 30-47% of the mass for 34-60 steps at mean rank ~1
and are split candidates on under 3% of their steps. expire_weak is a source
term for the bottom of the distribution; nothing in the filter ever challenges
a persistent winner.

But a root that ranks first every step is EITHER vacuous OR simply right, and
the logged ranks cannot tell those apart -- both look like sustained victory.
That is the question here, and it is not answerable from the logs alone: it
needs a counterfactual the run never ran.

THE TEST. Generality means excluding few observations, so substitute the
observation. Hold the slate and the context fixed exactly as logged, replace
only <observed next action> with an action drawn from a DIFFERENT step of the
same run, and re-score. A discriminating hypothesis was earning its rank from
that particular action and should fall when the action changes. A vacuous one
predicts the decoy as happily as the original and holds its place.

  delta = mean(rank under decoys) - rank under the true action

Positive delta is discrimination (better on its own observation), delta near
zero is swap-invariance, which is vacuity.

WHY RANK AND NOT SCORE. Rank is a permutation of the slate, so its mean is
fixed at (n+1)/2 whatever the decoy does to the model's overall enthusiasm. A
decoy that simply depresses every allocation moves no ranks, and the comparison
stays paired and within-slate. Raw allocations are not comparable across calls
and are reported only as a diagnostic.

CONTROLS
  paired      the true action is RE-SCORED in the same batch rather than read
              off the log, so both arms come from one model at one time. The
              logged rank was produced by whichever model version ran that
              month; reusing it would confound the swap with model drift.
              --reuse-logged opts out and halves the cost.
  positional  slate order is never touched. replay_likelihood measured rank
              tracking array index, so a shuffle between arms would put that
              artifact straight into the delta.
  near-decoys rejected. An action from an adjacent step, or one sharing most of
              its content words with the true action, is not a substitution;
              --min-gap and --max-overlap set both.
  negative    every OTHER hypothesis on the slate is measured by the same calls
              at no extra cost, so the suspects are read against the population
              that shared their prompts rather than against an absolute.
  uninformative-test guard. If nothing on any slate moves under substitution,
              the scorer is not observation-sensitive at this granularity and
              the suspects' flat deltas mean nothing. That case is reported as
              INCONCLUSIVE rather than as evidence of vacuity.

WHY THE TRACER DUMP AND NOT *.steps.jsonl. _log_step zips pre-operator
likelihoods positionally onto the post-operator population, so on any step
where an operator fired the ranks in the steps file belong to other particles.
The dump carries the scorer's own prompt, which names the slate it scored --
authoritative by construction. Slate entries are mapped back to roots by TEXT
lookup within the step rather than by position, for the same reason; entries
that resolve to nothing are counted and reported, never quietly dropped.

The prompts are rebuilt with tracer.rank_scorer_prompts, the same function
production calls, and --verify-rebuild checks byte-for-byte against the logged
prompt before any call is made.

  python audit_vacuity.py --screen                      # no LLM: who is suspect
  python audit_vacuity.py --run ep_Katara --dry-run     # the plan and the bill
  python audit_vacuity.py --run ep_Katara --self-test   # pipeline, fake model
  python audit_vacuity.py --run ep_Katara --decoys 3
"""
import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict

from tracer import (map_allocation_to_hypotheses, parse_allocation,
                    parse_ranking, rank_scorer_prompts)
from replay_likelihood import find_traces, load_steps, replay_run

# The three blocks of the rank-mode user prompt. Anchored at both ends and
# non-greedy nowhere that matters: a prompt that does not match this exactly is
# refused rather than half-parsed, because a mis-cut <observed next action>
# would silently leave the true action inside the context block and the whole
# substitution would be a no-op that looks like perfect vacuity.
PROMPT_RE = re.compile(
    r"^<previous context>\n(?P<ctx>.*)\n</previous context>\n\n"
    r"<candidate hypotheses about (?P<target>.*?)'s thoughts>\n(?P<slate>.*)\n"
    r"</candidate hypotheses about (?P=target)'s thoughts>\n\n"
    r"<observed next action>\n(?P<action>.*)\n</observed next action>$", re.S)

# How many leading characters must agree for a tier-2 text match. Long enough
# that agreement is not coincidental in free-text hypotheses about one agent in
# one scene, and the match must additionally be unique within the step.
PREFIX_MATCH = 120

STOP = set("a an the and or of to for from with by in on at into over about is are was "
           "were be been being do does did this that it he she they them his her their "
           "i you we not no yes so but if then than as up out".split())


# --------------------------------------------------------------------------
# reading the dump
# --------------------------------------------------------------------------

# The slate is built as "\n".join(f"{i+1}. {h.strip()}") and a hypothesis is
# FREE TEXT that routinely runs to several paragraphs -- and that text often
# contains its OWN numbered list. Two failures, both measured on Oppenheimer
# runs:
#   split on newlines      -> n_sent=14 for an 8-particle population
#   split on every "N. "   -> item numbers came back [1,2,3,4,5,6,1,2,3,4,7,8],
#                             a hypothesis body enumerating its own reasons
# So walk the candidate boundaries and accept one only when its number is the
# NEXT ONE EXPECTED, at column 0. A nested list restarting at 1 is then simply
# passed over, because 1 is not what comes after 6.
# Production writes items as exactly "N. " -- column 0, one space. Nested lists
# inside a hypothesis body are markdown and in practice indent ("1.  Reduce
# overhead"), so trying the strict form FIRST resolves the common collision
# without having to guess. The lenient form is the fallback for bodies that use
# a single space, and when neither yields exactly the expected count the parse
# refuses.
SLATE_ITEM_STRICT = re.compile(r"(?m)^(\d+)\. (?! )")
SLATE_ITEM = re.compile(r"(?m)^(\d+)\.[ \t]")


def parse_prompt(prompt, expect_n=None):
    """Cut a logged scorer prompt into (context, slate items, action, target).

    `expect_n` is the slate size known independently (the length of the rank
    vector). Supplying it turns the parse into a checked one: a nested list
    that happens to restart at exactly the next expected number would otherwise
    steal a boundary, and the only defence against that is knowing how many
    items there should be. Returns None rather than a best guess -- a slate cut
    in the wrong place mis-attributes every rank in the step.
    """
    m = PROMPT_RE.match(prompt or "")
    if not m:
        return None
    slate_block = m.group("slate")

    def walk(pattern):
        out, expect = [], 1
        for x in pattern.finditer(slate_block):
            if int(x.group(1)) == expect:
                out.append(x)
                expect += 1
        return out

    marks = None
    for pattern in (SLATE_ITEM_STRICT, SLATE_ITEM):
        got = walk(pattern)
        if got and (expect_n is None or len(got) == expect_n):
            marks = got
            break
    if marks is None:
        return None
    bounds = [x.start() for x in marks] + [len(slate_block)]
    slate = [slate_block[marks[i].end():bounds[i + 1]].strip() for i in range(len(marks))]
    return {"ctx": m.group("ctx"), "target": m.group("target"),
            "slate": slate, "slate_block": slate_block, "action": m.group("action")}


def read_run(path):
    """Per-step records: the scored slate, its corrected ranks, and its roots.

    Ranks come from replay_run, i.e. the ALLOCATION re-keyed against the model's
    own stated RANKING. The logged raw_scores predate that fix on most runs, and
    a screen built on them would be ranking by array index on the ~11% of steps
    where the keying went the other way.
    """
    steps = load_steps(path)
    if not steps:
        return []
    ranks_by_step = {r["step"]: r["new_rank"] for r in replay_run(steps)}

    out, skipped = [], []
    for idx, s in enumerate(steps):
        prompts = (s.get("weight_details") or {}).get("prompts")
        if not isinstance(prompts, list) or not prompts:
            continue
        ranks = ranks_by_step.get(idx)
        if not ranks:
            continue
        # The rank vector's length IS the slate size, so the parse is checked
        # against it rather than trusted. Slot i in one must be slot i in the
        # other; a zip over mismatched lengths would mis-attribute every rank
        # in the step, which is the failure replay_likelihood exists to undo.
        parsed = parse_prompt(prompts[0], expect_n=len(ranks))
        if parsed is None:
            skipped.append(idx)
            continue

        # Slate -> commitment by TEXT. The dump's arrays are post-operator on
        # steps where one fired, so position is not identity and a positional
        # read would hand each score to whichever particle landed in that slot.
        #
        # Two tiers, both textual, neither positional:
        #   exact   the scored text survived into the dump untouched
        #   prefix  propagation rewrote the tail but not the first PREFIX chars
        # A prefix match is taken only when it is UNIQUE in the step. Measured
        # on Oppenheimer runs, sibling hypotheses do share long openings, and an
        # ambiguous match is the same mis-attribution as a positional one. What
        # neither tier resolves is counted, not guessed: on operator-heavy runs
        # that is ~45% of slots, which is a fact about what the dump preserves.
        texts = [str(t).strip() for t in (s.get("texts") or [])]
        where = {}
        for j, t in enumerate(texts):
            where.setdefault(t, j)

        def locate(txt):
            j = where.get(txt)
            if j is not None:
                return j, "exact"
            key = txt[:PREFIX_MATCH]
            hits = [j for j, t in enumerate(texts)
                    if t.startswith(key) or txt.startswith(t[:PREFIX_MATCH])]
            if len(set(hits)) == 1:
                return hits[0], "prefix"
            return None, None
        anchors, roots, weights = s.get("anchors") or [], s.get("root_ids") or [], s.get("weights") or []
        members = []
        for i, txt in enumerate(parsed["slate"]):
            j, how = locate(txt)
            members.append({
                "slot": i,
                "text": txt,
                "rank": ranks[i] if i < len(ranks) else None,
                "anchor": anchors[j] if j is not None and j < len(anchors) else None,
                "root": roots[j] if j is not None and j < len(roots) else None,
                "weight": float(weights[j]) if j is not None and j < len(weights) else None,
                "resolved": j is not None,
                "match": how,
            })
        out.append({"step": idx, "target": parsed["target"], "ctx": parsed["ctx"],
                    "slate_block": parsed["slate_block"], "action": parsed["action"],
                    "n_sent": len(parsed["slate"]), "members": members,
                    "prompt": prompts[0]})
    if skipped:
        out = [r for r in out]
        for r in out:
            r["skipped_steps"] = len(skipped)
    return out


# --------------------------------------------------------------------------
# the screen -- no LLM
# --------------------------------------------------------------------------

def screen(recs, min_life=8, min_mass=0.14, max_trigger=0.10, weight_quantile=0.20):
    """Anchors that are heavy, long-lived, top-ranked and never split candidates.

    The split-candidate test is reproduced from trace_log.split_candidates --
    top-quantile by weight AND likelihood_rank at or past the median -- applied
    per step to this step's slate, so `trigger` is the fraction of its own live
    steps on which the existing trigger COULD have fired on it. A suspect is a
    commitment carrying real mass that the trigger structurally cannot reach.

    Keyed on the anchor rather than root_id: anchor IS root identity here, and
    the anchor survives the dump's post-operator array reordering intact while a
    root_id read at the wrong position does not.
    """
    per = defaultdict(list)
    for r in recs:
        n = len(r["members"])
        ranked = [m for m in r["members"] if m["rank"] is not None and m["weight"] is not None]
        if not ranked:
            continue
        top_k = max(1, int(len(ranked) * weight_quantile))
        heavy = {id(m) for m in sorted(ranked, key=lambda m: -m["weight"])[:top_k]}
        mid = len(ranked) // 2
        for m in ranked:
            if not m["anchor"]:
                continue
            # rank is 1-based, split_candidates' likelihood_rank is 0-based
            candidate = id(m) in heavy and (m["rank"] - 1) >= mid
            per[m["anchor"]].append({"step": r["step"], "rank": m["rank"], "n": n,
                                     "weight": m["weight"], "candidate": candidate})

    rows = []
    for anchor, obs in per.items():
        if len(obs) < min_life:
            continue
        rows.append({
            "anchor": anchor, "life": len(obs),
            "mass": st.mean(o["weight"] for o in obs),
            "rank": st.mean(o["rank"] for o in obs),
            "rank_sd": st.pstdev([o["rank"] for o in obs]),
            "trigger": sum(o["candidate"] for o in obs) / len(obs),
            "steps": [o["step"] for o in obs],
        })
    for row in rows:
        row["suspect"] = (row["mass"] > min_mass
                          and row["rank"] <= st.mean(len(r["members"]) for r in recs) / 2
                          and row["trigger"] <= max_trigger)
    return sorted(rows, key=lambda r: (-r["suspect"], -r["mass"]))


# --------------------------------------------------------------------------
# decoy selection
# --------------------------------------------------------------------------

def overlap(a, b):
    """Content-word Jaccard. A decoy that restates the true action is no decoy."""
    def words(s):
        return {w for w in re.findall(r"[a-z']+", (s or "").lower()) if w not in STOP}
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if (x | y) else 0.0


def pick_decoys(recs, i, k, min_gap, max_overlap, rng):
    true_action = recs[i]["action"]
    pool = [j for j, r in enumerate(recs)
            if abs(r["step"] - recs[i]["step"]) >= min_gap
            and overlap(true_action, r["action"]) <= max_overlap]
    rng.shuffle(pool)
    return pool[:k]


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

def build(rec, action):
    return rank_scorer_prompts(rec["target"], rec["n_sent"], rec["ctx"],
                               rec["slate_block"], action)


def ranks_from(raw, n):
    """Corrected ranks, 1-based, in slate order. None if the response is unusable."""
    alloc, _ = parse_allocation(raw or "", n)
    if alloc is None:
        return None
    order, _ = parse_ranking(raw or "", n)
    if order is None:
        return None
    scores, _, _ = map_allocation_to_hypotheses(alloc, order)
    out = [0] * n
    for pos, i in enumerate(sorted(range(n), key=lambda j: -scores[j]), start=1):
        out[i] = pos
    return out


def plan(recs, decoys, min_gap, max_overlap, max_steps, reuse_logged, rng, only_steps=None):
    """Which (step, decoy) pairs to score. Steps carrying a suspect come first."""
    chosen = [i for i, r in enumerate(recs) if only_steps is None or r["step"] in only_steps]
    chosen = chosen[:max_steps]
    jobs = []
    for i in chosen:
        ds = pick_decoys(recs, i, decoys, min_gap, max_overlap, rng)
        if not ds:
            continue
        jobs.append({"i": i, "decoys": ds, "true_call": not reuse_logged})
    return jobs


def run_probe(model, recs, jobs, logged_ranks):
    """One batch of calls, then deltas per slate slot."""
    prompts, sys_prompts, tags = [], [], []
    for job in jobs:
        rec = recs[job["i"]]
        if job["true_call"]:
            s, p = build(rec, rec["action"])
            sys_prompts.append(s); prompts.append(p); tags.append((job["i"], "true", None))
        for j in job["decoys"]:
            s, p = build(rec, recs[j]["action"])
            sys_prompts.append(s); prompts.append(p); tags.append((job["i"], "decoy", j))

    raws = model.batch_interact(prompts, system_prompts=sys_prompts,
                                temperature=0, max_tokens=2048)

    true_rank, decoy_ranks, unparsed = {}, defaultdict(list), 0
    for (i, kind, j), raw in zip(tags, raws):
        rk = ranks_from(raw, recs[i]["n_sent"])
        if rk is None:
            unparsed += 1
            continue
        if kind == "true":
            true_rank[i] = rk
        else:
            decoy_ranks[i].append(rk)
    for job in jobs:
        if not job["true_call"]:
            true_rank[job["i"]] = logged_ranks[job["i"]]

    rows = []
    for i, tr in true_rank.items():
        if not decoy_ranks[i]:
            continue
        for m in recs[i]["members"]:
            s = m["slot"]
            ds = [d[s] for d in decoy_ranks[i]]
            rows.append({"step": recs[i]["step"], "anchor": m["anchor"], "slot": s,
                         "weight": m["weight"], "rank_true": tr[s],
                         "rank_decoy": st.mean(ds), "decoy_sd": st.pstdev(ds),
                         "delta": st.mean(ds) - tr[s], "n_decoys": len(ds)})
    return rows, unparsed


class FakeModel:
    """Deterministic stand-in for --self-test.

    Exercises prompt construction, parsing, keying and aggregation without a
    key. Its ranking is a hash, so this is a test of the PIPELINE and
    emphatically not evidence about any run. Two modes, because both branches
    of the report need exercising: 'moving' keys on (slate entry, action) and
    must produce non-zero deltas, 'flat' keys on the slate alone and must be
    caught by the INCONCLUSIVE guard.
    """

    def __init__(self, mode="moving"):
        self.mode = mode

    def batch_interact(self, prompts, system_prompts=None, temperature=0, max_tokens=None):
        out = []
        for p in prompts:
            parsed = parse_prompt(p)
            n = len(parsed["slate"])
            # 'flat' ranks on the slate alone, so substitution changes nothing
            # and every delta is exactly 0 -- the state the guard must catch.
            key = (lambda i: hash(parsed["slate"][i])) if self.mode == "flat" \
                else (lambda i: hash((parsed["slate"][i], parsed["action"])))
            order = sorted(range(n), key=key)
            lines = ["RANKING"] + [f"{r+1}st: {h+1}" for r, h in enumerate(order)]
            lines += ["", "ALLOCATION"] + [f"{r+1}: {100 - 5*r}" for r in range(n)]
            lines += ["", "REASONING", "self-test"]
            out.append("\n".join(lines))
        return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="substring of a run id")
    ap.add_argument("--screen", action="store_true", help="no LLM: list suspect commitments")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and call budget")
    ap.add_argument("--self-test", nargs="?", const="moving", choices=["moving", "flat"],
                    help="run the pipeline on a fake model: 'moving' reacts to the action, "
                         "'flat' ignores it and must trip the INCONCLUSIVE guard")
    ap.add_argument("--verify-rebuild", action="store_true",
                    help="check rebuilt prompts against the logged ones and stop")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--decoys", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8, help="probe steps per run")
    ap.add_argument("--min-gap", type=int, default=5, help="minimum step distance for a decoy")
    ap.add_argument("--max-overlap", type=float, default=0.30,
                    help="reject a decoy sharing more than this content-word Jaccard")
    ap.add_argument("--reuse-logged", action="store_true",
                    help="read the true-action rank off the log instead of re-scoring")
    ap.add_argument("--min-life", type=int, default=8)
    ap.add_argument("--min-mass", type=float, default=0.14)
    ap.add_argument("--max-trigger", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write per-slot rows here")
    a = ap.parse_args()

    paths = find_traces(a.run)
    if not paths:
        print("no tracer dumps matched"); return
    rng = random.Random(a.seed)

    # KEYED BY PATH, NOT RUN ID. A run id can name more than one dump --
    # ep_Katara executed twice concurrently -- and those are two samples, not a
    # stale copy. Keying the suspect lists by id let the second dump's (empty)
    # list overwrite the first's, so the probe ran with nothing marked suspect
    # and reported every slot as "other".
    all_rows, all_suspects, budget, jobs_by_run = [], {}, 0, {}
    runs = []
    for rid, path in paths:
        recs = read_run(path)
        if len(recs) < a.min_life:
            continue
        rows = screen(recs, a.min_life, a.min_mass, a.max_trigger)
        runs.append({"rid": rid, "path": path, "recs": recs, "rows": rows})
        all_suspects[path] = [r for r in rows if r["suspect"]]

    if not runs:
        print("no run had enough scored steps"); return

    # ---- rebuild verification: nothing is called until this passes ----
    if a.verify_rebuild:
        ok = bad = 0
        for run in runs:
            for r in run["recs"]:
                _, rebuilt = build(r, r["action"])
                ok += rebuilt == r["prompt"]; bad += rebuilt != r["prompt"]
        print(f"rebuilt prompts identical to logged: {ok}    differing: {bad}")
        if bad:
            print("REFUSING to probe: the rebuilt prompt is not the prompt the filter used.")
        return

    # ---- the screen ----
    tier = defaultdict(int)
    for run in runs:
        for r in run["recs"]:
            for m in r["members"]:
                tier[m["match"]] += 1
    total_slots = sum(tier.values()) or 1
    print(f"{len(runs)} runs, {sum(len(run['recs']) for run in runs)} scored steps, "
          f"{total_slots} slate slots")
    print(f"  resolved to a commitment: {tier['exact']/total_slots:.1%} exact + "
          f"{tier['prefix']/total_slots:.1%} unique-prefix; "
          f"{tier[None]/total_slots:.1%} unresolved and excluded")
    print(f"\n{'run':<24}{'life':>5}{'mass':>7}{'rank':>6}{'sd':>6}{'trig':>6}  commitment")
    print("-" * 100)
    shown = 0
    for run in runs:
        rid = run["rid"]
        for r in run["rows"]:
            if not r["suspect"]:
                continue
            shown += 1
            print(f"{rid[:23]:<24}{r['life']:>5}{r['mass']:>7.3f}{r['rank']:>6.1f}"
                  f"{r['rank_sd']:>6.2f}{r['trigger']:>6.2f}  {str(r['anchor'])[:44]}")
    print("-" * 100)
    print(f"{shown} suspect commitments: heavy, long-lived, top-ranked, and a split "
          f"candidate on <={a.max_trigger:.0%} of their steps.")
    if a.screen:
        return

    # ---- plan ----
    for run in runs:
        suspect_steps = {s for r in run["rows"] if r["suspect"] for s in r["steps"]}
        jobs = plan(run["recs"], a.decoys, a.min_gap, a.max_overlap, a.max_steps,
                    a.reuse_logged, rng, only_steps=suspect_steps or None)
        jobs_by_run[run["path"]] = jobs
        budget += sum(len(j["decoys"]) + (1 if j["true_call"] else 0) for j in jobs)
    print(f"\nplan: {sum(len(j) for j in jobs_by_run.values())} probe steps, "
          f"{a.decoys} decoys each, {budget} LLM calls"
          f"{' (true arm re-scored)' if not a.reuse_logged else ' (true arm read off the log)'}")
    if a.dry_run:
        for run in runs:
            jobs = jobs_by_run.get(run["path"]) or []
            if not jobs:
                continue
            print(f"  {run['rid'][:23]:<24}{len(jobs)} probe steps: "
                  f"{', '.join(str(run['recs'][j['i']]['step']) for j in jobs[:12])}")
        return

    if a.self_test:
        model = FakeModel(a.self_test)
    else:
        from agents.load_model import load_model
        model = load_model(a.model, {"model": a.model})

    unparsed_tot = 0
    for run in runs:
        jobs = jobs_by_run.get(run["path"]) or []
        if not jobs:
            continue
        recs = run["recs"]
        logged = {i: [m["rank"] for m in recs[i]["members"]] for i in range(len(recs))}
        probe_rows, unparsed = run_probe(model, recs, jobs, logged)
        unparsed_tot += unparsed
        for r in probe_rows:
            r["run"] = run["rid"]
        all_rows += probe_rows

    if not all_rows:
        print("\nno usable responses"); return

    suspect_anchors = {r["anchor"] for rows in all_suspects.values() for r in rows}
    sus = [r for r in all_rows if r["anchor"] in suspect_anchors]
    rest = [r for r in all_rows if r["anchor"] not in suspect_anchors]

    print(f"\n{'':24}{'n':>5}{'rank(true)':>12}{'rank(decoy)':>13}{'delta':>8}{'|delta|':>9}")
    print("-" * 72)
    for name, group in (("suspect commitments", sus), ("every other slot", rest)):
        if not group:
            continue
        print(f"{name:<24}{len(group):>5}{st.mean(r['rank_true'] for r in group):>12.2f}"
              f"{st.mean(r['rank_decoy'] for r in group):>13.2f}"
              f"{st.mean(r['delta'] for r in group):>8.2f}"
              f"{st.mean(abs(r['delta']) for r in group):>9.2f}")
    print("-" * 72)
    moved = st.mean(abs(r["delta"]) for r in all_rows)
    print(f"unparsed responses: {unparsed_tot}")
    if moved < 0.5:
        print(f"\nINCONCLUSIVE. Mean |delta| across ALL slots is {moved:.2f} rank positions: "
              f"substituting the observation barely moves anything, so the scorer is not "
              f"observation-sensitive at this granularity and a flat suspect delta is not "
              f"evidence of vacuity. Raise --decoys or lower --max-overlap before reading "
              f"the table above.")
    elif sus:
        d_s, d_r = st.mean(r["delta"] for r in sus), st.mean(r["delta"] for r in rest or sus)
        print(f"\nSuspects move {d_s:+.2f} rank positions under substitution against "
              f"{d_r:+.2f} for the rest of their own slates.")
        print("A suspect delta at or below zero while the slate moves is the vacuity signal: "
              "the commitment held its rank on an action it never saw.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(all_rows, fh, indent=1)
        print(f"\nwrote {len(all_rows)} rows to {a.json}")


if __name__ == "__main__":
    main()
