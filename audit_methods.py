"""What did each method surface that nothing else did?

The eval table scores HYGIENE -- which methods misbehave. This scores YIELD.
They are different questions and the second is the one the twelve methods
exist to answer: HYPOTHESES.md's claim is that method predicts failure mode,
and DEBUG_HANDOFF.md's open problem is that "in a Slack work thread nobody
states why they want a parameter... none of them makes it able to infer one
that is never said."

So the unit here is the CLAUSE, and the test is whether it can be quoted from
the transcript. DEBUG_HANDOFF.md already draws that line: specifics like "XXL
over 22mm" or "Yon Rha" are retrievable by search, and "the thing only
inference can supply is WHY the person is saying it."

Three passes, each filtering the last:

  1. UNIQUE YIELD (no LLM). The commitments each method FOUNDED, diffed
     against every other method and against a baseline run.
  2. EARNED OR MERELY NOVEL (no LLM). Each unique clause classified from its
     root's trajectory crossed with echo/boundness.
  3. THE NAMED-THING AUDIT (LLM, one call per method). Only the survivors go
     to a judge: quote the line that states this, or answer NOT STATED.

Two controls, both mandatory, because novelty is trivially easy to manufacture:

  * the BASELINE floor -- a commitment the default prompts already produce is
    not a method's yield;
  * the SAME-METHOD, DIFFERENT-SEED floor -- two seeds of one method also
    produce mutually-unique commitments, and that rate is what any
    cross-method novelty number has to clear. Same guard as
    eval_motive_sep.self_control, for the same reason.

Reads musing_out only. Passes 1 and 2 make no LLM calls at all.

  python audit_methods.py --run me_katara --baseline base_katara
  python audit_methods.py --run me_katara --baseline base_katara --judge
  python audit_methods.py --run me_katara --seeds me_katara_s1,me_katara_s2
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

from methods import METHODS, family_of
from restatement import BOUND, bag, stem

# A clause has to survive this many steps to count as anything but noise. Three
# is the shortest run that cannot be produced by a single scoring step plus the
# resample that follows it.
SURVIVAL_STEPS = 3
# Content-word overlap with the scored turn, above which a clause is reading
# the message back rather than explaining it.
ECHO_HIGH = 0.25

CLASSES = ('hidden', 'unearned', 'restatement', 'noise')


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def steps_for(run):
    """Every step of a run, from either output layout.

    Older runs sit flat in musing_out/; newer ones under musing_out/runs/<family>/.
    Both have to be globbed or an audit silently covers half the evidence.
    """
    fs = (sorted(glob.glob(f"musing_out/runs/*/{run}-*.steps.jsonl"))
          or sorted(glob.glob(f"musing_out/{run}-*.steps.jsonl")))
    out = []
    for f in fs:
        with open(f, encoding="utf-8") as fh:
            out.append((f, [json.loads(line) for line in fh if line.strip()]))
    return out


def run_meta(run):
    """The meta block run_musing wrote for this run, or {}.

    Needed because a standard-axis method shapes a run WITHOUT labelling any
    particle: it speaks only through the extractor's standard block, and the
    extractor makes one call for the whole population. Without reading meta,
    such a method looks like it produced nothing, when in fact it produced
    everything and is simply not attributable per particle.
    """
    try:
        with open("musing_out/meta/runs.jsonl", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    except OSError:
        return {}
    for r in reversed(rows):
        if str(r.get("run_id", "")).startswith(run):
            return r
    return {}


def founding_events(steps, axis='anchor'):
    """Every (method, clause) pair at the step the clause first appeared.

    Founding events, not every particle: a commitment inherited for thirty
    steps is one idea, and counting it per step would rank methods by how long
    their commitments happened to survive rather than by what they proposed.

    Keyed on (root_id, clause) because the anchor==root invariant means a new
    clause founds a new root -- so the pair is the identity of a proposal.
    """
    seen, events = set(), {}
    for i, st in enumerate(steps):
        for p in st.get("particles", []):
            clause = p.get(axis)
            if not clause:
                continue
            key = (p.get("root_id"), clause.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            events[key] = {
                "clause": clause.strip(),
                "method": p.get("method") or "(none)",
                "root_id": p.get("root_id"),
                "first_step": i,
                "steps_alive": 0,
                "peak_mass": 0.0,
                "ever_led": False,
                "beat_null": False,
            }
    return events


def trajectories(steps, events, axis='anchor'):
    """Fill each founding event's trajectory: how the population treated it.

    Mass is summed PER ROOT, not read per particle: resampling encodes the
    posterior in multiplicity, so a root holding three copies at 0.1 carries
    0.3, and reading any one particle understates it by 3x.
    """
    by_root = {e["root_id"]: e for e in events.values()}
    for st in steps:
        mass = collections.defaultdict(float)
        for p in st.get("particles", []):
            mass[p.get("root_id")] += float(p.get("weight") or 0.0)
        if not mass:
            continue
        leader = max(mass, key=mass.get)
        # EARNED IS NOT THE SAME AS WON. `surprise` means the null outranked
        # every live commitment, so on a surprise step nothing on the board
        # beat the null. On any other step the board did -- and a root that was
        # carrying at least its fair share of the mass was part of what beat
        # it. Requiring the root to have LED would score at most one clause per
        # step as earned and file every runner-up under "the evidence never
        # supported it", which is a claim about rank, not about evidence.
        board_beat_null = not st.get("surprise")
        fair_share = 1.0 / max(1, len(mass))
        for rid, m in mass.items():
            e = by_root.get(rid)
            if e is None:
                continue
            e["steps_alive"] += 1
            e["peak_mass"] = max(e["peak_mass"], m)
            if rid == leader:
                e["ever_led"] = True
            if board_beat_null and m >= fair_share:
                e["beat_null"] = True
    return events


def echo_for(clause, steps, axis='anchor'):
    """Mean content-word overlap between a clause and the turns scored while
    it was alive. High echo means the clause reads the message back."""
    hb = bag(clause)
    if not hb:
        return 0.0
    hits = []
    for st in steps:
        if not any((p.get(axis) or "").strip().lower() == clause.strip().lower()
                   for p in st.get("particles", [])):
            continue
        act = bag(st.get("scored_action") or "")
        if act:
            hits.append(len(hb & act) / len(hb))
    return sum(hits) / len(hits) if hits else 0.0


# --------------------------------------------------------------------------
# pass 1 -- unique yield
# --------------------------------------------------------------------------

def pass1(events, baseline_stems, keyfn=None):
    """Per group: which of its founded clauses no OTHER group, and no baseline
    run, also produced.

    Canonicalised by stem, so "Define 2-Purple by pigment" and "Define 2-Purple
    by HPLC" are one idea. Without that, novelty measures phrasing.

    `keyfn` chooses the grouping. By method is the default and the finest
    attribution available. By FAMILY pools methods that share a generation
    mechanism, and the two answer different questions: two methods in one
    family that each found something the other missed are counted as two unique
    clauses by method and as two unique clauses by family, but a clause BOTH
    found is unique at family level and unique at NEITHER method level. So
    family novelty is not the sum of its methods', and pooling can only raise
    it. That is the point -- it is the measurement that survives when the
    per-method split is too thin to read.
    """
    keyfn = keyfn or (lambda e: e["method"])
    by_method = collections.defaultdict(list)
    for e in events.values():
        by_method[keyfn(e)].append(e)
    stems_of = {m: {stem(e["clause"]) for e in es} for m, es in by_method.items()}
    out = {}
    for m, es in by_method.items():
        others = set().union(*[s for k, s in stems_of.items() if k != m]) if len(stems_of) > 1 else set()
        floor = others | baseline_stems
        uniq = [e for e in es if stem(e["clause"]) not in floor]
        out[m] = {
            "total": len(es),
            "unique": uniq,
            "n_unique": len(uniq),
            "novelty": len(uniq) / len(es) if es else 0.0,
            "shared_with_baseline": sum(1 for e in es if stem(e["clause"]) in baseline_stems),
        }
    return out


# --------------------------------------------------------------------------
# pass 2 -- earned, or merely novel
# --------------------------------------------------------------------------

def classify(e, echo):
    """One of four, in priority order.

    `restatement` outranks `hidden` deliberately: a clause that reads the
    message back is not made informative by having survived, and boundness is
    the specific failure the anti-restatement rule was written against.
    `noise` outranks `unearned` because a clause that vanished in two steps
    was never tested enough for "the evidence did not support it" to mean
    anything about it.
    """
    if echo >= ECHO_HIGH or BOUND.search(e["clause"]):
        return 'restatement'
    if e["steps_alive"] < 2:
        return 'noise'
    if not e["beat_null"]:
        return 'unearned'
    if e["steps_alive"] < SURVIVAL_STEPS:
        return 'noise'
    return 'hidden'


def pass2(p1, steps, axis='anchor'):  # noqa: D401
    for m, d in p1.items():
        counts = collections.Counter()
        for e in d["unique"]:
            e["echo"] = echo_for(e["clause"], steps, axis)
            e["class"] = classify(e, e["echo"])
            counts[e["class"]] += 1
        d["classes"] = {c: counts.get(c, 0) for c in CLASSES}
        d["hidden"] = [e for e in d["unique"] if e["class"] == 'hidden']
    return p1


# --------------------------------------------------------------------------
# pass 3 -- the named-thing audit
# --------------------------------------------------------------------------

JUDGE_SYS = (
    "You check whether a claim about someone's motive is STATED in a transcript "
    "or merely INFERRED from it.\n\n"
    "For each numbered claim, answer on one line:\n"
    "  <n>: QUOTED -- <the exact line from the transcript that states it>\n"
    "or\n"
    "  <n>: NOT STATED\n\n"
    "Rules:\n"
    "- Answer QUOTED only if you can reproduce a line from the transcript that "
    "states the claim. If you cannot quote it, the answer is NOT STATED.\n"
    "- A line that states the same SUBJECT MATTER is not enough. The line must "
    "state the motive itself -- what the person is trying to achieve.\n"
    "- Do not judge whether the claim is true, likely, or flattering. Only "
    "whether the transcript says it.\n"
    "- A claim you can infer from the transcript but not quote from it is "
    "NOT STATED. That is the answer we expect most often.")


def judge(model, transcript, clauses, target):
    block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clauses))
    prompt = (f"<transcript>\n{transcript}\n</transcript>\n\n"
              f"<claims about {target}>\n{block}\n</claims about {target}>")
    raw = model.interact(prompt, system_prompt=JUDGE_SYS, temperature=0, max_tokens=2048)
    out = {}
    for m in re.finditer(r"^\s*\**\s*(\d+)\s*\**\s*[:.\)]\s*\**\s*(QUOTED|NOT\s*STATED)(.*)$",
                         raw or "", re.I | re.M):
        n = int(m.group(1))
        verdict = 'QUOTED' if m.group(2).upper().startswith('QUOTED') else 'NOT STATED'
        out[n] = (verdict, m.group(3).strip(" -:\t"))
    return out, raw


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------

def seed_floor(runs, axis='anchor', keyfn=None):
    """Same configuration, different seed: the novelty rate that means nothing.

    Two seeds of one method also produce commitments unique to each other, and
    any novelty number that does not clear that rate is measuring the sampler.
    Reported, never silently subtracted.

    MEASURED AT THE SAME GRANULARITY AS THE ROWS IT GATES. A family pools
    several methods, so it has more chances to rediscover its own stems across
    seeds and its floor is not the run's floor -- comparing a family novelty
    against a whole-run floor would be comparing two different quantities.
    Returns (overall, n_pairings, {group: floor}).
    """
    per_run = []
    for r in runs:
        loaded = steps_for(r)
        if not loaded:
            continue
        ev = founding_events(loaded[0][1], axis)
        groups = collections.defaultdict(set)
        allst = set()
        for e in ev.values():
            groups[keyfn(e) if keyfn else e["method"]].add(stem(e["clause"]))
            allst.add(stem(e["clause"]))
        per_run.append((allst, groups))
    if len(per_run) < 2:
        return None, 0, {}

    def rate(sets):
        out = []
        for i, si in enumerate(sets):
            others = set().union(*[s for j, s in enumerate(sets) if j != i]) \
                if len(sets) > 1 else set()
            if si:
                out.append(len(si - others) / len(si))
        return sum(out) / len(out) if out else None

    overall = rate([a for a, _ in per_run])
    per_group = {}
    for g in set().union(*[set(gr) for _, gr in per_run]):
        sets = [gr.get(g, set()) for _, gr in per_run]
        # a group absent from a seed has no pairing to measure
        if sum(1 for x in sets if x) >= 2:
            per_group[g] = rate(sets)
    return overall, len(per_run), per_group


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run-id prefix of the mixed-population run")
    ap.add_argument("--baseline", default=None,
                    help="run-id prefix of the same scene with --methods unset")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated run-ids of the same method on different "
                         "seeds, for the novelty noise floor")
    ap.add_argument("--axis", default="anchor", choices=("anchor", "standard"),
                    help="standard-axis methods (crystal, signpost) found no anchors; "
                         "audit them on the settling condition instead")
    ap.add_argument("--by-family", action="store_true",
                    help="pool methods that share a generation mechanism "
                         "(methods.py FAMILIES) instead of reporting each one. "
                         "Use when the per-method split is too thin to read.")
    ap.add_argument("--judge", action="store_true",
                    help="run pass 3 (the only pass that calls an LLM)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--show", action="store_true", help="print every surviving clause")
    a = ap.parse_args()

    loaded = steps_for(a.run)
    if not loaded:
        raise SystemExit(f"no steps found for {a.run}")
    path, steps = loaded[0]
    events = trajectories(steps, founding_events(steps, a.axis), a.axis)
    if not events:
        raise SystemExit(f"no {a.axis} values in {path}")

    baseline_stems = set()
    if a.baseline:
        bl = steps_for(a.baseline)
        if not bl:
            print(f"!! baseline {a.baseline} not found -- novelty is measured against "
                  f"the other methods ONLY, which overstates it", file=sys.stderr)
        else:
            baseline_stems = {stem(e["clause"])
                              for e in founding_events(bl[0][1], a.axis).values()}
    else:
        print("!! no --baseline: a commitment the default prompts already produce "
              "will be counted as a method's yield", file=sys.stderr)

    keyfn = (lambda e: family_of(e["method"])) if a.by_family else None
    p = pass2(pass1(events, baseline_stems, keyfn), steps, a.axis)

    labelled = [m for m in p if m != "(none)"]
    if not labelled:
        raise SystemExit(
            f"{path} has no method labels -- it was run without --methods, so "
            f"there is nothing to attribute. Re-run with --methods.")

    meta = run_meta(a.run)
    declared = meta.get("methods") or []
    # A standard-axis method never labels a particle, so its absence from the
    # table below is a property of where it acts, not a finding about it.
    unattributable = [k for k in declared
                      if k in METHODS and METHODS[k].axis == 'standard']

    # which methods actually appear, per group -- a family row backed by one
    # method is a family row in name only and must say so
    members = collections.defaultdict(set)
    for e in events.values():
        members[family_of(e["method"]) if a.by_family else e["method"]].add(
            e["method"])

    head = 'family' if a.by_family else 'method'
    tail = 'methods present' if a.by_family else 'predicted failure'
    print(f"\n{os.path.basename(path)}   axis={a.axis}   {len(steps)} steps"
          f"   grouped by {head}\n")
    print(f"{head:<15}{'found':>6}{'uniq':>6}{'novel':>7}"
          f"{'hidden':>8}{'unearn':>8}{'restate':>9}{'noise':>7}   {tail}")
    for m in sorted(p, key=lambda k: -p[k]["n_unique"]):
        d, c = p[m], p[m]["classes"]
        if a.by_family:
            note = ", ".join(sorted(x for x in members[m] if x))
        else:
            note = (METHODS[m].expected_failure if m in METHODS else "")[:46]
        print(f"{m:<15}{d['total']:>6}{d['n_unique']:>6}{d['novelty']:>7.2f}"
              f"{c['hidden']:>8}{c['unearned']:>8}{c['restatement']:>9}{c['noise']:>7}"
              f"   {note}")
    if a.by_family:
        thin = [m for m in p if m != "(none)" and
                len([x for x in members[m] if x]) < 2]
        if thin:
            print(f"\n  {', '.join(thin)}: one method only -- pooling changed "
                  f"nothing for these rows.")

    if unattributable:
        print(f"\nnot in the table: {', '.join(unattributable)} -- standard-axis, so "
              f"they shape the SETTLING CONDITION for the whole population through "
              f"one extractor call and label no particle.")
        print("  Their absence above is where they act, not a finding about them. "
              "To attribute one per particle, run it alone and compare the run "
              "against a baseline.")

    if a.seeds:
        floor, n, per_group = seed_floor(
            [s.strip() for s in a.seeds.split(",") if s.strip()], a.axis, keyfn)
        if floor is None:
            print("\nnoise floor: not computable (need >=2 runs)")
        else:
            print(f"\nnovelty noise floor, same config across {n} seeds: "
                  f"{floor:.2f} overall")
            beat, short = [], []
            for m in labelled:
                f = per_group.get(m)
                if f is None:
                    short.append(m)          # not present in both seeds
                elif p[m]["novelty"] > f:
                    beat.append(f"{m} ({p[m]['novelty']:.2f} vs {f:.2f})")
            for m in sorted(per_group):
                if m in labelled:
                    print(f"    {m:<15}{per_group[m]:>6.2f}")
            if beat:
                print(f"  clearing its own floor: {', '.join(beat)}")
            else:
                print("  clearing its own floor: NONE -- every novelty figure "
                      "above is within sampling variance")
            if short:
                print(f"  no floor for {', '.join(short)}: absent from one seed")
    else:
        print("\n!! no --seeds: the novelty column has no noise floor and should not "
              "be read as a ranking")

    # pass 3
    if a.judge:
        pool = {m: [e for e in p[m]["hidden"]] for m in labelled if p[m]["hidden"]}
        if not pool:
            print("\nnothing survived pass 2; pass 3 skipped")
        else:
            from agents.load_model import load_model
            model = load_model(a.model, {"model": a.model})
            transcript = "\n".join(
                st.get("scored_action") or "" for st in steps)
            target = (steps[0].get("target_agent")
                      or os.path.basename(path).split("-")[0])
            print(f"\n--- pass 3: named-thing audit ({sum(len(v) for v in pool.values())} "
                  f"clauses, {len(pool)} calls) ---")
            for m, es in pool.items():
                verdicts, _ = judge(model, transcript, [e["clause"] for e in es], target)
                for i, e in enumerate(es, start=1):
                    v, ev = verdicts.get(i, (None, ""))
                    e["judge"] = v
                    e["judge_evidence"] = ev
                n_ns = sum(1 for e in es if e.get("judge") == 'NOT STATED')
                print(f"{m:<14}{n_ns:>3}/{len(es)} NOT STATED and survived")

    if a.show:
        print("\n--- surviving clauses, in the HYPOTHESES.md record format ---")
        n = 0
        for m in sorted(p):
            for e in p[m]["hidden"]:
                n += 1
                strength = ("strong" if e["peak_mass"] >= 0.4 else
                            "medium" if e["peak_mass"] >= 0.2 else "weak")
                colour = "red" if e["ever_led"] and e["peak_mass"] >= 0.4 else "yellow"
                verdict = {'NOT STATED': 'plausible -- inferred, not stated',
                           'QUOTED': 'unsupported -- the transcript states it'}.get(
                               e.get("judge"), "plausible")
                print(f"\n**H{n:03d}** [{colour}] {e['clause']}. ({strength} | {m})")
                print(f"B: founded step {e['first_step']}, alive {e['steps_alive']} steps, "
                      f"peak mass {e['peak_mass']:.2f}, "
                      f"{'led the board' if e['ever_led'] else 'never led'}, "
                      f"echo {e['echo']:.2f}")
                if e.get("judge_evidence"):
                    print(f"F: {e['judge_evidence'][:200]}")
                print(f"V: {verdict}")
        if not n:
            print("(none)")


if __name__ == "__main__":
    main()
