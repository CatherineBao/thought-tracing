"""Does an external baseline profile produce MOTIVES the transcript never states?

The companion question to audit_methods.py, asked of arms rather than methods.
An arm is a way of priming the seeder; the three that matter are

  none     the default prompts, no prior at all
  infer    --infer-motive, a SEAT/STAKE prior read off this transcript
  profile  --character-profile, a standing record held before the conversation

and the third is only worth integrating if it beats the second. Beating `none`
shows a prior helps; beating `infer` shows an EXTERNAL prior helps, which is the
claim an integration would rest on.

THE CRITERION. A win is a commitment that names an internal motivation the
transcript never states, and that then survives contact with the evidence. Two
filters, and neither works alone:

  hidden       audit_methods.classify(): not a restatement, carried at least its
               fair share of belief mass on a step where the board beat the
               null, and still alive after SURVIVAL_STEPS.
  NOT STATED   audit_methods.judge(): an LLM cannot quote a line of transcript
               that states the claim. Stating the same SUBJECT MATTER does not
               count; the line has to state the motive.

NOT STATED alone is gamed by inventing a motive -- an unfalsifiable claim is
trivially unquotable. `hidden` alone still admits a fluent restatement the
classifier missed. The conjunction is the thing the system exists to produce,
and it is reported as a rate over commitments FOUNDED so that an arm cannot win
by proposing more.

  earned inference rate = |hidden AND NOT STATED| / |founded|

WHY THERE IS ALSO A HEAD-TO-HEAD. The rate is an absolute measure against a
classifier and a judge, both of which have their own biases; two arms can differ
on it for reasons that have nothing to do with which reads better. Pass 3b puts
the surviving clauses side by side, blind, and asks which better explains why
the person is saying and doing these things. Both controls are mandatory:

  blinding        the judge never sees which arm a clause came from.
  order swapping  every pair is judged in both orders. A pair whose verdict
                  flips with order is a TIE, not a win for whoever happened to
                  be printed first, and the flip rate is published -- a judge
                  reading position rather than content is then visible instead
                  of being laundered into a result.

AND THE SEED FLOOR. eval_motive_sep measured this corpus's between-run noise as
about as wide as the effects being chased. Same-arm-different-seed spread is
reported beside every number; a between-arm gap that does not clear it is not a
result, and this script says so rather than leaving it to the reader.

  python audit_profile.py --targets Wolf,Lyubovsky,Rovani --seeds 0,1,2
  python audit_profile.py --targets Wolf --seeds 0,1,2 --judge
  python audit_profile.py --targets Wolf --seeds 0,1,2 --judge --head-to-head
"""
import argparse
import collections
import json
import os
import random
import re

from audit_methods import (steps_for, run_meta, founding_events, trajectories,
                           echo_for, classify, judge, CLASSES)
from restatement import bag, stem

ARMS = ("none", "infer", "profile", "scram")
ARM_PREFIX = {"none": "prof_none", "infer": "prof_infer",
              "profile": "prof_prof", "scram": "prof_scram"}

# The derangement used by the scramble control: nobody receives their own
# record, and the cycle is fixed rather than random so a rerun is the same
# experiment. Wolf is handed the engineer's seat, Lyubovsky the other
# engineer's, Rovani the manager's -- three people whose stakes genuinely
# differ, which is what makes a wrong record detectable at all.
SCRAMBLE = {"Wolf": "Lyubovsky", "Lyubovsky": "Rovani", "Rovani": "Wolf"}


def run_id(arm, seed, target):
    return f"{ARM_PREFIX[arm]}_s{seed}_{target}"


# --------------------------------------------------------------------------
# passes 1 and 2 -- no LLM
# --------------------------------------------------------------------------

def collect(arm, seed, target):
    """Founded commitments for one run, each with its trajectory and class."""
    rid = run_id(arm, seed, target)
    loaded = steps_for(rid)
    if not loaded:
        return None
    path, steps = loaded[0]
    events = trajectories(steps, founding_events(steps))
    meta = run_meta(rid)
    ptext = profile_text_for(meta)
    out = []
    for e in events.values():
        e["echo"] = echo_for(e["clause"], steps)
        # Scored against the PROFILE ARM's prior for every arm, so the number is
        # comparable: the baseline arms never saw it, and their overlap is the
        # floor that vocabulary alone produces.
        e["pecho"] = profile_echo(e["clause"], ptext) if ptext else None
        e["class"] = classify(e, e["echo"])
        e["arm"], e["seed"], e["target"], e["run"] = arm, seed, target, rid
        out.append(e)
    return {"run": rid, "path": path, "steps": steps, "events": out,
            "meta": meta, "profile_text": ptext}


def profile_text_for(meta):
    """The prior block this run was actually given, rebuilt from its meta.

    Rebuilt rather than re-rendered from defaults: the roster is per-context, so
    a run traced over different sets saw a different block, and scoring against
    the wrong one would mismeasure exactly the arm under test.
    """
    if not meta.get("character_profile") or not meta.get("profile_target"):
        return ""
    import profiles as _p
    recs = _p.load_profiles(meta.get("corpus") or "")
    if not recs:
        return ""
    return _p.render_profile(recs, meta["profile_target"],
                             meta.get("profile_roster") or ())


def profile_echo(clause, ptext):
    """Share of a commitment's content words that come from the PRIOR.

    THE BLIND SPOT THIS COVERS. audit_methods.judge asks whether a claim is
    stated in the transcript, and a profile's content is by construction NOT in
    the transcript -- so a commitment that merely reads the profile back scores
    NOT STATED automatically and is indistinguishable from an inferred motive.
    The earned-inference rate cannot see the difference; this can.

    It is the exact analogue of restatement.echo, pointed at the prior instead
    of the thread. High profile-echo means the arm is parroting what it was
    handed, which is a different failure from restating the transcript and is
    invisible to every other number here.
    """
    hb = bag(clause)
    pb = bag(ptext)
    if not hb or not pb:
        return 0.0
    return len(hb & pb) / len(hb)


def blind_dump(bundles, targets, arms, path, seed=0):
    """Write the arms' commitments with their identities stripped, plus a key.

    WHY BLINDED. An auditor told which set came from the profile arm will
    rationalise toward whatever it expects that arm to do -- the same failure
    the head-to-head's order swap exists to remove, one level up. Labels are
    randomised PER TARGET, so an auditor cannot learn "set A is always the
    profile arm" from the first target and carry it to the rest.

    The transcript travels with each target because the question is not whether
    a commitment sounds insightful. It is whether the transcript already says
    it -- and that can only be judged against the transcript the run actually
    scored.
    """
    rng = random.Random(seed)
    out, key = {}, {}
    letters = "ABCDEFGH"
    for t in targets:
        present = [a_ for a_ in arms if any(k[0] == a_ and k[1] == t for k in bundles)]
        if not present:
            continue
        shuffled = list(present)
        rng.shuffle(shuffled)
        sets, transcript = {}, ""
        for i, arm in enumerate(shuffled):
            label = f"set-{letters[i]}"
            key[f"{t}/{label}"] = arm
            evs = [e for s in sorted({k[2] for k in bundles if k[0] == arm and k[1] == t})
                   for e in bundles[(arm, t, s)]["events"]]
            if not transcript:
                first_seed = sorted({k[2] for k in bundles
                                     if k[0] == arm and k[1] == t})[0]
                transcript = transcript_for(bundles[(arm, t, first_seed)])
            sets[label] = [
                {"commitment": e["clause"],
                 "turns_alive": e["steps_alive"],
                 "peak_belief_mass": round(e["peak_mass"], 3)}
                for e in sorted(evs, key=lambda x: -x["peak_mass"])
            ]
        out[t] = {"transcript_the_system_scored": transcript, "sets": sets}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    keypath = path.replace(".json", "") + ".key.json"
    with open(keypath, "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=1)
    print(f"wrote {path} (blinded) and {keypath} (key -- do not show the auditor)")
    return out, key


def pecho_matrix(bundles, targets, arms, profiles_by_token):
    """For each arm and traced target, overlap against EVERY profile.

    This is the question the scramble asks, in one table. Reading down a column
    for the scramble arm:

      commitments echo the SUPPLIED profile   -> the prior steers the output.
        Whoever's record you hand over is who the model describes, so the
        profile is not carrying information about the person being traced and
        an integration would be shipping a very expensive prompt-nudge.

      commitments echo the TRUE person        -> the transcript is dominating
        and the prior is largely inert, wrong or right.

      neither moves                           -> the prior is not reaching the
        commitments at all, and the profile arm's earlier result was noise.

    The diagonal of the unscrambled profile arm is the reference: that is how
    much a CORRECT record gets echoed, and the scramble arm's supplied-profile
    number has to be read against it, not against zero.
    """
    out = {}
    for arm in arms:
        for t in targets:
            evs = [e for s in (bundles.get((arm, t, s)) for s in set(
                       k[2] for k in bundles if k[0] == arm and k[1] == t))
                   if s for e in s["events"]]
            if not evs:
                continue
            row = {}
            for token, ptext in profiles_by_token.items():
                vals = [profile_echo(e["clause"], ptext) for e in evs]
                row[token] = sum(vals) / len(vals) if vals else 0.0
            out[(arm, t)] = row
    return out


def novel_against(events, baseline_stems):
    """Clauses whose STEM no baseline run produced.

    Stem-canonicalised for the same reason audit_methods.pass1 is: without it,
    "keep the accounts confident" and "keep accounts confident" count as a new
    idea, and novelty becomes a measure of phrasing.
    """
    return [e for e in events if stem(e["clause"]) not in baseline_stems]


def rates(events):
    """Class breakdown plus the headline, over commitments FOUNDED."""
    n = len(events)
    counts = collections.Counter(e["class"] for e in events)
    judged = [e for e in events if e.get("judge")]
    earned = [e for e in events
              if e["class"] == "hidden" and e.get("judge") == "NOT STATED"]
    return {
        "founded": n,
        **{c: counts.get(c, 0) for c in CLASSES},
        "hidden_rate": counts.get("hidden", 0) / n if n else 0.0,
        "n_judged": len(judged),
        "earned": len(earned),
        # None rather than 0 when nothing was judged: an unjudged arm has no
        # earned-inference rate, and printing 0.000 would read as a measured
        # failure instead of a missing measurement.
        "earned_rate": (len(earned) / n if n else 0.0) if judged else None,
    }


# --------------------------------------------------------------------------
# pass 3b -- blind head-to-head
# --------------------------------------------------------------------------

H2H_SYS = (
    "You are given a transcript and two competing accounts of why one person in "
    "it is saying and doing what they do.\n\n"
    "Answer on one line:\n"
    "  WINNER: A   or   WINNER: B   or   WINNER: TIE\n"
    "then one line:\n"
    "  WHY: <one sentence>\n\n"
    "Rules:\n"
    "- Prefer the account that gives a REASON the person would act this way. An "
    "account that restates what they did, or names the topic they were "
    "discussing, is worse -- however accurate it is.\n"
    "- Prefer the account that would still make sense if this same person "
    "appeared in a different conversation next week.\n"
    "- An account that the transcript flatly contradicts loses regardless.\n"
    "- Do not prefer the longer, more specific or more confident wording. "
    "Specificity about the TOPIC is not insight about the PERSON.\n"
    "- TIE is a real answer. Use it when neither account explains more than the "
    "other.")


def head_to_head(model, transcript, target, a_clause, b_clause):
    """One pairing, judged in BOTH orders.

    Returns 'A', 'B', 'TIE' (from the caller's perspective, where A is always
    a_clause) and whether the two orders disagreed. Order is the strongest
    nuisance variable in a pairwise LLM judgement, and swapping is the cheapest
    control that removes it.
    """
    def ask(first, second):
        prompt = (f"<transcript>\n{transcript}\n</transcript>\n\n"
                  f"<account A of why {target} acts as they do>\n{first}\n</account A>\n\n"
                  f"<account B of why {target} acts as they do>\n{second}\n</account B>")
        raw = model.interact(prompt, system_prompt=H2H_SYS, temperature=0, max_tokens=256)
        m = re.search(r"WINNER:\s*\**\s*(A|B|TIE)\b", raw or "", re.I)
        return (m.group(1).upper() if m else "TIE"), (raw or "")

    fwd, raw1 = ask(a_clause, b_clause)
    # swapped: a 'A' verdict here is a vote for b_clause
    rev, raw2 = ask(b_clause, a_clause)
    rev_for_a = {"A": "B", "B": "A", "TIE": "TIE"}[rev]
    if fwd == rev_for_a:
        return fwd, False, (raw1, raw2)
    return "TIE", True, (raw1, raw2)


_CORPUS_CACHE = {}


def transcript_for(bundle):
    """THE WHOLE CONTEXT THE TRACER READ -- not the per-step scored actions.

    Corrected after a measured failure. The first version joined each step's
    `scored_action`, on the reasoning that the judge should see what the tracer
    saw. That was wrong: a scored_action is the single turn being weighed at
    that step, and the tracer reads the entire stitched context. On the
    bloomfield color thread the join came to 4,535 of 33,032 characters -- 14%
    -- and it silently dropped four speakers who are in the conversation.

    The damage was not subtle. A blind auditor asked "is this stated in the
    transcript?" ruled `Validate Patadia's resource relevance` and `Adopt
    Saxena's storage proposal` to be claims about people who do not exist, and
    made "hallucinated entities" its headline discriminator between arms. Both
    are in the record: Patadia posts a resource page, Saxena argues about
    storing point clouds. The same truncation inflates every arm's NOT STATED
    rate, because a claim absent from 14% of a transcript is trivially
    unquotable.

    Rebuilt from the corpus by the run's own `set_ids`, which is what the
    driver stitched and handed to the tracer. Falls back to the scored actions
    only when the meta cannot identify the sets, and says nothing either way --
    the caller cannot act on the difference, but a silent empty transcript
    would make every claim NOT STATED and look like a result.
    """
    meta = bundle.get("meta") or {}
    corpus, set_ids = meta.get("corpus"), meta.get("set_ids")
    if corpus and set_ids:
        if corpus not in _CORPUS_CACHE:
            import run_musing as _rm
            _CORPUS_CACHE[corpus] = _rm.load_corpus(corpus)
        by_id = _CORPUS_CACHE[corpus]["by_id"]
        parts = [by_id[s]["full_context"] for s in set_ids if s in by_id]
        if parts:
            return "\n".join(parts)
    return "\n".join(st.get("scored_action") or "" for st in bundle["steps"])


# --------------------------------------------------------------------------

def spread(values):
    """Across-seed spread: the floor a between-arm gap has to clear."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    return max(vals) - min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="Wolf,Lyubovsky,Rovani")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--baseline", default="none",
                    help="arm whose commitments do not count as any arm's yield")
    ap.add_argument("--judge", action="store_true", help="pass 3a (LLM)")
    ap.add_argument("--head-to-head", action="store_true", help="pass 3b (LLM)")
    ap.add_argument("--vs", default="none", help="arm the profile arm is paired against in 3b")
    ap.add_argument("--pairs", type=int, default=6, help="max pairings per target/seed")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--json", default=None, help="write the full record here")
    ap.add_argument("--dump-blind", default=None, metavar="PATH",
                    help="write the arms' commitments with identities stripped, for a "
                         "qualitative auditor, plus a separate key file")
    a = ap.parse_args()

    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    bundles, missing = {}, []
    for arm in arms:
        for t in targets:
            for s in seeds:
                b = collect(arm, s, t)
                if b is None:
                    missing.append(run_id(arm, s, t))
                else:
                    bundles[(arm, t, s)] = b
    if missing:
        print(f"[!] {len(missing)} run(s) not on disk, skipped: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if not bundles:
        raise SystemExit("no runs found -- launch them with eval_profile.py first")

    # pass 1: the baseline arm's stems are nobody's yield
    base_stems = collections.defaultdict(set)
    for (arm, t, s), b in bundles.items():
        if arm == a.baseline:
            for e in b["events"]:
                base_stems[t].add(stem(e["clause"]))

    for (arm, t, s), b in bundles.items():
        b["novel"] = (b["events"] if arm == a.baseline
                      else novel_against(b["events"], base_stems[t]))

    # Score EVERY arm against the profile arm's prior, so the baseline arms
    # supply the floor that shared vocabulary produces on its own. Without that
    # floor the profile arm's overlap is a number with nothing to beat.
    ptext_for = {}
    for (arm, t, s), b in bundles.items():
        if arm == "profile" and b.get("profile_text"):
            ptext_for.setdefault(t, b["profile_text"])
    for (arm, t, s), b in bundles.items():
        pt = ptext_for.get(t)
        for e in b["events"]:
            e["pecho"] = profile_echo(e["clause"], pt) if pt else None

    if a.dump_blind:
        blind_dump(bundles, targets, arms, a.dump_blind)

    model = None
    if a.judge or a.head_to_head:
        from agents.load_model import load_model
        model = load_model(a.model, {"model": a.model})

    # pass 3a: judge the hidden survivors
    if a.judge:
        total = sum(len([e for e in b["novel"] if e["class"] == "hidden"])
                    for b in bundles.values())
        print(f"\n--- pass 3a: stated-or-inferred ({total} clauses, {len(bundles)} calls) ---")
        for (arm, t, s), b in sorted(bundles.items()):
            es = [e for e in b["novel"] if e["class"] == "hidden"]
            if not es:
                continue
            verdicts, _ = judge(model, transcript_for(b), [e["clause"] for e in es], t)
            for i, e in enumerate(es, start=1):
                v, ev = verdicts.get(i, (None, ""))
                e["judge"], e["judge_evidence"] = v, ev
            ns = sum(1 for e in es if e.get("judge") == "NOT STATED")
            print(f"  {b['run']:<28}{ns:>3}/{len(es)} NOT STATED")

    # ---- the table ----
    print(f"\n{'arm':<10}{'target':<12}{'founded':>8}{'hidden':>8}{'restmt':>8}"
          f"{'judged':>8}{'earned':>8}{'rate':>8}{'p-echo':>8}   across-seed spread")
    summary = {}
    for arm in arms:
        for t in targets:
            per_seed = []
            for s in seeds:
                b = bundles.get((arm, t, s))
                if b:
                    per_seed.append(rates(b["novel"]))
            if not per_seed:
                continue
            agg = {k: sum(r[k] for r in per_seed) for k in
                   ("founded", "hidden", "restatement", "n_judged", "earned")}
            rate = (agg["earned"] / agg["founded"]
                    if agg["founded"] and agg["n_judged"] else None)
            sp = spread([r["earned_rate"] for r in per_seed]) if a.judge \
                else spread([r["hidden_rate"] for r in per_seed])
            summary[(arm, t)] = {"rate": rate, "spread": sp, **agg}
            rate_s = f"{rate:>8.3f}" if rate is not None else f"{'--':>8}"
            sp_s = f"{sp:.3f}" if sp is not None else "--"
            pv = [e["pecho"] for s in seeds
                  for e in (bundles.get((arm, t, s)) or {"novel": []})["novel"]
                  if e.get("pecho") is not None]
            pe = sum(pv) / len(pv) if pv else None
            pe_s = f"{pe:>8.3f}" if pe is not None else f"{'--':>8}"
            summary[(arm, t)]["pecho"] = pe
            print(f"{arm:<10}{t:<12}{agg['founded']:>8}{agg['hidden']:>8}"
                  f"{agg['restatement']:>8}{agg['n_judged']:>8}{agg['earned']:>8}"
                  f"{rate_s}{pe_s}   {sp_s}")

    # ---- the 3-way scramble matrix ----
    import profiles as _profiles
    recs = _profiles.load_profiles("bloomfield")
    ptexts = {}
    if recs:
        rosters = {}
        for (arm, t, s), b in bundles.items():
            rosters.setdefault(t, (b["meta"].get("profile_roster") or []))
        allspk = sorted({x for v in rosters.values() for x in v} | set(targets))
        for tok in targets:
            ptexts[tok] = _profiles.render_profile(recs, tok, allspk)
    if ptexts and any(k[0] == "scram" for k in bundles):
        mat = pecho_matrix(bundles, targets, arms, ptexts)
        print("\np-echo against EACH profile (rows: arm/traced target, cols: whose profile)")
        print(f"  {'arm/traced':<22}" + "".join(f"{('~' + c):>12}" for c in targets)
              + f"{'supplied':>11}{'own':>8}")
        for arm in arms:
            for t in targets:
                row = mat.get((arm, t))
                if not row:
                    continue
                supplied = SCRAMBLE.get(t) if arm == "scram" else (t if arm == "profile" else None)
                sup = f"{row.get(supplied, 0):.3f}" if supplied else "--"
                print(f"  {arm + '/' + t:<22}"
                      + "".join(f"{row.get(c, 0):>12.3f}" for c in targets)
                      + f"{sup:>11}{row.get(t, 0):>8.3f}")
        print("  scram rows: 'supplied' is the WRONG record handed over, 'own' is the "
              "true person.\n  supplied > own means the prior steers the output "
              "rather than describing the person.")

    # ---- does any gap clear the floor? ----
    if a.judge:
        print("\nbetween-arm gaps, against the across-seed floor:")
        for t in targets:
            base = summary.get((a.baseline, t), {}).get("rate")
            for arm in arms:
                if arm == a.baseline:
                    continue
                r = summary.get((arm, t), {}).get("rate")
                if r is None or base is None:
                    continue
                gap = r - base
                spreads = [summary.get((arm, t), {}).get("spread"),
                           summary.get((a.baseline, t), {}).get("spread")]
                measured = [x for x in spreads if x is not None]
                if not measured:
                    # ONE SEED MEASURES NOTHING. Treating an unmeasured floor as
                    # a floor of zero makes every gap "significant", which is
                    # precisely how this corpus has produced invalid numbers
                    # before -- eval_motive_sep found the same person on two
                    # seeds separating about as far as the two real parties did.
                    print(f"  {t:<12}{arm:>8} vs {a.baseline:<8}{gap:+.3f}  "
                          f"(NO FLOOR -- 1 seed)  NOT INTERPRETABLE, run >=3 seeds")
                    continue
                floor = max(measured)
                verdict = ("clears the floor" if abs(gap) > floor
                           else "INSIDE THE NOISE -- not a result")
                print(f"  {t:<12}{arm:>8} vs {a.baseline:<8}{gap:+.3f}  "
                      f"(floor {floor:.3f})  {verdict}")

    # ---- pass 3b ----
    h2h_rows = []
    if a.head_to_head:
        rng = random.Random(0)
        print(f"\n--- pass 3b: blind head-to-head, profile vs {a.vs} "
              f"(each pair judged in both orders) ---")
        for t in targets:
            rec = collections.Counter()
            flips = 0
            for s in seeds:
                pb, nb = bundles.get(("profile", t, s)), bundles.get((a.vs, t, s))
                if not pb or not nb:
                    continue
                pa = [e for e in pb["novel"] if e["class"] == "hidden"]
                na = [e for e in nb["novel"] if e["class"] == "hidden"]
                if not pa or not na:
                    continue
                rng.shuffle(pa)
                rng.shuffle(na)
                for pe, ne in list(zip(pa, na))[:a.pairs]:
                    w, flipped, _ = head_to_head(model, transcript_for(pb), t,
                                                 pe["clause"], ne["clause"])
                    rec[{"A": "profile", "B": a.vs, "TIE": "tie"}[w]] += 1
                    flips += 1 if flipped else 0
                    h2h_rows.append({"target": t, "seed": s, "winner": w,
                                     "order_flip": flipped,
                                     "profile_clause": pe["clause"],
                                     "other_clause": ne["clause"]})
            n = sum(rec.values())
            if not n:
                print(f"  {t:<12}no comparable pairs")
                continue
            fr = flips / n
            note = "  [!] judge is reading POSITION -- record void" if fr > 0.4 else ""
            print(f"  {t:<12}profile {rec['profile']:>3} | {a.vs} {rec[a.vs]:>3} | "
                  f"tie {rec['tie']:>3}   order-flip {fr:.0%}{note}")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({
                "summary": {f"{k[0]}/{k[1]}": v for k, v in summary.items()},
                "events": [{kk: vv for kk, vv in e.items() if kk != "steps"}
                           for b in bundles.values() for e in b["novel"]],
                "head_to_head": h2h_rows,
            }, fh, indent=1)
        print(f"\nwrote {a.json}")

    print("\nearned inference rate = (hidden AND NOT STATED) / founded. Higher is "
          "better.\nA gap inside the across-seed spread is not a result.")
    print("p-echo = overlap with the PROFILE text. The judge cannot see this: a "
          "commitment\nthat reads the prior back is unquotable from the "
          "transcript and scores NOT STATED\nregardless. A profile arm well "
          "above the baseline arms' p-echo is parroting, not inferring.")


if __name__ == "__main__":
    main()
