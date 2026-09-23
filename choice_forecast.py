"""Forecast the held-out choices, and score everything against what a careful
reader would already have said.

    python choice_forecast.py --corpora bloomfield,boeing,oppenheimer
    python choice_forecast.py --fake      # pipeline only; every arm must sit at chance

THE BAR IS NOT CHANCE. Beating chance on a four-option choice means almost
nothing: a reader who knows only that engineers mostly hold and managers mostly
escalate is already well above it. So the scored quantity here is LIFT OVER A
HUMAN-OBVIOUS FORECAST -- the same model, the same transcript, the same options,
and no motive. A motive earns its place only by being right where the obvious
read is wrong. This is the discipline PRECHECKS says was missing when three
culture detectors were retired: each raised the real score and the control with
it.

FOUR BASELINES, EACH ANSWERING A DIFFERENT OBJECTION.

  habit    this person's own action mix on the EARLIER choices. Answers "you
           have discovered that people repeat themselves".
  role     what everybody ELSE in the same channel did on the earlier choices.
           Answers "you have discovered what people in this seat do". Derived
           from the record, never from data/musing/*_profiles.json, which is
           marked FABRICATED in its own _README.
  obvious  the model with the transcript and the options and no motive. The
           bar every hypothesis is scored against.
  majority the single most common action in the EARLIER choices, corpus-wide.
           Added after the extraction showed HOLD is ~70% of every corpus and
           before any model forecast was read: on a task that skewed, a
           constant is a harder floor than either frequency baseline, and
           leaving it out would have flattered everything above it. It uses
           dev counts only.
  blind    the options and NOTHING else -- no transcript, no person, no
           situation. This one is not a baseline to beat, it is a TRIPWIRE. The
           extractor saw the outcome when it wrote the alternatives, so if the
           option set alone predicts the answer, the alternatives leak and every
           other number in this report is void.

THE CONTROL ARM IS A SWAPPED PERSON. Each motive is also run against a
DIFFERENT person's held-out choices, with everything else identical. A motive
that forecasts anybody's choices is reading the transcript, not the person, and
the swap arm is what separates those. It also supplies the honest null for the
surprise list: how many "the obvious read failed and a hypothesis succeeded"
events a motive that cannot possibly apply still produces.

PAIRED STATISTICS, NOT TWO ACCURACIES. Every arm forecasts the same choice
points, so the comparison is within-item: McNemar's exact test on the
discordant pairs. Two independent accuracy numbers thrown at a t-test would
ignore that the hard items are hard for both, which is most of the variance.
Thirteen generators times a cast of people is a lot of tests, so q-values are
Benjamini-Hochberg and the family-level claim carries a permutation p.

THE OPTION ORDER IS FIXED PER CHOICE POINT AND SHARED ACROSS ARMS. Shuffled by
a hash of cp_id, so the taken option is not at a predictable position, and
identical for every arm, so no arm is advantaged by ordering.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics as st
import subprocess

from choice_llm import CachedModel, FakeModel
from choice_motives import MIN_DEV, MIN_TEST, load_motives
from choice_points import ACTIONS, Context, load as load_points

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "choice_forecast.json")
SURPRISES = os.path.join(HERE, "SURPRISES.md")

SYSTEM = (
    "You forecast what one named person will do at a decision point in a work "
    "conversation. You are shown what happened UP TO that moment and never "
    "past it.\n\nAnswer with exactly two lines:\nANSWER: <one option label>\n"
    "WHY: <= 20 words"
)

BLIND_SYSTEM = (
    "You are shown a list of options somebody chose between. You are shown "
    "nothing else -- no conversation, no person, no situation.\n\nGuess which "
    "was taken. Answer with exactly two lines:\nANSWER: <one option label>\n"
    "WHY: <= 20 words"
)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def options_block(cp):
    """The alternatives in a fixed, cp-specific order shared by every arm."""
    order = sorted(cp["alternatives"],
                   key=lambda a: hashlib.sha256(
                       (cp["cp_id"] + a["action"]).encode()).hexdigest())
    return order, "\n".join(
        f"  [{a['action']}] {a['gist']}  (costs them: {a['gives_up']})" for a in order)


def forecast_prompt(cp, ctx, motive=None, context_turns=40):
    _, opts = options_block(cp)
    belief = ""
    if motive:
        belief = (f"\nWhat we believe about {cp['person']}, from their earlier "
                  f"choices:\n  {motive['motive']}\n"
                  + (f"  (grounds: {motive['because']})\n" if motive.get("because") else ""))
    return (f"Conversation so far in #{cp['channel']}:\n\n"
            f"{ctx.prefix(cp, context_turns) or '(nothing earlier in this channel)'}\n\n"
            f"{cp['person']} is now facing this: {cp['situation']}\n"
            f"{belief}\nOptions open to them:\n{opts}\n\n"
            f"Which does {cp['person']} take?")


def blind_prompt(cp):
    _, opts = options_block(cp)
    return f"Options:\n{opts}\n\nWhich was taken?"


ANSWER = re.compile(r"ANSWER:\s*\[?([A-Za-z_\- ]+)\]?", re.I)


def parse_answer(raw, allowed):
    """The label the reply commits to, or None.

    Matched against the options actually offered rather than the whole
    taxonomy: a reply naming an action that was not on the list is not a
    forecast, and scoring it as a miss would credit the arm with an attempt it
    did not make. Those are counted separately as `unparsed`.
    """
    if not raw:
        return None
    m = ANSWER.search(raw)
    cand = (m.group(1) if m else raw).strip().upper().replace("-", "_").replace(" ", "_")
    if cand in allowed:
        return cand
    for a in allowed:                       # a reply that only mentions one option
        if re.search(rf"\b{a}\b", (raw or "").upper()):
            return a
    return None


# --------------------------------------------------------------------------
# baselines that need no model
# --------------------------------------------------------------------------

def majority_action(dev_points):
    """The commonest action overall, from the earlier choices.

    A constant. On a corpus where one action is 70% of the record this is the
    real floor, and it costs nothing -- which is exactly why it has to be in
    the table rather than left for a reader to work out.
    """
    c = collections.Counter(cp["actual"] for cp in dev_points)
    return c.most_common(1)[0][0] if c else None


def habit_table(dev_points):
    per = collections.defaultdict(collections.Counter)
    for cp in dev_points:
        per[(cp["corpus"], cp["person"])][cp["actual"]] += 1
    return per


def role_table(dev_points):
    """Action mix of everybody ELSE in the same channel, on the earlier choices.

    'Else' matters: including the person collapses this into habit, and the two
    baselines would then rise and fall together and neither would be a control
    on the other.

    Subtracted PER CHANNEL, not per person. habit_table counts a person across
    every channel they appear in, so taking it off one channel's counter
    removes choices they made somewhere else -- which can drive a count
    negative and hand `role` an action nobody in that channel ever took.
    """
    chan = collections.defaultdict(collections.Counter)
    chan_person = collections.defaultdict(collections.Counter)
    for cp in dev_points:
        chan[(cp["corpus"], cp["channel"])][cp["actual"]] += 1
        chan_person[(cp["corpus"], cp["channel"], cp["person"])][cp["actual"]] += 1
    return chan, chan_person


def others_in_channel(chan, chan_person, corpus, channel, person):
    """That channel's action mix with this person's own choices removed."""
    rest = collections.Counter(chan[(corpus, channel)])
    rest.subtract(chan_person[(corpus, channel, person)])
    return collections.Counter({k: v for k, v in rest.items() if v > 0})


def pick_from_counter(counter, allowed, tiebreak):
    """Most frequent allowed action; ties broken by a fixed hash, never by
    dict order -- which would make the baseline depend on insertion order."""
    if not allowed:
        return None
    best = max(allowed, key=lambda a: (counter.get(a, 0),
                                       hashlib.sha256((tiebreak + a).encode()).hexdigest()))
    return best


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def mcnemar(arm, ref):
    """Exact paired test. Returns (b, c, p) where b = arm right / ref wrong.

    One-sided on b > c, because the question is whether the motive BEATS the
    obvious read, not whether the two differ. A two-sided p would let a motive
    that is reliably worse register as a finding.
    """
    b = sum(1 for x, y in zip(arm, ref) if x and not y)
    c = sum(1 for x, y in zip(arm, ref) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(b, n + 1)) / (2.0 ** n)
    return b, c, min(1.0, p)


def benjamini_hochberg(pvals):
    """q-values. Thirteen generators times a cast is enough tests that an
    uncorrected 0.05 would be expected to produce findings from noise alone."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    q = [1.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        k = n - rank + 1
        prev = min(prev, pvals[i] * n / k)
        q[i] = prev
    return q


def permutation_p_clustered(clusters, iters=20000, seed=0):
    """Permutation p where the exchangeable unit is a CHOICE POINT, not a forecast.

    The pooled comparison reuses one `obvious` forecast as the reference for
    all thirteen of a person's hypotheses, so the thirteen discordant pairs on
    one choice point are not thirteen independent pieces of evidence -- they
    share their reference and most of their prompt. Flipping them independently
    (which a plain McNemar does implicitly) makes the pooled p anti-conservative
    by roughly the cluster size.

    Here a coin is flipped per choice point and applied to every pair inside it,
    which is the dependency the design actually has. This is the headline p;
    the unclustered McNemar is reported beside it and labelled.
    """
    rng = random.Random(seed)
    obs = sum(x - y for pairs in clusters.values() for x, y in pairs)
    keys = list(clusters)
    hits = 0
    for _ in range(iters):
        tot = 0
        for k in keys:
            flip = 1 if rng.random() < 0.5 else -1
            tot += flip * sum(x - y for x, y in clusters[k])
        if tot >= obs:
            hits += 1
    return (hits + 1) / (iters + 1)


def permutation_p(arm, ref, iters=20000, seed=0):
    """Family-level p by flipping which member of each pair belongs to which arm.

    The repo's standing rule: pool the events, permute the labels, report a p.
    Under the null the two arms are exchangeable within a choice point, so a
    coin flip per item is the right permutation.
    """
    rng = random.Random(seed)
    obs = sum(arm) - sum(ref)
    pairs = list(zip(arm, ref))
    hits = 0
    for _ in range(iters):
        tot = 0
        for x, y in pairs:
            if x != y:
                tot += 1 if rng.random() < 0.5 else -1
        if tot >= obs:
            hits += 1
    return (hits + 1) / (iters + 1)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def build_jobs(test_points, ctx, motives, context_turns, swap_seed=0):
    """Every (arm, choice point) that needs a model call, plus the swap pairing.

    The swap pairs each motive with a different person IN THE SAME CORPUS, so
    the control differs in the person and not in the setting. Pairing is by
    rotation over a sorted cast, so it is deterministic and every person is
    used exactly once as somebody else's control.
    """
    by_person = collections.defaultdict(list)
    for cp in test_points:
        by_person[(cp["corpus"], cp["person"])].append(cp)

    by_corpus = collections.defaultdict(list)
    for key in sorted(by_person):
        by_corpus[key[0]].append(key)
    swap_of = {}
    for corpus, keys in by_corpus.items():
        if len(keys) < 2:
            continue
        for i, k in enumerate(keys):
            swap_of[k] = keys[(i + 1) % len(keys)]

    jobs = []
    for cp in test_points:
        allowed = {a["action"] for a in cp["alternatives"]}
        jobs.append(("obvious", None, cp, forecast_prompt(cp, ctx, None, context_turns),
                     SYSTEM, allowed))
        jobs.append(("blind", None, cp, blind_prompt(cp), BLIND_SYSTEM, allowed))

    for h in motives:
        key = (h["corpus"], h["person"])
        for cp in by_person.get(key, []):
            allowed = {a["action"] for a in cp["alternatives"]}
            jobs.append(("hyp", h, cp, forecast_prompt(cp, ctx, h, context_turns),
                         SYSTEM, allowed))
        other = swap_of.get(key)
        if other:
            for cp in by_person.get(other, []):
                allowed = {a["action"] for a in cp["alternatives"]}
                jobs.append(("swap", h, cp, forecast_prompt(cp, ctx, h, context_turns),
                             SYSTEM, allowed))
    return jobs, swap_of


def cap_test(test, max_per_person):
    """The EARLIEST n held-out choices per person, or all of them.

    Chronological, never sampled: picking which held-out choices to score after
    the extraction exists is a selection on the data, and taking the earliest
    is the one rule that cannot be tuned. The cap exists because the forecast
    leg is quadratic in the cast -- every motive is also run against another
    person's choices -- and an uncapped run is ~7,000 calls for no more
    statistical power than a capped one.
    """
    if not max_per_person:
        return test
    per = collections.defaultdict(list)
    for cp in sorted(test, key=lambda c: (c.get("date") or "", c["at_turn"])):
        per[(cp["corpus"], cp["person"])].append(cp)
    return [cp for v in per.values() for cp in v[:max_per_person]]


def cap_cast(motives, test, max_cast):
    """Keep the people with the most held-out choices, ties broken by name.

    Chosen on COUNT, which is known before any forecast is made, so this is not
    a selection on how well anybody scores.
    """
    if not max_cast:
        return motives
    counts = collections.Counter((cp["corpus"], cp["person"]) for cp in test)
    keep = {k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_cast]}
    return [h for h in motives if (h["corpus"], h["person"]) in keep]


def run(corpora, model, context_turns=40, out=OUT, iters=20000,
        max_test_per_person=None, max_cast=None, chunk=32):
    points = [cp for c in corpora for cp in load_points(c)]
    motives = [h for h in load_motives() if h["corpus"] in corpora]
    if not motives:
        raise SystemExit("no motives on disk; run choice_motives.py first")

    cast = {(h["corpus"], h["person"]) for h in motives}
    dev = [cp for cp in points if cp["side"] == "dev"]
    test = [cp for cp in points if cp["side"] == "test"
            and (cp["corpus"], cp["person"]) in cast]
    if not test:
        raise SystemExit("no held-out choices for anybody in the cast")

    test = cap_test(test, max_test_per_person)
    motives = cap_cast(motives, test, max_cast)
    cast = {(h["corpus"], h["person"]) for h in motives}
    test = [cp for cp in test if (cp["corpus"], cp["person"]) in cast]

    ctx = Context(corpora)
    jobs, swap_of = build_jobs(test, ctx, motives, context_turns)
    print(f"  {len(jobs)} forecasts over {len(test)} held-out choices, "
          f"cast of {len(cast)}, {len(motives)} motives")
    raws = model.batch_interact([j[3] for j in jobs], system_prompts=[j[4] for j in jobs],
                                temperature=0, max_tokens=120, chunk=chunk)

    # arm -> cp_id -> predicted label
    pred = collections.defaultdict(dict)
    unparsed = collections.Counter()
    for (kind, h, cp, _, _, allowed), raw in zip(jobs, raws):
        arm = kind if h is None else f"{kind}:{h['hyp_id']}"
        got = parse_answer(raw, allowed)
        if got is None:
            unparsed[arm] += 1
        pred[arm][cp["cp_id"]] = got

    # the two model-free baselines
    habit = habit_table(dev)
    chan, chan_person = role_table(dev)
    mode = majority_action(dev)
    for cp in test:
        allowed = [a["action"] for a in cp["alternatives"]]
        key = (cp["corpus"], cp["person"])
        pred["habit"][cp["cp_id"]] = pick_from_counter(habit[key], allowed, cp["cp_id"])
        others = others_in_channel(chan, chan_person, cp["corpus"], cp["channel"],
                                   cp["person"])
        pred["role"][cp["cp_id"]] = pick_from_counter(others, allowed, cp["cp_id"])
        # A constant, offered only where it is on the menu. Where it is not,
        # it abstains rather than guessing -- an abstention scores as a miss,
        # which is the honest cost of being a constant.
        pred["majority"][cp["cp_id"]] = mode if mode in allowed else None

    truth = {cp["cp_id"]: cp["actual"] for cp in test}
    order = [cp["cp_id"] for cp in sorted(test, key=lambda c: c["cp_id"])]

    def vec(arm, ids=None):
        ids = ids or order
        return [1 if pred[arm].get(i) == truth[i] else 0 for i in ids]

    chance = st.mean(1.0 / len(cp["alternatives"]) for cp in test)
    ref = vec("obvious")

    summary = {}
    for arm in ("majority", "habit", "role", "obvious", "blind"):
        v = vec(arm)
        b, c, p = mcnemar(v, ref)
        summary[arm] = {"n": len(v), "accuracy": round(st.mean(v), 4),
                        "correct": sum(v), "unparsed": unparsed.get(arm, 0),
                        "vs_obvious": {"better": b, "worse": c, "p": round(p, 5)}}

    # per-hypothesis, scored only on its own person's held-out choices
    rows, pvals = [], []
    for h in motives:
        ids = sorted(cp["cp_id"] for cp in test
                     if (cp["corpus"], cp["person"]) == (h["corpus"], h["person"]))
        if not ids:
            continue
        real = vec(f"hyp:{h['hyp_id']}", ids)
        base = vec("obvious", ids)
        b, c, p = mcnemar(real, base)
        swap_ids = sorted(cp["cp_id"] for cp in test
                          if (cp["corpus"], cp["person"]) == swap_of.get(
                              (h["corpus"], h["person"]), (None, None)))
        swap = vec(f"swap:{h['hyp_id']}", swap_ids) if swap_ids else []
        swap_base = vec("obvious", swap_ids) if swap_ids else []
        rows.append({
            "hyp_id": h["hyp_id"], "corpus": h["corpus"], "person": h["person"],
            "method": h["method"], "family": h["family"], "motive": h["motive"],
            "n": len(ids), "accuracy": round(st.mean(real), 4),
            "obvious_accuracy": round(st.mean(base), 4),
            "lift": round(st.mean(real) - st.mean(base), 4),
            "better": b, "worse": c, "p": round(p, 5),
            "swap_n": len(swap),
            "swap_lift": (round(st.mean(swap) - st.mean(swap_base), 4) if swap else None),
            "unparsed": unparsed.get(f"hyp:{h['hyp_id']}", 0),
        })
        pvals.append(p)
    for row, q in zip(rows, benjamini_hochberg(pvals) if pvals else []):
        row["q"] = round(q, 5)
    rows.sort(key=lambda r: (-r["lift"], r["p"]))

    # pooled: every hypothesis forecast against the same obvious forecast
    pooled_real, pooled_base, pooled_swap, pooled_swap_base = [], [], [], []
    clusters = collections.defaultdict(list)
    swap_clusters = collections.defaultdict(list)
    for h in motives:
        ids = sorted(cp["cp_id"] for cp in test
                     if (cp["corpus"], cp["person"]) == (h["corpus"], h["person"]))
        r, b_ = vec(f"hyp:{h['hyp_id']}", ids), vec("obvious", ids)
        pooled_real += r
        pooled_base += b_
        for i, x, y in zip(ids, r, b_):
            clusters[i].append((x, y))
        sk = swap_of.get((h["corpus"], h["person"]))
        if sk:
            sids = sorted(cp["cp_id"] for cp in test
                          if (cp["corpus"], cp["person"]) == sk)
            sr, sb = vec(f"swap:{h['hyp_id']}", sids), vec("obvious", sids)
            pooled_swap += sr
            pooled_swap_base += sb
            for i, x, y in zip(sids, sr, sb):
                swap_clusters[i].append((x, y))

    b, c, p = mcnemar(pooled_real, pooled_base)
    sb, sc, sp = mcnemar(pooled_swap, pooled_swap_base) if pooled_swap else (0, 0, 1.0)
    pooled = {
        "forecasts": len(pooled_real),
        "hypothesis_accuracy": round(st.mean(pooled_real), 4) if pooled_real else None,
        "obvious_accuracy": round(st.mean(pooled_base), 4) if pooled_base else None,
        "lift": round(st.mean(pooled_real) - st.mean(pooled_base), 4) if pooled_real else None,
        "mcnemar": {"better": b, "worse": c, "p": round(p, 6),
                     "caveat": ("ANTI-CONSERVATIVE. One `obvious` forecast is the "
                                "reference for all of a person's hypotheses, so these "
                                "pairs are not independent. Use permutation_p.")},
        "permutation_p": round(permutation_p_clustered(clusters, iters), 5),
        "permutation_unit": "choice point (all hypotheses on one point flip together)",
        "n_clusters": len(clusters),
        "swap_control": {
            "forecasts": len(pooled_swap),
            "lift": round(st.mean(pooled_swap) - st.mean(pooled_swap_base), 4) if pooled_swap else None,
            "mcnemar": {"better": sb, "worse": sc, "p": round(sp, 6)},
            "permutation_p": (round(permutation_p_clustered(swap_clusters, iters), 5)
                              if swap_clusters else None),
        },
    }

    by_method = {}
    for m in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == m]
        by_method[m] = {
            "hypotheses": len(sub),
            "mean_lift": round(st.mean(r["lift"] for r in sub), 4),
            "mean_swap_lift": (round(st.mean(r["swap_lift"] for r in sub
                                             if r["swap_lift"] is not None), 4)
                               if any(r["swap_lift"] is not None for r in sub) else None),
            "best": max(sub, key=lambda r: r["lift"])["hyp_id"],
        }

    surprises = collect_surprises(test, truth, pred, motives, rows, swap_of)

    verdict = read_verdict(summary, pooled, chance)
    return {
        "_README": ("Choice-point forecasting. The scored quantity is lift over the "
                    "`obvious` arm, not accuracy. Read `tripwire` before anything else: "
                    "if the blind arm beats chance the alternative sets leak and every "
                    "number below is void."),
        "at": dt.date.today().isoformat(), "git_commit": git_head(),
        "model": model.model_name, "context_turns": context_turns,
        "cast": sorted(f"{c}:{p}" for c, p in cast),
        "n_test_choices": len(test), "n_dev_choices": len(dev),
        "majority_action": mode,
        "caps": {"max_test_per_person": max_test_per_person, "max_cast": max_cast,
                 "rule": "earliest choices per person; cast by held-out count, both "
                         "fixed before any forecast is made"},
        "chance_rate": round(chance, 4),
        "swap_pairing": {f"{a[0]}:{a[1]}": f"{b[0]}:{b[1]}" for a, b in swap_of.items()},
        "tripwire": verdict["tripwire"],
        "verdict": verdict["verdict"],
        "baselines": summary,
        "pooled": pooled,
        "by_method": by_method,
        "hypotheses": rows,
        "surprises": surprises,
        **model.stats(),
    }


def read_verdict(summary, pooled, chance):
    blind = summary["blind"]["accuracy"]
    majority = summary.get("majority", {}).get("accuracy")
    leak = blind > chance + 0.10
    # POST-HOC, and labelled as such. The pre-registered rule compares blind to
    # CHANCE, which conflates two things: an option set whose phrasing gives the
    # answer away (a real leak), and an option set on which the modal action is
    # usually right (the class prior, which is not a leak and which a blind
    # guesser recovers for free). Comparing blind to the majority constant
    # separates them. This refinement was written after the rule fired and does
    # not change the verdict it produced.
    beats_prior = (majority is not None and blind > majority)
    tripwire = {
        "blind_accuracy": blind, "chance_rate": round(chance, 4),
        "majority_accuracy": majority,
        "leaks": leak,
        "rule": "PRE-REGISTERED: blind > chance + 0.10",
        "reading": ("THE ALTERNATIVE SETS LEAK by the pre-registered rule. The options "
                    "alone predict the answer well above chance."
                    if leak else
                    "clean -- the options alone do not predict the answer"),
        "post_hoc": {
            "blind_beats_majority": beats_prior,
            "reading": (("blind also beats the majority constant, so the option "
                         "phrasing carries information beyond the class prior -- this "
                         "is a real leak")
                        if beats_prior else
                        ("blind does NOT beat the majority constant, so what it "
                         "recovers is the class prior (one action dominates the menu "
                         "and the record), not the phrasing. That is a degenerate "
                         "label distribution, not outcome leakage -- a different "
                         "problem, and not one a rerun of the same extraction fixes.")),
            "status": "written after the rule fired; does not change the verdict",
        },
    }
    if leak:
        return {"tripwire": tripwire,
                "verdict": ("VOID by the pre-registered tripwire -- blind "
                            f"{blind:.3f} > chance {chance:.3f} + 0.10")}
    lift = pooled["lift"]
    swap = (pooled["swap_control"] or {}).get("lift")
    if lift is None:
        return {"tripwire": tripwire, "verdict": "no forecasts"}
    p_head = pooled["permutation_p"]
    if p_head > 0.05:
        v = f"NO EFFECT -- pooled lift {lift:+.3f}, permutation p={p_head}"
    elif swap is not None and swap >= lift:
        v = (f"CONTROL MOVED WITH IT -- real lift {lift:+.3f}, swapped-person lift "
             f"{swap:+.3f}. A motive that forecasts somebody else's choices as well "
             f"as its own is reading the transcript, not the person.")
    elif swap is not None and swap >= 0.5 * lift:
        # Pre-registered middle band. A control at half the effect is not a
        # clean result and must not be reported as one, but it is also not the
        # flat null above -- naming it stops the reader picking whichever of
        # the two neighbouring verdicts they prefer.
        v = (f"WEAK -- pooled lift {lift:+.3f} (permutation p={p_head}) but the "
             f"swapped-person control reached {swap:+.3f}, at least half of it. Most "
             f"of what the motive buys is available without knowing whose motive it is.")
    else:
        v = (f"EFFECT -- pooled lift {lift:+.3f} over the obvious read "
             f"(permutation p={p_head}), swapped-person control {swap:+.3f}")
    return {"tripwire": tripwire, "verdict": v}


def collect_surprises(test, truth, pred, motives, rows, swap_of):
    """Choice points the obvious read missed and some motive caught.

    The honest null is printed beside it. With enough hypotheses somebody gets
    every item right by luck, so the same count is computed for the SWAPPED
    motives -- which cannot apply to the person whose choice they are
    forecasting -- and a surprise list no longer than its own control is not a
    finding. `cleared` marks the subset whose hypothesis also beat the obvious
    read overall at q <= 0.10, which is the list worth reading.
    """
    by_id = {cp["cp_id"]: cp for cp in test}
    credible = {r["hyp_id"] for r in rows if r.get("q", 1.0) <= 0.10 and r["lift"] > 0}
    out, swap_points = [], 0
    for cp_id, actual in truth.items():
        if pred["obvious"].get(cp_id) == actual:
            continue
        hits = [h for h in motives
                if pred.get(f"hyp:{h['hyp_id']}", {}).get(cp_id) == actual]
        # COUNT CHOICE POINTS, NOT HITS. The first version of this counted every
        # (motive, point) hit on the swap side against a count of POINTS on the
        # real side, so the null came out an order of magnitude too large and
        # would have buried a genuine list. Both sides are now "points where at
        # least one motive was right".
        swap_points += any(pred.get(f"swap:{h['hyp_id']}", {}).get(cp_id) == actual
                           for h in motives)
        if not hits:
            continue
        cp = by_id[cp_id]
        out.append({
            "cp_id": cp_id, "corpus": cp["corpus"], "channel": cp["channel"],
            "person": cp["person"], "date": cp["date"], "set_id": cp["set_id"],
            "situation": cp["situation"], "actual": actual,
            "obvious_said": pred["obvious"].get(cp_id),
            "alternatives": cp["alternatives"],
            "caught_by": [{"hyp_id": h["hyp_id"], "method": h["method"],
                           "motive": h["motive"], "because": h.get("because"),
                           "credible": h["hyp_id"] in credible} for h in hits],
            "any_credible": any(h["hyp_id"] in credible for h in hits),
        })
    out.sort(key=lambda s: (not s["any_credible"], -len(s["caught_by"])))
    return {
        "obvious_misses": sum(1 for i, a in truth.items() if pred["obvious"].get(i) != a),
        "caught_by_some_hypothesis": len(out),
        "caught_by_a_credible_hypothesis": sum(1 for s in out if s["any_credible"]),
        "same_misses_caught_by_a_swapped_motive": swap_points,
        "null_note": ("swapped motives cannot apply to the person whose choice they "
                      "forecast, so the points they still 'catch' are the luck rate. "
                      "Both numbers count CHOICE POINTS where at least one motive was "
                      "right. A list no longer than its own null is not a finding."),
        "items": out,
    }


def write_surprises(res, path=SURPRISES):
    s = res["surprises"]
    L = [f"# Things people did not see\n",
         f"{s['caught_by_a_credible_hypothesis']} of {s['obvious_misses']} choice points "
         f"where the obvious read was wrong and a motive that had already proved itself "
         f"was right.\n",
         f"- obvious read wrong on **{s['obvious_misses']}** of {res['n_test_choices']} held-out choices",
         f"- some hypothesis right on **{s['caught_by_some_hypothesis']}** of them",
         f"- a hypothesis that beat the obvious read overall (q <= 0.10) right on "
         f"**{s['caught_by_a_credible_hypothesis']}**",
         f"- the swapped-person control caught **{s['same_misses_caught_by_a_swapped_motive']}** "
         f"of the same points\n",
         f"> {s['null_note']}\n",
         f"**Verdict on the run as a whole:** {res['verdict']}\n", "---\n"]
    for i, item in enumerate(s["items"], 1):
        flag = "" if item["any_credible"] else "  _(no hypothesis here cleared the bar)_"
        L.append(f"## {i}. {item['person']} — {item['date']} — `{item['channel']}`{flag}\n")
        L.append(f"**Facing:** {item['situation']}\n")
        L.append(f"**Obvious read said** `{item['obvious_said']}`. "
                 f"**They actually** `{item['actual']}`.\n")
        L.append("| | option | gives up |")
        L.append("|---|---|---|")
        for a in item["alternatives"]:
            mark = "**taken**" if a["action"] == item["actual"] else ""
            L.append(f"| {mark} | `{a['action']}` {a['gist']} | {a['gives_up']} |")
        L.append("\n**Caught by:**\n")
        for h in item["caught_by"]:
            tick = "✓" if h["credible"] else "·"
            L.append(f"- {tick} `{h['method']}` — {h['motive']}")
        L.append(f"\n`{item['cp_id']}`\n\n---\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


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
    ap.add_argument("--context-turns", type=int, default=40)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--surprises", default=SURPRISES)
    ap.add_argument("--max-test-per-person", type=int, default=10)
    ap.add_argument("--max-cast", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]
    model = FakeModel() if a.fake else CachedModel(a.model)
    res = run(corpora, model, a.context_turns, a.out, a.iters,
              a.max_test_per_person, a.max_cast, a.chunk)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    write_surprises(res, a.surprises)

    print(f"\n{res['n_test_choices']} held-out choices, cast of {len(res['cast'])}, "
          f"chance {res['chance_rate']:.3f}")
    tw = res["tripwire"]
    print(f"\nTRIPWIRE  blind={tw['blind_accuracy']:.3f}  chance={tw['chance_rate']:.3f}  "
          f"majority={tw['majority_accuracy']}")
    print(f"  {tw['reading']}")
    print(f"  POST-HOC: {tw['post_hoc']['reading']}")
    print("\nbaselines:")
    for arm, d in res["baselines"].items():
        print(f"  {arm:<10} acc {d['accuracy']:.3f}  ({d['correct']}/{d['n']}) "
              f"unparsed {d['unparsed']}")
    p = res["pooled"]
    print(f"\npooled  hypothesis {p['hypothesis_accuracy']:.3f} vs obvious "
          f"{p['obvious_accuracy']:.3f}  lift {p['lift']:+.3f}  "
          f"clustered perm p={p['permutation_p']} over {p['n_clusters']} choice points "
          f"(McNemar p={p['mcnemar']['p']}, anti-conservative)")
    sc = p["swap_control"]
    print(f"swap    lift {sc['lift']}  (p={sc['mcnemar']['p']})")
    print("\nby method (mean lift / swapped-control lift):")
    for m, d in sorted(res["by_method"].items(), key=lambda kv: -kv[1]["mean_lift"]):
        print(f"  {m:<14} {d['mean_lift']:+.3f}   {d['mean_swap_lift']}")
    print(f"\nVERDICT: {res['verdict']}")
    print(f"surprises -> {a.surprises}  ({res['surprises']['caught_by_a_credible_hypothesis']} "
          f"credible of {res['surprises']['caught_by_some_hypothesis']} points, "
          f"swap null {res['surprises']['same_misses_caught_by_a_swapped_motive']} points)")
    print(f"\n{model.stats()}")
    if a.fake:
        print("FAKE MODEL -- nothing here is evidence")


if __name__ == "__main__":
    main()
