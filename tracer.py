from copy import deepcopy
import math
import re
import os
import json
import argparse
from typing import List
from abc import ABC, abstractmethod
import numpy as np

from rich import print, box
from rich.panel import Panel
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from agents.load_model import load_model
from utils import (
    load_prompt,
    softmax,
    prompting_for_ordered_list,
    overall_jaccard_similarity,
    jaccard_similarity,
    list_to_unordered_list_string,
    capture_and_parse_ordered_list,
    NpEncoder
)
from hypothesis import compute_ess, extract_question, resample_hypotheses_with_other_info, HypothesesSetV3
import musing_layout
import trace_log
from trace_log import StepRecord, ParticleRecord, RunLogger


_ANSWER_PATTERNS = [
    # boxed first: it is unambiguous, and it also appears *after* the phrase
    # "final answer is", which would otherwise capture the LaTeX wrapper.
    r"\\boxed\{\s*\(?([a-f])\)?\s*\}",
    r"Answer\s*:\s*(.+)",
    r"\*\*\s*Answer\s*\*\*\s*:?\s*(.+)",
    r"final answer is\s*(.+)",
]


def _normalize_answer(span: str) -> str:
    """Strip LaTeX/markdown wrapping so the bucket matcher sees a bare letter."""
    span = span.strip()
    span = re.sub(r"\\boxed\{\s*\(?([a-f])\)?\s*\}", r"(\1)", span, flags=re.I)
    span = span.replace("$", "").replace("\\", "").strip()
    span = span.strip("*").strip()
    if re.fullmatch(r"\{?\(?([a-fA-F])\)?\}?", span):
        return f"({span.strip('{}()')})"
    return span


def extract_answer_span(response: str):
    """Split a likelihood response into (reasoning, answer span).

    The original did response.split("Answer:")[-1], which returns the WHOLE
    response when "Answer:" is absent -- so the loose bucket matcher then
    scanned the full prose and matched whatever letter-shaped substring it hit
    first. Observed in practice: the model answers `The final answer is
    $\boxed{f}$`, which never contains "Answer:", so the verdict was dropped
    and scored as a parse failure.

    That fallback is not merely lossy, it is sign-flipping: a parse failure
    scores 0.001, which is identical to bucket 'f'. An unparsed "Very Likely"
    therefore becomes "Very Unlikely" -- the maximum possible error. Phase 1
    must give parse failure its own value rather than aliasing it onto 'f'.
    """
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, response, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return response[: m.start()].strip(), _normalize_answer(m.group(1))
    return response.strip(), ""


# A COMMITMENT is what the target is trying to achieve, in the target's own
# terms. Split and anchored perturbation both mint commitments from free text,
# and both were unconstrained: once the positional-likelihood bug was fixed and
# split started firing for the first time, 23 of 42 anchors in one run came back
# as requests to another character ("Ask Zuko to leverage any Fire Nation
# intelligence networks ..."). That is the next conversational move, not an aim,
# and --use-anchor then conditions propagation on it.
#
# Rejecting is safe here in a way that guessing is not: a dropped child means
# one fewer split, while a bad anchor founds a root and persists.
COMMITMENT_MAX_WORDS = 8
_BAD_COMMITMENT_VERB = re.compile(
    r"^\s*\**\s*(ask|asks|asking|prompt|prompts|inquire|inquires|question|questions|"
    r"tell|tells|request|requests|demand|demands|urge|urges|get|have|make)\b", re.I)
# "?" is decisive: a commitment is never interrogative.
_INTERROGATIVE = re.compile(r"\?\s*$|^\s*(why|how|what|whether|does|do|is|are|can|should|would)\b", re.I)


_ACK_OPENING = re.compile(
    r"^\s*(hi|hola|hey|hello|ok|okay|thanks|thank you|tha ?k you|thx|merry|happy|great|"
    r"perfect|sure|yes|yep|no problem|got it|noted|will do|understood|good morning)\b", re.I)


def action_content_words(action):
    """Content words in an utterance, ignoring the speaker prefix and mentions."""
    a = re.sub(r"^[^:\n]{0,40}:\s*", "", action or "")
    a = re.sub(r"<@[A-Z0-9]+>", " ", a)
    a = re.sub(r"&[a-z]+;", " ", a)
    return len(re.findall(r"[A-Za-z0-9]+", a))


def is_low_content(action, min_words=7, ack_words=12):
    """Can this utterance reveal a commitment at all?

    A bare "Okay! Thanks!" cannot, whatever a scorer says about it, so this is
    decided deterministically rather than asked. It exists because the routine
    null could not: asked to choose between "no commitment explains this" and
    "this is routine traffic -- scheduling, logistics, tooling", the model
    matched LONG scheduling messages to the routine null and left bare
    acknowledgements to the generic one. Measured on v5_bb_Rodriguez, surprise
    fired on 25 steps of which 16 opened with an acknowledgement and 8 were six
    words or shorter, while off_topic caught the 41-word logistics messages --
    exactly inverted. Length is the signal the wording could not carry.
    """
    n = action_content_words(action)
    if n < min_words:
        return True
    return bool(_ACK_OPENING.match((action or "").split(":", 1)[-1].strip())) and n <= ack_words


def compose_aim_belief(raw, target_agent):
    """Flatten a propagated AIM/BELIEF response into the single belief text.

    Downstream -- the scorer, the summary, every eval -- consumes
    `hypotheses.texts` as one string, so the two fields are composed rather
    than stored apart. Keeping the AIM sentence IN the text is the point: it is
    what makes the belief say why the target is acting, and it makes text and
    anchor comparable, which is what the anchor/text divergence in FINDINGS.md
    §4 lacked.

    Degrades to the raw response if the fields are absent. Propagation runs on
    every particle on every step, so a parse failure must never lose a belief;
    that is strictly worse than an unlabelled one.
    """
    if not raw or not raw.strip():
        return raw or ""
    m_a = re.search(r"AIM\s*:\s*(.+?)(?=\n\s*BELIEF\s*:|$)", raw, re.I | re.S)
    m_b = re.search(r"BELIEF\s*:\s*(.+)", raw, re.I | re.S)
    if not m_a and not m_b:
        return raw.strip()
    aim = " ".join(m_a.group(1).split()).strip().strip("*").strip() if m_a else ""
    belief = " ".join(m_b.group(1).split()).strip().strip("*").strip() if m_b else ""
    if aim and not aim.endswith((".", "!", "?")):
        aim += "."
    return (f"{aim} {belief}".strip() if aim else belief) or raw.strip()


# ---------------------------------------------------------------------------
# STANDARD prompt variants.
#
# The field is tuned against eval_motive_sep.py, which scores whether two
# groups' settlement modes separate THE SAME WAY across independent seeds.
# A variant is kept only if it raises that score, and the variants live here
# rather than inline so a run records which one it used and two runs are
# comparable.
#
#   v1  the first working version: "what would settle it", one worked example
#   v2  names WHO supplies the evidence and WHOSE acceptance ends the question
#
# {t} is the target's name.
# ---------------------------------------------------------------------------
STANDARD_PROMPTS = {
    "v1": (
        "- The STANDARD is what would SETTLE the question for {t}: the evidence they "
        "would accept as showing they were wrong. It is not what they want and not "
        "what they believe -- it is the test they would apply. Two proposals may "
        "share an aim and differ only here.\n"
    ),
    # v1 left WHO free, so the same hypothesis came back as "a recount shows a
    # lower number" on one seed and "the customer confirms the count" on the
    # next -- the axis moved between runs while the aim stayed put. Naming the
    # source and the acceptor makes the two halves of the answer explicit
    # instead of implicit in phrasing.
    "v2": (
        "- The STANDARD is what would SETTLE the question for {t}. Give it in two "
        "parts, in this order and in one sentence:\n"
        "    WHO OR WHAT produces the evidence -- a named person, a customer or "
        "grower, an instrument or measurement, or a written document; and\n"
        "    WHOSE ACCEPTANCE ends the question -- who has to be satisfied before "
        "{t} stops pressing.\n"
        "  Do not hedge between sources. If {t} would be satisfied by a colleague "
        "saying so, say that; if only a measurement would do, say that. The "
        "difference between those two answers is the point.\n"
        "  It is NOT the commitment restated as an outcome: \"Prevent mislabeling\" "
        "-> \"no images are mislabeled\" is WRONG, that is the aim achieved rather "
        "than evidence of it.\n"
    ),
    # v3: the instability is that a standard is invented per COMMITMENT, from
    # plausibility, while the thing we want to measure is a property of the
    # PERSON. Commitments churn every step; the person does not. Tying the
    # answer to what the target has visibly accepted or refused in the record
    # makes it a reading rather than a guess.
    "v3": (
        "- The STANDARD is what would SETTLE the question for {t}, and it must be "
        "grounded in the record you have been shown, not in what is generally "
        "reasonable.\n"
        "  Look at what {t} has actually treated as settling something so far: what "
        "they accepted without argument, what they kept pressing on after an answer, "
        "and what made them stop pressing. Answer in that pattern.\n"
        "  Say WHO OR WHAT supplies the evidence -- a named colleague, a customer or "
        "grower, a measurement or instrument, or a written document.\n"
        "  If the record does not show {t} accepting that kind of evidence, do not "
        "propose it.\n"
        "  It is NOT the commitment restated as an outcome: \"Prevent mislabeling\" "
        "-> \"no images are mislabeled\" is WRONG.\n"
    ),
    # v4: attacks MEASUREMENT noise rather than the model. Free-text standards
    # vary in phrasing, and the scorer then bins them inconsistently -- 39% of
    # Wolf's standards fell in no bin at all, which alone could flip a modal
    # gap between seeds. Forcing a choice from a fixed set removes the binning
    # step from the loop. The set is deliberately about the SOURCE of evidence,
    # which is the distinction the corpus turned out to carry.
    "v4": (
        "- The STANDARD has two parts, separated by a semicolon.\n"
        "  FIRST, exactly one of these words, naming what would settle the question "
        "for {t} -- choose the single best fit, never two:\n"
        "      COLLEAGUE    a named person on the team saying so, or agreeing\n"
        "      CUSTOMER     a customer, grower or other outside party saying so\n"
        "      MEASUREMENT  an instrument, count, statistic or reproducible test\n"
        "      DOCUMENT     a written definition, guideline or spec being in place\n"
        "  SECOND, one short sentence saying concretely what that evidence would be.\n"
        "  Example form: \"MEASUREMENT; two labellers independently assign the same "
        "class to the same 50 images.\"\n"
        "  Example form: \"CUSTOMER; the grower accepts the delivered figures without "
        "querying them.\"\n"
        "  Pick on what {t} has actually been satisfied by in the record, not on what "
        "would be most rigorous. It is NOT the commitment restated as an outcome.\n"
    ),
}


def standard_rule(target, variant=None):
    v = variant or "v1"
    return STANDARD_PROMPTS.get(v, STANDARD_PROMPTS["v1"]).format(t=target)


def valid_commitment(clause, max_words=COMMITMENT_MAX_WORDS, enabled=True):
    """Is this clause an aim of the target's, rather than a move against someone?

    Returns (ok, reason). The caller drops the candidate on a reason -- see the
    note above on why dropping beats repairing.
    """
    if not enabled:
        return True, ""
    if not clause or not clause.strip():
        return False, "empty"
    c = " ".join(clause.strip().split())
    c = c.strip("*").strip()
    n = len(c.split())
    if n > max_words:
        return False, f"{n} words > {max_words}"
    if _BAD_COMMITMENT_VERB.match(c):
        return False, "directs an action at another character"
    if _INTERROGATIVE.search(c):
        return False, "phrased as a question"
    return True, ""


def accumulate_weights(lineage_ids, likelihood, accum, alpha=0.85, beta=1.0, eps_frac=0.12):
    """One step of w_t proportional to w_{t-1}^alpha * L_t^beta, with a floor.

    Pure arithmetic, no particle objects: `Tracer.accumulate` wraps it and
    replay_likelihood.py calls it directly, so an offline sweep cannot drift
    away from what production actually computes.

    `accum` maps lineage_id -> previous normalized weight. A lineage not in it
    starts at 1/n, uninformed. Returns the new weights, the unnormalized log
    accumulators, and the updated map.
    """
    n = len(lineage_ids)
    if n == 0:
        return None
    eps = float(eps_frac) / n

    L = [float(x) for x in likelihood]
    tot = sum(L)
    L = [x / tot for x in L] if tot > 0 else [1.0 / n] * n

    raw, prior = [], []
    for lid, l in zip(lineage_ids, L):
        w_prev = accum.get(lid)
        if w_prev is None:
            w_prev = 1.0 / n          # a new lineage starts uninformed
        prior.append(w_prev)
        # Unnormalized accumulated log-weight: the particle's OWN evidence
        # trajectory. Reversals are counted on this and never on the
        # normalized weights, which are coupled across particles.
        raw.append(alpha * math.log(max(w_prev, 1e-12)) + beta * math.log(max(l, 1e-12)))

    m = max(raw)
    ex = [math.exp(r - m) for r in raw]
    z = sum(ex) or 1.0
    w = [x / z for x in ex]

    floored = [max(x, eps) for x in w]
    z2 = sum(floored) or 1.0
    w_final = [x / z2 for x in floored]
    # Flooring moves mass from high-weight particles to floored ones. This
    # is logged, never asserted on -- only merge and split conserve exactly.
    mass_moved = sum(abs(a - b) for a, b in zip(w, w_final)) / 2.0

    return {'weights': w_final, 'raw_accumulator': raw, 'prior': prior,
            'accum': {lid: x for lid, x in zip(lineage_ids, w_final)},
            'mass_moved_by_floor': mass_moved, 'epsilon': eps,
            'alpha': alpha, 'beta': beta}


def parse_ranking(text: str, n: int):
    """Parse the RANKING block into a 0-based permutation, best first.

    Returns (order, parse_error), where order[r] is the 0-based hypothesis the
    model placed at rank r+1. Tolerates "1st: 3", "1. 3", "**1st**: 3", "1) H3".

    Refuses to guess unless the result is a full permutation. A partial ranking
    is worse than none: it is used below to decide how the ALLOCATION block is
    keyed, and a half-parsed order would silently mis-key every score.

    The ALLOCATION block is cut off first. Its lines are "1: 95"-shaped and
    would otherwise be read as rank-to-hypothesis pairs.
    """
    head = re.split(r"ALLOCATION|REASONING|EXPLANATION", text, flags=re.I)[0]
    pairs = re.findall(
        r"^\s*\**\s*(\d+)\s*(?:st|nd|rd|th)?\s*\**\s*[:.\)\-]\s*\**\s*(?:H|#)?\s*(\d+)",
        head, flags=re.M | re.I)
    ranks = {}
    for pos, hyp in pairs:
        p, h = int(pos), int(hyp)
        if 1 <= p <= n and 1 <= h <= n and p not in ranks:
            ranks[p] = h - 1
    if len(ranks) != n:
        return None, f"expected {n} ranks, parsed {len(ranks)}"
    order = [ranks[p] for p in range(1, n + 1)]
    if len(set(order)) != n:
        return None, "ranking is not a permutation"
    return order, None


def order_agreement(scores, order):
    """Fraction of ranked pairs a score vector agrees with, ties counting half.

    1.0 means the scores reproduce the stated order exactly; 0.5 is chance.
    Used to decide which of two readings of an ALLOCATION block the model meant.
    """
    n = len(order)
    conc, tot = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order[i], order[j]      # a was ranked better than b
            tot += 1
            if scores[a] > scores[b]:
                conc += 1.0
            elif scores[a] == scores[b]:
                conc += 0.5
    return conc / tot if tot else 0.0


def map_allocation_to_hypotheses(alloc, order):
    """Resolve an ALLOCATION block against the model's stated RANKING.

    The rank-mode prompt asks for RANKING ("1st: <hypothesis>") and then
    ALLOCATION ("1: <0-100>"). Following the ordinals, the model usually reads
    the allocation keys as RANK POSITIONS -- so "1: 95" means "the hypothesis I
    ranked first scores 95", not "hypothesis 1 scores 95". Measured over 2068
    scored steps in musing_out, it used that convention on ~89% of them and the
    hypothesis-keyed convention on the rest, with no marker distinguishing them.

    Reading rank-keyed output as hypothesis-keyed hands the top likelihood to
    whatever sits at index 0, every step, regardless of evidence.

    So build both readings and keep whichever reproduces the stated ranking.
    That is self-correcting: a rank-keyed reading of a non-monotone allocation
    contradicts the ranking it came from, and a hypothesis-keyed reading of a
    monotone one usually does too.

    Returns (scores_in_hypothesis_order, keying, agreement).
    """
    n = len(alloc)
    by_rank = [0.0] * n
    for r, h in enumerate(order):
        by_rank[h] = alloc[r]
    by_hyp = list(alloc)

    ag_rank = order_agreement(by_rank, order)
    ag_hyp = order_agreement(by_hyp, order)

    if by_rank == by_hyp:
        return by_rank, 'rank', ag_rank
    if ag_rank > ag_hyp:
        return by_rank, 'rank', ag_rank
    if ag_hyp > ag_rank:
        return by_hyp, 'hypothesis', ag_hyp
    # Equally consistent and genuinely different. Prefer the dominant
    # convention, but say so, because the two disagree about the answer.
    return by_rank, 'ambiguous', ag_rank


def parse_allocation(text: str, n: int):
    """Parse a 100-point allocation block into n floats.

    Returns (values, parse_error). Tolerates "1: 34", "1. 34", "H1 - 34",
    "**1**: 34" and a leading ALLOCATION header. Returns an error string rather
    than guessing when the count is wrong: a short list silently zip-truncated
    the population in the old code path.

    The values come back keyed by whatever number the model wrote. In rank mode
    that key may be a rank position rather than a hypothesis index -- see
    map_allocation_to_hypotheses, which resolves it against the RANKING block.
    """
    head = re.split(r"REASONING|EXPLANATION", text, flags=re.I)[0]
    # Drop the RANKING block when present: its "1st: 3" lines survive the loose
    # fallback pattern below and would be parsed as allocations.
    if re.search(r"ALLOCATION", head, flags=re.I):
        head = re.split(r"ALLOCATION", head, flags=re.I)[-1]
    pairs = re.findall(r"^\s*\**\s*(?:H|#)?\s*(\d+)\s*\**\s*[:.\)\-]\s*\**\s*(-?\d+(?:\.\d+)?)",
                       head, flags=re.M)
    if not pairs:
        pairs = re.findall(r"(?:H|#)?\s*(\d+)\s*[:.\)\-]\s*(-?\d+(?:\.\d+)?)", head)
    vals = {}
    for idx, val in pairs:
        i = int(idx)
        if 1 <= i <= n and i not in vals:
            vals[i] = float(val)
    if len(vals) != n:
        return None, f"expected {n} allocations, parsed {len(vals)}"
    out = [vals[i] for i in range(1, n + 1)]
    if any(v < 0 for v in out):
        return None, "negative allocation"
    # Zero-sum is NOT rejected here. On a free 0-100 scale an all-zero result is
    # a real verdict ("no hypothesis predicts this action"); only the sum-to-100
    # mode, which cannot normalize it, treats it as invalid. The caller decides.
    return out, None


def get_tracer_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-tracing', action='store_true', help='whether to run the model with thought tracing')
    parser.add_argument('--tracing-model', type=str, help='Model to use to answer final question.')
    parser.add_argument('--n-hypotheses', type=int, default=4, help='number of hypotheses to generate for each input', )
    parser.add_argument('--target-perceptions', type=str, default='sight',help='target perceptions to test')  #'sight,hearing,overall', 
    parser.add_argument('--use-helper-llm', action='store_true', help='whether to use user helper llm for identifying target agent and labeling actions',)
    parser.add_argument('--existing-traces', default=None, help='path to existing traces')
    parser.add_argument('--input-is-chat', action='store_true', help="whether the input is a chat or not")
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--likelihood-estimate', default="prompting", type=str, choices=['rollout', 'prompting'], help='likelihood estimation method')
    parser.add_argument('--tracer-type', type=str, help='tracer type')
    return parser

class BaseTracer(ABC):
    def __init__(self, args):
        self.tracer_model = load_model(args.tracing_model, **args.__dict__)
        self.tracer_model.args.model = self.tracer_model.args.tracing_model
        if args.tracing_model == args.model:
            self.base_model = self.tracer_model
        else:
            self.base_model = load_model(args.model, **args.__dict__)
        tracer_name = args.tracing_model.replace("/", "-")
        base_name = args.model.replace("/", "-")
        self.output_file = os.path.join(
            musing_layout.traces_dir(args.output_dir, create=True),
            f"tracer-{tracer_name}_model-{base_name}_runid-{args.run_id}_nhypotheses-{args.n_hypotheses}.jsonl")
        self.trace_header = "Let's trace [target agent]'s thoughts step by step through the context.\n"
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)
        # Phase 0 instrumentation. Off unless a driver attaches a RunLogger, so
        # the upstream eval scripts behave exactly as before.
        self.run_logger = None
        self._accum = {}
        self._low_mass_run = 0
        self._weak_run = {}
        # Retired commitments, kept rather than discarded. Expiry was built to
        # clear dead weight, and on a short span that is all it does. On a long
        # one a topic goes quiet for a hundred turns and comes back, and the
        # commitment that fitted it has been retired in the meantime -- so the
        # filter has to re-invent it, and measurably does not: sz_Product held
        # 'Predict harvest timing for labor projections' at G0 and no later
        # generation ever proposed it again. Reviving is cheaper than minting
        # (no generation call), and it preserves the root, so a commitment that
        # comes back is the SAME hypothesis returning rather than a new one that
        # happens to read alike.
        self._retired = {}

    def accumulate(self, hypotheses, likelihood):
        """w_t proportional to w_{t-1}^alpha * L_t^beta, with a floor.

        Previously `update_weights` overwrote the prior with the fresh
        likelihood every step (tracer.py:746), so strengthening and weakening
        could not exist as trends -- only as single-step blips that the next
        step erased. Posterior ESS was therefore identical to likelihood ESS on
        28/28 steps, and the Panel B gap was purely the parse-failure mask.

        Keyed on lineage_id, not position: propagation mints a new particle_id
        every step, so a positional prior would attribute history to whichever
        particle happened to land at that index.

        The arithmetic lives in the module-level `accumulate_weights` so that
        offline replay (replay_likelihood.py) runs the SAME code as production
        rather than a copy that can drift away from it.
        """
        n = len(hypotheses.hypotheses)
        if n == 0:
            return None
        out = accumulate_weights(
            [h.lineage_id for h in hypotheses.hypotheses], likelihood, self._accum,
            alpha=float(getattr(self.args, 'alpha', 0.85)),
            beta=float(getattr(self.args, 'beta', 1.0)),
            eps_frac=float(getattr(self.args, 'eps_frac', 0.12)))

        hypotheses.update_weights(np.array(out['weights'], dtype=float))
        hypotheses.update_accumulators(out['raw_accumulator'])
        self._accum = dict(out['accum'])
        return {'mass_moved_by_floor': out['mass_moved_by_floor'], 'epsilon': out['epsilon'],
                'alpha': out['alpha'], 'beta': out['beta'],
                'prior': out['prior'], 'raw_accumulator': out['raw_accumulator']}

    def expire_weak(self, hypotheses, just_resampled=False):
        """Retire a commitment that has held negligible mass for k informative steps.

        Resampling decides existence stochastically, and systematic resampling
        guarantees a copy only at >= 1/n, so between resamples a hypothesis can
        sit at the floor indefinitely -- measured at 32 consecutive turns at the
        production divisor, where resampling fires ~0.4 times per run. That is a
        slot spending three LLM calls a turn on a commitment the evidence
        stopped supporting, and a slot a new commitment could hold.

        TWO CORRECTIONS over the first implementation, both found by running it:

        1. KEYED ON root_id, NOT lineage_id. Resample duplicates and perturbed
           particles get fresh lineage ids, so a lineage-keyed counter was
           destroyed by the very operators it needed to survive. root_id is
           inherited through propagation and resampling, so it is the only
           identity that persists. Mass is summed over the root's particles --
           multiplicity is how a resample encodes strength, so a root duplicated
           three times correctly reads as strong rather than as three weaklings.

        2. RESAMPLE STEPS DO NOT COUNT. Resampling resets every weight to
           exactly 1/n, which is above any sub-fair-share threshold, so every
           counter cleared on contact -- measured 0 retirements with the counter
           never exceeding 5 of a required 8, against a median inter-resample
           gap of 7. Uniform weights carry no information about which hypothesis
           is weak, so those steps are FROZEN: neither incremented nor reset.
           Without this the rule cannot fire whenever k exceeds the gap, which
           is the same "can never fire" class as the original split trigger.

        Cost, measured rather than assumed: sub-threshold spells that later
        recover run as long as 33 turns, so no (threshold, k) separates a dead
        hypothesis from a reviving one. Accepted because the replacement MINTS A
        NEW ROOT -- expiry is a source term, not just a remover.
        """
        n = len(hypotheses.hypotheses)
        if n == 0:
            return [], {}
        frac = float(getattr(self.args, 'expiry_weight_frac', 0.5))
        k = int(getattr(self.args, 'expiry_steps', 6))
        thr = frac / n
        mass, members = {}, {}
        for j, h in enumerate(hypotheses.hypotheses):
            mass[h.root_id] = mass.get(h.root_id, 0.0) + float(hypotheses.weights[j])
            members.setdefault(h.root_id, []).append(j)
        n_below = sum(1 for r, m in mass.items() if m < thr)
        if not just_resampled:
            for r, m in mass.items():
                self._weak_run[r] = (self._weak_run.get(r, 0) + 1) if m < thr else 0
            for r in [x for x in self._weak_run if x not in mass]:
                del self._weak_run[r]
        dead_roots = [r for r in mass if self._weak_run.get(r, 0) >= k]
        # retire the particles of expired roots, lightest first; never more than
        # half the population at once -- expiry introduces alternatives, it does
        # not restart the filter.
        weak = [j for r in dead_roots for j in members[r]]
        weak = sorted(weak, key=lambda j: float(hypotheses.weights[j]))[:max(0, n // 2)]
        # Keep what is being retired. Peak weight, not the dying weight: a
        # commitment that once led and then faded is a far better revival
        # candidate than one that never got off the floor.
        for j in weak:
            self.retire(hypotheses.hypotheses[j], hypotheses.weights[j])
        return weak, {'expiry_threshold': thr, 'expiry_steps': k,
                      'expiry_weight_condition': n_below,
                      'expiry_duration_condition': len(weak),
                      'expiry_frozen': bool(just_resampled),
                      'max_weak_run': max(self._weak_run.values()) if self._weak_run else 0}

    def enforce_population_cap(self, hypotheses):
        """Cap at 1.5*N, dropping the lightest particles.

        Load-bearing now rather than theoretical: 3e mints 4-6 roots per repair
        event and split adds 2-3 children per firing, so without a cap the
        population grows with every operator. Each added particle costs a
        propagation call, a coherence check, and a slot in the comparative
        likelihood prompt -- which also gets harder for the evaluator to
        allocate across as it grows, feeding back into the Phase 1 failure mode.
        """
        cap = int(getattr(self.args, 'population_cap', 0) or
                  round(1.5 * float(getattr(self.args, 'n_hypotheses', 8))))
        hyps = hypotheses.hypotheses
        if len(hyps) <= cap:
            return hypotheses, cap, False
        keep = sorted(range(len(hyps)), key=lambda j: -float(hypotheses.weights[j]))[:cap]
        keep.sort()
        import numpy as _np
        texts = [hypotheses.texts[j] for j in keep]
        w = _np.array([float(hypotheses.weights[j]) for j in keep])
        w = w / w.sum() if w.sum() > 0 else _np.ones(len(w)) / len(w)
        out = HypothesesSetV3(
            hypotheses.target_agent, hypotheses.contexts, hypotheses.perceptions,
            texts, w, parent_hypotheses=[hyps[j] for j in keep],
            anchors=[hyps[j].anchor for j in keep],
            accumulators=[hyps[j].raw_accumulator for j in keep],
            lineage_ids=[hyps[j].lineage_id for j in keep])
        for dst, j in zip(out.hypotheses, keep):
            dst.root_id = hyps[j].root_id
        return out, cap, True

    def sync_accumulator(self, hypotheses):
        """Re-key the prior after an operator rebuilds the population.

        Resampling resets weights to 1/N and mints fresh lineage ids for
        duplicates, so the carried prior has to follow the surviving particles
        or the next step would read history off dead lineages.
        """
        self._accum = {h.lineage_id: float(w)
                       for h, w in zip(hypotheses.hypotheses, hypotheses.weights)}

    def attach_logger(self, run_logger):
        self.run_logger = run_logger
        return run_logger

    def _log_step(self, idx, hypotheses, weight_results, operators, likelihood_ess=None, ess_value=None, likelihood_ess_norm=None, accum_info=None, state_action=None, pre_snapshot=None, perturb_info=None, perturb_conditions=None, cap_info=None, split_info=None, expiry_info=None):
        """Emit one StepRecord. Instrumentation only -- never alters the filter."""
        if self.run_logger is None:
            trace_log.RECORDER.drain()
            return None
        weights = [float(w) for w in (hypotheses.weights if hypotheses.weights is not None else [])]
        texts = list(hypotheses.texts)
        likes = None
        if weight_results is not None:
            raw = weight_results.get('weights')
            if raw is not None:
                likes = [float(x) for x in raw]
        ranks = {}
        if likes:
            for rank, i in enumerate(sorted(range(len(likes)), key=lambda j: -likes[j]), start=1):
                ranks[i] = rank
        particles = []
        for i, h in enumerate(hypotheses.hypotheses):
            particles.append(ParticleRecord(
                particle_id=h.particle_id,
                lineage_id=h.lineage_id,
                root_id=h.root_id,
                parent_id=h.parent_id,
                split_child=getattr(h, 'split_child', False),
                resample_duplicate=getattr(h, 'resample_duplicate', False),
                text=texts[i] if i < len(texts) else '',
                anchor=h.anchor,
                standard=getattr(h, 'standard', None),
                weight=weights[i] if i < len(weights) else 0.0,
                raw_accumulator=h.raw_accumulator,
                likelihood=likes[i] if likes and i < len(likes) else None,
                likelihood_rank=ranks.get(i),
                anchor_revisions=getattr(h, 'anchor_revisions', 0),
            ))
        rec = StepRecord(step_idx=idx, particles=particles)
        rec.weights_post = weights
        rec.likelihood_ess = likelihood_ess
        rec.likelihood_ess_norm = likelihood_ess_norm
        if accum_info:
            rec.mass_moved_by_floor = accum_info.get('mass_moved_by_floor')
            rec.epsilon = accum_info.get('epsilon')
        if state_action is not None:
            rec.scored_action = state_action.get('action')
        if weight_results is not None:
            prompts = weight_results.get('prompts') or []
            rec.likelihood_prompt = prompts[0] if prompts else None
            rec.likelihood_system_prompt = weight_results.get('system_prompt')
            rec.scored_texts = list(texts)
            rec.surprise = weight_results.get('surprise')
            rec.off_topic = weight_results.get('off_topic')
            rec.routine_score = weight_results.get('routine_score')
            rec.baseline_score = weight_results.get('baseline_score')
            rec.baseline_rank = weight_results.get('baseline_rank')
            rec.best_margin = weight_results.get('best_margin')
            rec.ranking = weight_results.get('ranking')
            rec.alloc_keying = weight_results.get('alloc_keying')
            rec.rank_alloc_agreement = weight_results.get('rank_alloc_agreement')
            rec.allocation_raw = weight_results.get('allocation_raw')
        # ESS of the particles actually listed below (post-operator), so the
        # record is internally consistent. The pre-operator value that drove the
        # resample decision is kept separately.
        rec.posterior_ess = trace_log.ess(weights)
        rec.posterior_ess_pre = ess_value
        rec.root_mass_ess = trace_log.root_mass_ess_over_n(hypotheses.hypotheses)
        rec.distinct_roots = len({h.root_id for h in hypotheses.hypotheses})
        by_root = {}
        for h in hypotheses.hypotheses:
            by_root.setdefault(h.root_id, []).append(h.text)
        multi = [v for v in by_root.values() if len(v) > 1]
        rec.multi_particle_roots = len(multi)
        rec.within_root_divergence_defined = bool(multi)
        if multi:
            divs = [1 - overall_jaccard_similarity(v) for v in multi]
            rec.within_root_divergence = round(sum(divs) / len(divs), 4)
        else:
            rec.within_root_divergence = None   # undefined, never 0.0
        if expiry_info:
            rec.expiry_threshold = round(float(expiry_info['expiry_threshold']), 5)
            rec.expiry_steps = int(expiry_info['expiry_steps'])
            # logged separately so a zero-expiry run is attributable: weight
            # never low, versus low but never for long enough.
            rec.expiry_weight_condition = int(expiry_info['expiry_weight_condition'])
            rec.expiry_duration_condition = int(expiry_info['expiry_duration_condition'])
            rec.max_weak_run = int(expiry_info['max_weak_run'])
            rec.expiry_frozen = bool(expiry_info['expiry_frozen'])
        if cap_info is not None:
            rec.population_cap, rec.cap_bound = int(cap_info[0]), bool(cap_info[1])
        else:
            rec.population_cap = int(round(1.5 * float(getattr(self.args, 'n_hypotheses', 8))))
        if perturb_conditions is not None:
            rec.perturb_mass_condition = bool(perturb_conditions[0])
            rec.perturb_collapse_condition = bool(perturb_conditions[1])
            rec.perturb_candidates = int(perturb_conditions[2])
            if len(perturb_conditions) > 3:
                rec.perturb_sustained_steps = int(perturb_conditions[3])
                rec.perturb_path = perturb_conditions[4]
        if perturb_info:
            acc_idx = perturb_info.get('accepted') or []
            rec.minted_roots = [hypotheses.hypotheses[j].root_id for j in acc_idx
                                if j < len(hypotheses.hypotheses)]
            if rec.minted_roots:
                rec.minted_root_weight = round(sum(
                    float(hypotheses.weights[j]) for j in acc_idx
                    if j < len(hypotheses.weights)), 4)
            acc = set(perturb_info.get('accepted') or [])
            rej = set(perturb_info.get('rejected') or [])
            unp = set(perturb_info.get('unparsed') or [])
            for j, prec in enumerate(rec.particles):
                if j in acc or j in rej or j in unp:
                    prec.perturbed = True
                    # None => attempted but unparsed; not a coherence verdict
                    prec.perturb_accepted = True if j in acc else (False if j in rej else None)
        if pre_snapshot:
            rec.pre_operator_particles = list(pre_snapshot)
            rec.weights_pre = [p['weight'] for p in pre_snapshot]
        rec.top_to_median_ratio = trace_log.top_to_median_ratio(weights)
        rec.population_pre = rec.population_post = len(particles)
        rec.operators_fired = list(operators)
        _div = float(getattr(self.args, 'ess_divisor', 3.0))
        rec.ess_threshold = len(hypotheses.hypotheses) / _div
        rec.ess_threshold_name = f'N/{_div:g}'
        if weight_results is not None:
            raw_scores = weight_results.get('raw_scores')
            letters = weight_results.get('letters')
            rec.raw_verdicts = (list(letters) if letters is not None
                                else ([] if raw_scores is None else [float(x) for x in raw_scores]))
            rec.parsed_scores = likes if likes else []
            rec.parse_failures = int(weight_results.get('parse_failures') or 0)
            pm = weight_results.get('parse_mask')
            if pm:
                for prec, failed in zip(rec.particles, pm):
                    prec.parse_failed = bool(failed)
        # split trigger: computed on the ParticleRecords, which carry
        # likelihood_rank. HypothesisV3 does not -- rank is derived here from
        # weight_results, so this must come AFTER rec.particles is built.
        if split_info:
            rec.split_weight_condition = split_info.get('weight_condition')
            rec.split_disagree_condition = split_info.get('rank_condition')
            # these existed in `info` but were never persisted, so every split
            # step read split_children=None / merged=None on disk
            rec.split_children = split_info.get('children')
            rec.split_net = split_info.get('split_net')
            rec.expanded = split_info.get('expanded')
            rec.merged_count = split_info.get('merged')
            rec.net_new_roots = split_info.get('net_new_roots')
            for col in (split_info.get('anchor_collapse') or []):
                rec.anchor_collapses.append(col)
        sc, wc, dc = trace_log.split_candidates(rec.particles)
        rec.split_candidates = len(sc)
        rec.split_weight_condition = wc
        rec.split_disagree_condition = dc
        if len(texts) > 1:
            n = len(texts)
            pairs = [jaccard_similarity(texts[i], texts[j])
                     for i in range(len(texts)) for j in range(i + 1, len(texts))]
            rec.merge_metric = 'jaccard'
            rec.merge_percentile = float(getattr(self.args, 'merge_percentile', 95.0))
            rec.merge_threshold_resolved = trace_log.resolve_merge_threshold(
                pairs, rec.merge_percentile)
            rec.jaccard_matrix = [[round(jaccard_similarity(texts[i], texts[j]), 4) for j in range(n)] for i in range(n)]
            rec.mean_pairwise_jaccard = round(overall_jaccard_similarity(texts), 4)
        return self.run_logger.step(rec)

    def identify_target(self, input_text: str) -> str:
        """
        Identify the target agent that we have to trace by looking at the question.

        Args:
            input_text (str): The context text.

        Returns:
            str: The target agent.
        """
        question = extract_question(input_text)
        target_identification_prompt = f"'{question}'\n\nMain question: Who is the subject of the above question? Whose perspective is this question primarily about? Provide the name of the individual, their title, or the group. If the subject of the question is not related to a person or a group, state 'none'.\nThe concise answer to the main question is (e.g, name):"

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o', run_id=self.args.run_id)
            output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
        else:
            output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
        target_agent = output.split("\n")[0].split(":")[-1].strip().strip(".").replace("*", "")

        return target_agent

    def label_action(self, agent: str, input_text: str) -> List[dict]:
        """
        Segment the text into interleaved chunks regarding the target agent's actions.

        Args:
            agent (str): The target character.
            input_text (str): The context text.

        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        action_labeling_prompt = load_prompt(f'label_actions_{self.args.dataset}.txt')
        text = input_text.strip().removesuffix("Answer:").strip()
        match = re.search(r'(.+?)(Output:|Choose one of the following:)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        action_labeling_prompt = action_labeling_prompt.replace('<<context>>', text)
        action_labeling_prompt = action_labeling_prompt.replace('<<target_character>>', agent)

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o', run_id=self.args.run_id)
            labeled_text = llm.interact(action_labeling_prompt, temperature=0)
        else:
            labeled_text = self.tracer_model.interact(action_labeling_prompt, temperature=0)

        return labeled_text

    def label_action_for_chat(self, target_agent: str, text: str) -> List[dict]:
        if "\nInformation: " in text:
            separator = "\nInformation: "
        elif "\nTarget: " in text:
            separator = "\nTarget: "
        else:
            separator = "\nQuestion: "
        convo = text.split(separator)[0].split("Meeting:")[-1].strip().split("\n")
        labeled_convo = []
        for line in convo:
            line = line.strip()
            if line != "":
                if line.startswith(target_agent + ":"):
                    line = line.strip() + "<action>"
                else:
                    # capture only the text with alphabets in the line and check whether that text starts with the character's name
                    patterns = re.findall(r'[a-zA-Z]+', line)
                    if patterns[0].startswith(target_agent):
                        line = line.strip() + "<action>"
                    else:
                        line = line.strip() + "<no action>"
                labeled_convo.append(line)

        # merge actions that are split into multiple lines in conversations. This is needed because we don't get perception for utterances from the target speaker.
        for idx, line in enumerate(labeled_convo):
            if line.endswith("<action>"):
                if idx + 1 < len(labeled_convo) and labeled_convo[idx + 1].endswith("<action>"):
                    labeled_convo[idx] = labeled_convo[idx].removesuffix("<action>") + "\n" + labeled_convo[idx + 1].removesuffix("<action>") + "<action>"
                    labeled_convo.pop(idx + 1)

        return "\n".join(labeled_convo)

    def label_action_for_mmtom(self, target_agent: str, text: str) -> List[dict]:
        context, action_text = text.split("\nActions")
        context = context.strip() + "<no action>"
        action_text = "\nActions" + action_text
        _actions = action_text.split(". ")
        actions = []
        for idx, a in enumerate(_actions):
            if idx == 0:
                a = a.strip(".") + ".<no action>" # the first action is not an action -- e.g., David is situated in the kitchen.
                actions.append(a)
            else:
                if a != "":
                    a = a.strip(".") + ".<action>"
                    actions.append(a)

        labeled_text = context + "".join(actions)
        return labeled_text

    def interleave_states_and_actions(self, labeled_text: str, agent: str) -> List[dict]:
        """
        Segment the text into interleaved chunks regarding the target agent's actions and states
        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """

        sentences = labeled_text.split(">")
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]

        # Group parts with no actions, so that the text is segmented into interleaved chunks of actions and states
        segmented_text = []
        if self.args.input_is_chat:
            separator = "\n"
        else:
            separator = " "
        state = ""
        actions = ""
        for sentence in sentences:
            if sentence.endswith("<no action"):
                no_action_sentence = sentence.removesuffix("<no action").strip()
                if actions != "":
                    segmented_text.append({'action': True, 'text': actions})
                    actions = ""

                if state == "":
                    state = no_action_sentence
                else:
                    state = state + separator + no_action_sentence

            elif sentence.endswith("<action"):
                action_sentence = sentence.removesuffix("<action").strip()
                if state != "":
                    segmented_text.append({'action': False, 'text': state})
                    state = ""

                if actions == "":
                    actions = action_sentence
                else:
                    segmented_text.append({'action': True, 'text': actions})
                    # actions = actions + separator + action_sentence
                    actions = action_sentence
            else:
                clean_sentence = sentence.strip()
                if actions != "":
                    segmented_text.append({'action': True, 'text': actions})
                    actions = ""

                if state == "":
                    state = clean_sentence
                else:
                    state = state + separator + clean_sentence

        if state != "":
            segmented_text.append({'action': False, 'text': state})
        if actions != "":
            segmented_text.append({'action': True, 'text': actions})

        return segmented_text

    def set_trajectory(self, state_action_segments: List[dict]) -> List[dict]:
        """
        Set the trajectory of the target agent.

        Args:
            state_action_segments (List[dict]): A list of dictionaries containing the action label and the text.

        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        trajectory = []
        for idx, segment in enumerate(state_action_segments):
            if segment['action']:
                if idx - 1 >= 0:
                    previous = state_action_segments[idx - 1]
                    if previous['action']:
                        trajectory.append({'state': None, 'action': segment['text']})
                    else:
                        trajectory.append({'state': previous['text'], 'action': segment['text']})
                else:
                    trajectory.append({'state': None, 'action': segment['text']})
        if segment['action'] is False:
            trajectory.append({'state': segment['text'], 'action': None})

        if len(trajectory) == 0:
            for idx, segment in enumerate(state_action_segments):
                trajectory.append({'state': segment['text'], 'action': None})

        return trajectory

    def preprocess_input(self, text: str, target_agent=None) -> dict:
        """
        Preprocess the input_text before passing it to the tracer model. Identify the target agent (i.e., character) and label the actions.

        Args:
            text (str): The context text.

        Returns:
            str: The target agent.
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        if target_agent is None:
            target_agent = self.identify_target(text)

        if target_agent.lower() == "none":
            print(Panel(text, title="Input Text: with target character 'none'?", style="red", expand=False, box=box.SIMPLE_HEAD))
            return None

        if self.args.print:
            print(Panel(text, title="Input Text", style="blue", expand=False, box=box.SIMPLE_HEAD))
        context = text.split("\nQuestion:")[0]

        if self.args.input_is_chat:
            action_labeled_text = self.label_action_for_chat(target_agent, context)
        elif self.args.dataset == "mmtom":
            action_labeled_text = self.label_action_for_mmtom(target_agent, context)
        else:
            action_labeled_text = self.label_action(target_agent, context)

        state_action_segments = self.interleave_states_and_actions(action_labeled_text, target_agent)
        trajectory = self.set_trajectory(state_action_segments)
        if len(trajectory) == 0:
            print(Panel(text, title="Input Text: No actions found", style="red", expand=False, box=box.SIMPLE_HEAD))

        if self.args.print:
            print(cf.bold | cf.green("Target character: " + target_agent))
            print()

        question = extract_question(text)

        return {'question': question, 'action_labeled_text': state_action_segments, 'trajectory': trajectory, 'context': context, 'target_agent': target_agent}

    @abstractmethod
    def track_perception(self, target_agent: str, context: str) -> dict:
        """
        Track the perceptions of the target character or agent.

        Args:
            target_agent (str): The target character.
            context (str): The context text.

        Returns:
            dict: A dictionary containing the perception of the target agent and the prompt used for inference.
            {'text': perception_inference, 'prompt': perception_query}
        """
        pass

    @abstractmethod
    def initialize(self, target_agent, context):
        pass

    @abstractmethod
    def trace(self, text):
        pass

    def batch_trace(self, texts, use_tracings=None):
        if use_tracings is None:
            responses = [self.trace(text) for text in texts]
        else:
            responses = []
            for use_tracing, text in zip(use_tracings, texts):
                if use_tracing:
                    responses.append(self.trace(text))
                else:
                    responses.append("")
        return responses

    def dump(self, traced_thought: dict, hypotheses_list: List[HypothesesSetV3]):
        """
        Dump to jsonl

        Args:
            traced_thought (dict): _description_
            hypotheses_list (List[HypothesesSet]): _description_
        """
        dumped_hypotheses_list = [h.dump() for h in hypotheses_list]
        traced_thought['hypotheses'] = dumped_hypotheses_list
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(traced_thought, cls=NpEncoder) + '\n')

    def interact(self, text: str, temperature=0, max_tokens: int=256):
        return self.base_model.interact(text, temperature=temperature)

    def batch_interact(self, texts: list, temperature: float=0, max_tokens: int=256):
        return self.base_model.batch_interact(texts, temperature=temperature, max_tokens=max_tokens)

    def batch_cot(self, texts: list, temperature: float=0, max_tokens: int=256):
        return self.base_model.batch_cot(texts, temperature=temperature, max_tokens=max_tokens)

class Tracer(BaseTracer):
    # Tracer's weighted average is a model-written paragraph; TracerLight's is a
    # concatenated "**Prediction n**" block that reads better opened on its own
    # line. Held as an attribute so both share one chain implementation without
    # either one's emitted trace text changing.
    _thoughts_block_prefix = ""

    def __init__(self, args):
        super().__init__(args)
        self.trace_base_header = "To answer this question, let's analyze the context step by step regarding [target agent]'s perceptions and thoughts."
        self.cache_db = {}

    def set_tracer_variables(self, preprocessed_text):
        self.question = preprocessed_text['question']
        self.input_context = preprocessed_text['context']
        self.target_agent = preprocessed_text['target_agent']
        self.trace_header = self.trace_base_header.replace("[target agent]", self.target_agent)
        self._role_prior = None

    def infer_role_prior(self):
        """One call: what seat is this person sitting in, and what does that seat want?

        Every hypothesis source in this filter reads the transcript and reports what it
        says. That works when the motive is spoken -- ATLA characters announce theirs --
        and fails when it is not: on a Slack thread the honest summary of what someone's
        messages DO is "define numerical thresholds", which is a description of the
        message, not a reason for sending it.

        A prior is the missing half. It is derived ONCE from the whole transcript rather
        than per step, and it names the person's stake, not their utterances -- so the
        proposer has somewhere to generate FROM other than the words in front of it.
        Whole-transcript on purpose: goal-seeding sees only the opening scene, which is
        why nothing ever proposed forgiveness on the ATLA episode.
        """
        if self._role_prior is not None:
            return self._role_prior
        ctx = (self.input_context or "")[:24000]
        sys_p = (
            f"You identify what position someone occupies and what that position gives them "
            f"a stake in.\n\n"
            f"Answer in exactly this form:\n"
            f"SEAT: <who {self.target_agent} is here and who they answer to, one clause>\n"
            f"STAKE: <what someone in that seat is trying to bring about, one clause>\n"
            f"PRESSURE: <what would count as a bad outcome for them, one clause>\n\n"
            f"Rules:\n"
            f"- Describe the POSITION, not the messages. Do not summarise what they said.\n"
            f"- Do not name the specific document, feature, number or object under discussion.\n"
            f"- If their role is not stated anywhere, infer it from who defers to whom, who "
            f"is relaying someone else's requirements, and who is defending a decision.")
        try:
            raw = self.tracer_model.interact(
                f"<transcript>\n{ctx}\n</transcript>\n\nWho is {self.target_agent} here?",
                system_prompt=sys_p, temperature=0, max_tokens=512, stage='role_prior')
        except Exception:
            raw = ""
        self._role_prior = " ".join((raw or "").split()).strip()
        if self._role_prior:
            print(Panel(self._role_prior[:400], title=f"Role prior — {self.target_agent}",
                        style="cyan", box=box.SIMPLE_HEAD))
        return self._role_prior

    def get_perception_tracking_prompts(self, state_action: dict, context_history: List[dict] = None, target_agent: str = None) -> List[str]:
        target_agent = self.target_agent if target_agent is None else target_agent
        state = state_action['state']
        action = state_action['action']
        context_history_list = [c['text'] for c in context_history] if context_history is not None else []
        prompts = []
        sysprompts = []

        if state:
            sysprompt_for_state = f"You are an expert perception tracker tasked with determining whether {target_agent} perceived the target context. Briefly describe what {target_agent} saw or why {target_agent} could not see the target context. Make your response concise."
            if len(context_history) > 0:
                context_input_for_state = ""
                context_input_for_state += list_to_unordered_list_string(context_history_list, list_bullet="")
                context_input_for_state += f"\n<target context>\n{state}\n</target context>"
            else:
                context_input_for_state = f"<context>\n{state}\n</context>"
            query_for_state = f"{context_input_for_state}" 
            prompts.append(query_for_state)
            sysprompts.append(sysprompt_for_state)

        if action:
            sysprompt_for_action = f"You are an expert perception tracker tasked with determining what {target_agent} have perceived during {target_agent}'s action/utterance. Briefly describe what {target_agent} saw during the new actions/utterance. Make your response concise."
            context_input_for_action = "<context>\n"
            if state:
                context_history_list.append(state)
            context_input_for_action += list_to_unordered_list_string(context_history_list, list_bullet="")
            context_input_for_action += f"\n</context>"
            context_input_for_action += f"\n\n<action>{action}</action>"
            query_for_action = f"{context_input_for_action}"
            prompts.append(query_for_action)
            sysprompts.append(sysprompt_for_action)
        return {'prompts': prompts, 'sysprompts': sysprompts}

    def get_perception_tracking_prompts_for_chat(self, state_action: dict, context_history: List[dict] = None, target_agent: str = None) -> List[str]:
        target_agent = self.target_agent if target_agent is None else target_agent
        state = state_action['state']
        action = state_action['action']
        context_history_list = [c['text'] for c in context_history] if context_history is not None else []
        prompts = []
        sysprompts = []

        if state:
            sysprompt_for_state = f"You are an expert perception tracker tasked with determining whether {target_agent} was involved in the conversation or was not. If the {target_agent} was in the scene, they must have perceived the context. If they were away, they did not perceive the context."
            if len(context_history) > 0:
                context_input_for_state = ""
                context_input_for_state += list_to_unordered_list_string(context_history_list, list_bullet="")
                context_input_for_state += f"\n<target context>\n{state}\n</target context>"
            else:
                context_input_for_state = f"<context>\n{state}\n</context>"
            if action:
                context_input_for_state += f"\n<response>\n{action}\n</response>"
            query_for_state = f"{context_input_for_state}" 
            prompts.append(query_for_state)
            sysprompts.append(sysprompt_for_state)

        return {'prompts': prompts, 'sysprompts': sysprompts}

    def track_perception(self, trajectory, target_agent=None):
        trace_log.set_stage('perception')
        prompts = []
        sysprompts = []
        context_history = []
        if target_agent is None:
            target_agent = self.target_agent

        for idx, t in enumerate(trajectory):
            if self.args.input_is_chat:
                perception_prompts = self.get_perception_tracking_prompts_for_chat(t, context_history, target_agent)
            else:
                perception_prompts = self.get_perception_tracking_prompts(t, context_history, target_agent)
            prompts.extend(perception_prompts['prompts'])
            sysprompts.extend(perception_prompts['sysprompts'])
            if t['state']:
                context_history.append({'text': t['state'], 'action': False})
            if t['action']:
                context_history.append({'text': t['action'], 'action': True})

        perception_inferences = self.tracer_model.batch_interact(prompts, temperature=0, system_prompts=sysprompts)
        perception_trajectory = [{'state': None, 'action': None} for _ in range(len(trajectory))]
        for idx, c in enumerate(trajectory):
            if c['state']:
                perception_trajectory[idx]['state'] = perception_inferences.pop(0)
            if c['action'] and not self.args.input_is_chat:
                perception_trajectory[idx]['action'] = perception_inferences.pop(0)

        return perception_trajectory

    def get_agent_state(self, target_agent, context):
        prompt = f"{context}\n\nQuestion: What are the facts regarding {target_agent} in the above context? Please only output the facts directly related to {target_agent} without any additional comments."
        state = self.tracer_model.interact(prompt, temperature=0)
        return state

    def preprocess_input(self, text: str, target_agent=None) -> dict:
        """
        Preprocess the input_text before passing it to the tracer model. Identify the target agent (i.e., character) and label the actions.
        """
        preprocessed_text = BaseTracer.preprocess_input(self, text, target_agent)
        if preprocessed_text is None:
            return None
        preprocessed_text['perceptions'] = self.track_perception(preprocessed_text['trajectory'], preprocessed_text['target_agent'])
        preprocessed_text['assumption'] = self.get_assumption(preprocessed_text['question'])
        self.assumption = f"\n{preprocessed_text['assumption']}"

        return preprocessed_text

    def initialize(self, state_action, perceptions):
        trace_log.set_stage('initialize')
        context_input = ""
        state, action = state_action['state'], state_action['action']
        if state:
            if self.args.input_is_chat:
                context_input += f"{state}\n"
            else:
                agent_state = self.get_agent_state(self.target_agent, state)
                context_input += f"<state>\n{agent_state.strip()}\n</state>\n"
            context_input += f"<note>{perceptions['state']}</note>\n\n"
        
        if action:
            if self.args.input_is_chat:
                context_input += f"<response>\n{action}\n</response>\n"
            else:
                context_input += f"<action>\n{action}\n</action>\n"

            if perceptions['action']:
                context_input += f"<note>{perceptions['action']}</note>\n"

        n_hypotheses_str = str(self.args.n_hypotheses)

        if self.args.n_hypotheses > 1:
            # PHASE 3a: seed at the GOAL level, not the belief level.
            #
            # Two particles can differ in believed facts and still predict the
            # same next utterance, which is a likelihood the evaluator cannot
            # separate -- measured at 88% bucket 'a' with diversity 0.743, i.e.
            # textually various, predictively identical. Goal-level differences
            # are what produce divergent action predictions downstream.
            #
            # PREDICTED NULL: this raises the INTERCEPT (more distinct starting
            # roots) and cannot touch the SLOPE (~0.375-0.50 decay per resample).
            # A higher ending root count is the trivial consequence of a higher
            # start and must not be read as success.
            goal_seeded = getattr(self.args, 'goal_seeding', False)
            if goal_seeded:
                axis = (f"Each hypothesis must propose a DIFFERENT thing {self.target_agent} "
                        f"wants or is trying to achieve -- not merely a different belief about "
                        f"the facts. Two hypotheses that share a goal but differ in wording are "
                        f"the same hypothesis. Make the goals mutually exclusive where possible.")
            else:
                axis = ""
            # Seeding sees only the opening scene, so on a long trajectory the whole
            # commitment vocabulary is fixed by scene 0 -- which is why nothing ever
            # proposed forgiveness on the ATLA episode. The role prior is derived from
            # the WHOLE record, so it widens the seed without showing step 0 the future.
            if getattr(self.args, 'infer_motive', False):
                rp = self.infer_role_prior()
                if rp:
                    axis = (f"{axis} What is known about {self.target_agent}'s position from the "
                            f"whole record: {rp} Let the hypotheses follow from that SEAT and "
                            f"STAKE. State the reason the action is worth taking, never a "
                            f"description of what the action accomplishes. ")
            if action:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what were {self.target_agent}'s thoughts (e.g., beliefs, intent) that led to the action above. {axis}Do not add any additional comments."
            else:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what {self.target_agent} will be thinking (e.g., beliefs). {axis}Do not add any additional comments."
            _hypotheses_list = prompting_for_ordered_list(self.tracer_model, prompt=belief_query, n=self.args.n_hypotheses)
            hypotheses_list = [hypothesis.strip() for hypothesis in _hypotheses_list]
        else:
            belief_query = f"{context_input}\n\nQuestion: What will {self.target_agent} be thinking now?"
            hypothesis = self.tracer_model.interact(belief_query, temperature=0, max_tokens=1024)
            hypotheses_list = [hypothesis]

        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)
        # Anchor EXTRACTION is instrumentation: it is what canonicalises roots,
        # so without it every founding particle gets its own root and step-0
        # root count is N by construction -- making any intercept comparison
        # vacuous. Anchor-IN-PROPAGATION is 3b's actual intervention. These were
        # one flag; they are now separate, so a baseline can be measured with the
        # instrument running and the intervention off.
        anchors = None
        want_extract = (getattr(self.args, 'extract_anchors_flag', False)
                        or getattr(self.args, 'use_anchor', False))
        if want_extract and len(hypotheses_list) > 1:
            anchors = self.extract_anchors(hypotheses_list, self.target_agent)
        initial_hypotheses = HypothesesSetV3(target_agent=self.target_agent, contexts=[state_action], perceptions=[perceptions], texts=hypotheses_list, weights=weights, anchors=anchors)
        # extract_anchors parks the per-anchor standards on the tracer because
        # it returns a bare list; attach them to the founding particles here.
        for h, sd in zip(initial_hypotheses.hypotheses, getattr(self, '_last_standards', []) or []):
            h.update_standard(sd)

        return initial_hypotheses

    def get_assumption(self, question: str):
        if self.args.dataset == 'mmtom':
            if ", " in question:
                assumption_substring = question.split(", ")[0]
            else:
                return None
        else:
            return ""
        prompt = f"{assumption_substring}\n\nTask: Convert the above into a sentence in present tense. Do not add any additional comments."
        model = load_model("gpt-4o-2024-08-06", run_id=self.args.run_id) if self.args.use_helper_llm else self.tracer_model
        assumption = model.interact(prompt, temperature=0, max_tokens=128)
        if self.args.print:
            print(Panel(assumption, title="Assumption", style="green"))

        return assumption 

    def interleave_context_and_perception(self, context_history: List[dict], perception_history: List[dict], target_agent: str = None, chat: bool = False) -> str:
        if target_agent is None:
            target_agent = self.target_agent
        context_and_perception = ""
        for idx, (c, p) in enumerate(zip(context_history, perception_history)):
            if c['state'] or c['action']:
                context_and_perception += f"<context {idx + 1}>\n"
                if c['state']:
                    # Content AND the perception note, matching setup_propagation
                    # (which has always done it this way for the CURRENT step).
                    #
                    # This used to write p['state'] -- the perception SUMMARY --
                    # in place of c['state'], the thing that was actually said.
                    # A perception summary reads "Product perceived the context"
                    # and carries no content, so a target's history held its own
                    # utterances in full and nothing but stubs for everyone
                    # else's. Every belief about another party therefore had to
                    # come from the single current state, which is why beliefs
                    # read as commentary on the latest message.
                    #
                    # Measured consequence: at sz_Product step 2, where Burdett
                    # denies the customer's spec, the whole November negotiation
                    # was absent and the only visible other-party content was
                    # Rodriguez's question -- so 0.823 of Product's mass asserted
                    # the CUSTOMER's spec, and the pair of traces testified that
                    # the two sides agreed at the exact moment they discovered
                    # they did not.
                    #
                    # Perception still bounds what the target can know: it rides
                    # as a <note>, the same mechanism that carries the false-belief
                    # result (FINDINGS.md: perception-tracking bounds what a
                    # target can know), rather than by deleting the record.
                    context_and_perception += f"<state>{c['state']}</state>\n"
                    if p.get('state'):
                        context_and_perception += f"<note>{p['state']}</note>\n\n"
                    else:
                        context_and_perception += "\n"
                if c['action']:
                    if self.args.input_is_chat:
                        context_and_perception += f"<response>\n{c['action']}\n</response>\n"
                    else:
                        context_and_perception += f"<action>{c['action']}</action>\n"

                    if p['action']:
                        context_and_perception += f"<note>{p['action']}</note>\n"
            context_and_perception = context_and_perception.strip() + f"\n</context {idx + 1}>\n\n"
        return context_and_perception.strip()

    def setup_propagation(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions:dict) -> HypothesesSetV3:
        target_agent = existing_hypotheses.target_agent
        context_history = deepcopy(existing_hypotheses.contexts)
        perception_history = deepcopy(existing_hypotheses.perceptions)
        context_and_perception_str = self.interleave_context_and_perception(context_history, perception_history, target_agent)
        context_history.append(state_action)
        perception_history.append(perceptions)

        # PHASE 1 / de-circularisation.
        #
        # The action at step k was previously included in new_context, so every
        # particle was rewritten with sight of the observation it was about to
        # be scored against. weigh() then prunes that action from the LIKELIHOOD
        # context (prompt_likelihood), so the evaluator cannot tell the
        # hypotheses already encode it, and unanimously answers "Very Likely".
        # Measured: action visible -> letters a,a,a,a, likelihood ESS 4.000
        # (exactly uniform); action held out -> a,a,b,b, ESS 3.774.
        #
        # Correct particle-filter semantics: propagate on the prior and the
        # state, THEN weight by the likelihood of the new observation. The
        # action must be a genuinely held-out observation at propagation time.
        # It still enters context_history above, so later steps see it.
        hide_action = getattr(self.args, 'hide_action_from_propagation', True)
        new_context = ""
        if state_action['state']:
            if self.args.input_is_chat:
                new_context += f"<state>\n{state_action['state']}\n</state>\n"
            else:
                new_context += f"<state>{state_action['state']}</state>\n"
            if perceptions['state']:
                new_context += f"<note>{perceptions['state']}</note>\n\n"
        if state_action['action'] and not hide_action:
            if self.args.input_is_chat:
                new_context += f"<response>\n{state_action['action']}\n</response>\n"
            else:
                new_context += f"<action>{state_action['action']}</action>\n"
            if perceptions['action']:
                new_context += f"<note>{perceptions['action']}</note>"

        return {"target_agent": target_agent, "hide_action": hide_action, "context_and_perception_str": context_and_perception_str, "new_context": new_context.strip(), "context_history": context_history, "perception_history": perception_history}

    def weigh(self, hypotheses: HypothesesSetV3, action: str, mode: str) -> dict:
        # Weight the hypotheses based on the next action
        hypotheses_texts = hypotheses.texts
        target_agent = hypotheses.target_agent
        context_history = hypotheses.contexts
        perception_history = hypotheses.perceptions

        if mode == "prompting":
            if getattr(self.args, 'scorer', 'comparative') == 'comparative':
                results = self.prompt_likelihood_comparative(hypotheses_texts, context_history, perception_history, action)
            else:
                results = self.prompt_likelihood(hypotheses_texts, context_history, perception_history, action)
        else:
            raise NotImplementedError

        return results

    def prompt_likelihood(self, existing_hypotheses: list, context_history: list, perception_history: list, action: str, target_agent: str = None):
        def map_response_to_letter(response: str, mapping: dict):
            for option in mapping.keys():
                if response.startswith(f"({option})") or response.startswith(f"{option})") or response.startswith(f"{option}.") or response.startswith(f"{option} ") or f"({option})" in response or f" {option})" in response or f" {option}." in response or option == response:
                    return option
            return None

        def map_response_to_score(response: str, mapping: dict):
            """Score, or None when nothing parsed.

            Returning 0.001 on failure (as this did) is identical to bucket 'f',
            so an unparsed 'Very Likely' silently became 'Very Unlikely' -- the
            maximum possible error, and sign-flipped rather than merely lossy.
            A parse failure carries NO information and must not be scored as a
            confident low verdict.
            """
            for option in mapping.keys():
                if response.startswith(f"({option})") or response.startswith(f"{option})") or response.startswith(f"{option}.") or response.startswith(f"{option} ") or f"({option})" in response or f" {option})" in response or f" {option}." in response or option == response:
                    return mapping[option]
            return None

        trace_log.set_stage('likelihood')
        target_agent = self.target_agent if target_agent is None else target_agent
        # TODO: maybe trim the last action and its perception because the thought actually comes before the action.
        pruned_context_history = deepcopy(context_history)
        pruned_perception_history = deepcopy(perception_history)
        # to remove the last action and its perception, because we will be evaluating the thought before the action
        pruned_context_history[-1]['action'] = None
        pruned_perception_history[-1]['action'] = None 
        if len(pruned_context_history) > 1:
            context_and_perception_str = self.interleave_context_and_perception(pruned_context_history, pruned_perception_history)
        else:
            if pruned_context_history[-1]['state']:
                context_and_perception_str = f"{pruned_context_history[-1]['state']}"
            else:
                context_and_perception_str = ""

        system_prompt = f"Your job is to evaluate the probability of actions/utterance under a given fact. Use common sense: for instance, if someone is searching for an item, they are likely to take it once they find it rather than merely observing it. If they don't take it and just sees it, it indicates a lack of interest and that was not what they were looking for. Briefly explain the probability of the action/utterance under the given fact first and then give the answer option with prefix 'Answer:'"
        question = f"Question: Based on the context and {target_agent}'s thoughts provided, would {target_agent} do the next actions or say the next utterances described above? Let's think step by step and give the final answer."
        word_mapping = {'a': "Very Likely (Around 90%)", 'b': "Likely (Around 70%)", 'c': "Somewhat Likely (Around 60%)", 'd': "Somewhat Unlikely (Around 25%)", 'e': "Unlikely (Around 20%)", 'f': "Very Unlikely (Below 10%)"}
        score_mapping = {'a': 3, 'b': 2.5, 'c': 2, 'd': 1, 'e': 0.5, 'f': 0.001}
        multiple_choice_options = ""
        for k, v in word_mapping.items():
            multiple_choice_options += f"({k}) {v}\n"
        if self.args.input_is_chat:
            likelihood_prompts = [f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis}\n</{target_agent}'s thoughts>\n\n<response>\n{action}\n</response>\n\n{question}\n{multiple_choice_options}" for hypothesis in existing_hypotheses]
        else:
            likelihood_prompts = [f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis}\n</{target_agent}'s thoughts>\n\n<next action>{action}</next action>\n<note>{perception_history[-1]['action']}</note>\n\n{question}\n{multiple_choice_options}" for hypothesis in existing_hypotheses]
        # 512 truncates the visible chain of thought this prompt explicitly asks
        # for ("Let's think step by step ... then give the answer option with
        # prefix 'Answer:'"), so the answer line never arrives and every verdict
        # parses as a failure. Transport fix, not a scoring change.
        raw_predictions = self.tracer_model.batch_interact(likelihood_prompts, temperature=0, system_prompts=system_prompt, max_tokens=4096)
        reasonings = []
        answers = []
        for response in raw_predictions:
            reasoning, a = extract_answer_span(response)
            reasonings.append(reasoning)
            answers.append(a)
        # Instrumentation only: map_response_to_score returns 0.001 both for a
        # genuine 'f' and for a parse failure (tracer.py:583), so the two are
        # indistinguishable downstream. Record the matched letter separately --
        # a run that looks like unanimous 'Very Unlikely' may just be a parser
        # that never matched anything.
        letters = [map_response_to_letter(j, score_mapping) for j in answers]
        scored = [map_response_to_score(j, score_mapping) for j in answers]
        parse_mask = [x is None for x in scored]
        parsed_vals = [x for x in scored if x is not None]
        # A particle whose verdict did not parse gets the mean of the parsed
        # scores -- i.e. "no information", neutral against its peers -- instead
        # of the old 0.001, which asserted confident disbelief. If nothing
        # parsed the step carries no evidence at all and the weights stay flat.
        neutral = float(np.mean(parsed_vals)) if parsed_vals else 0.0
        raw_scores = np.array([neutral if x is None else x for x in scored], dtype=float)
        weights = softmax(raw_scores)

        results = {
            'prompts': likelihood_prompts,
            'raw_predictions': raw_predictions,
            'reasonings': reasonings,
            'raw_scores': raw_scores,
            'weights': weights,
            'answers': answers,
            'letters': letters,
            'parse_mask': parse_mask,
            'parse_failures': sum(parse_mask),
        }

        return results


    def split_and_merge(self, hypotheses, context_and_perception_str=None):
        """PHASE 4: split (source) then merge (sink), in the SAME step.

        Merge runs immediately after split rather than as a later stage: the
        1.5*N population cap needs a sink in place or it binds every step and
        split's output is discarded as fast as it is produced.

        Split children receive new, mutually exclusive anchors and therefore
        new roots (anchor==root invariant) -- this is what makes split a second
        source term alongside 3e's perturbation.

        ACCUMULATOR: children SPLIT the parent's accumulated log-weight, they do
        not inherit it. raw_accumulator is an unnormalized log-weight; copying
        it to each of n children would multiply the evidence by n, the same
        double-counting that makes "carry weight through resampling" wrong.
        Subtracting log(n) conserves total mass in log space and keeps
        ancestral-mass reversals correct.
        """
        trace_log.set_stage('split')
        info = {'split_fired': 0, 'children': 0, 'merged': 0, 'anchor_collapse': [],
                'expanded': 0, 'split_net': 0, 'net_new_roots': 0}
        n_children = int(getattr(self.args, 'split_children', 2))
        wq = float(getattr(self.args, 'split_weight_quantile', 0.20))

        recs = []
        for i, h in enumerate(hypotheses.hypotheses):
            recs.append(trace_log.ParticleRecord(
                particle_id=h.particle_id, weight=float(hypotheses.weights[i]),
                likelihood_rank=getattr(h, '_likelihood_rank', None)))
        idxs, wc, dc = trace_log.split_candidates(recs, weight_quantile=wq)
        info['weight_condition'], info['rank_condition'] = wc, dc
        if not idxs:
            return hypotheses, info

        # one call per splitting particle: mutually exclusive refinements
        live = [h.anchor for h in hypotheses.hypotheses if h.anchor]
        exclude = "\n".join(f"- {a}" for a in dict.fromkeys(live)) or "- (none)"
        target = hypotheses.target_agent
        sys_p = (
            f"Refine one account of what {target} wants into {n_children} MORE SPECIFIC and "
            f"MUTUALLY EXCLUSIVE versions.\n\n"
            f"Each must be a genuine refinement of the parent commitment -- not a restatement, "
            f"and not a different commitment altogether. They must be incompatible with each "
            f"other: {target} can hold at most one.\n\n"
            f"Avoid duplicating any commitment already in play:\n{exclude}\n\n"
            # The frame above ("what {target} wants") was always right; nothing
            # enforced it, so the model answered with the next conversational
            # move instead of an aim. State the form as a rule, the way the
            # rank scorer states its rules.
            f"A COMMITMENT is what {target} is trying to achieve, stated as {target}'s own aim.\n"
            f"- NEVER a question, and never phrased as one.\n"
            f"- NEVER a request, instruction or challenge directed at another character. "
            f"\"Ask X to ...\", \"Tell X ...\", \"Get X to ...\" are all wrong.\n"
            f"- NEVER a description of what {target} says or does. Name the END, not the move.\n"
            f"- At most {COMMITMENT_MAX_WORDS} words. Plain verb phrase, e.g. "
            f"\"Avenge her mother\", \"Earn Katara's trust\".\n\n"
            + standard_rule(target, getattr(self.args, 'standard_prompt', None)) +
            f"Children may share an aim and be genuinely exclusive because their "
            f"standards differ.\n\n"
            f"Answer exactly:\n" +
            "\n".join(f"{k+1}. COMMITMENT: <clause> | BELIEF: <one sentence> "
                       f"| STANDARD: <what would settle it for them>"
                       for k in range(n_children)))
        prompts = [f"<parent commitment>\n{hypotheses.hypotheses[i].anchor}\n</parent commitment>\n\n"
                   f"<parent account>\n{hypotheses.hypotheses[i].text}\n</parent account>"
                   for i in idxs]
        raws = self.tracer_model.batch_interact(prompts, system_prompts=sys_p,
                                                temperature=0.7, max_tokens=1024, stage='split')

        texts = list(hypotheses.texts)
        weights = [float(w) for w in hypotheses.weights]
        anchors = [h.anchor for h in hypotheses.hypotheses]
        accs = list(hypotheses.accumulators)
        parents = list(hypotheses.hypotheses)
        new_anchor_for = {}
        new_standard_for = {}
        # indices produced by each splitting parent, so the NET outcome can be
        # classified after merge has had its turn on the same step.
        split_groups = {}

        for i, raw in zip(idxs, raws):
            pairs = [(m.group(1).strip(), m.group(2).strip(), (m.group(3) or "").strip())
                     for m in re.finditer(
                         r"COMMITMENT\s*:\s*(.+?)\s*\|\s*BELIEF\s*:\s*(.+?)"
                         r"(?:\s*\|\s*STANDARD\s*:\s*(.+?))?\s*$", raw, re.I | re.M)][:n_children]
            # Drop children whose commitment is not an aim. A rejected child
            # costs one split; an accepted bad one founds a root and persists.
            kept = []
            for c, b, sd in pairs:
                ok, why = valid_commitment(c, enabled=not getattr(self.args,'legacy_form',False))
                if ok:
                    kept.append((c, b, sd))
                else:
                    info.setdefault('rejected_commitments', []).append({'clause': c, 'reason': why})
            pairs = kept
            if len(pairs) < 2:
                # Splitting into one child is an expand, not a split, and a
                # parent that only yields rejects should not be touched at all.
                continue
            share = weights[i] / len(pairs)
            acc_par = accs[i] if accs[i] is not None else 0.0
            acc_child = acc_par - math.log(len(pairs))     # SPLIT, not copied
            texts[i], weights[i], accs[i] = pairs[0][1], share, acc_child
            new_anchor_for[i] = pairs[0][0]
            new_standard_for[i] = pairs[0][2]
            for c, b, sd in pairs[1:]:
                texts.append(b); weights.append(share); anchors.append(c)
                accs.append(acc_child); parents.append(hypotheses.hypotheses[i])
                new_anchor_for[len(texts) - 1] = c
                new_standard_for[len(texts) - 1] = sd
            split_groups[hypotheses.hypotheses[i].particle_id] = (
                [i] + list(range(len(texts) - (len(pairs) - 1), len(texts))))
            info['split_fired'] += 1
            info['children'] += len(pairs)

        if not info['split_fired']:
            return hypotheses, info

        for k, a in new_anchor_for.items():
            if k < len(anchors):
                anchors[k] = a
        out = HypothesesSetV3(hypotheses.target_agent, hypotheses.contexts, hypotheses.perceptions,
                              texts, np.array(weights, dtype=float),
                              parent_hypotheses=parents, accumulators=accs)
        out.update_anchors(anchors)          # new anchors found new roots
        for k in new_anchor_for:
            if k < len(out.hypotheses):
                out.hypotheses[k].split_child = True
                # update_anchors() founds the root; the standard is set after,
                # so a child does not inherit the parent's settling condition
                # for a commitment it no longer holds.
                out.hypotheses[k].update_standard(new_standard_for.get(k))
                out.hypotheses[k].note_operator('split')

        group_roots = {pid: {out.hypotheses[k].root_id for k in grp
                             if k < len(out.hypotheses)}
                       for pid, grp in split_groups.items()}

        # MERGE, immediately, same step -- the sink for split's output
        out, merged, collapses = self.merge_similar(out)
        info['merged'] = merged
        info['anchor_collapse'] = collapses

        # SPLIT vs EXPAND, decided by what SURVIVED the merge.
        # A split is a PARTITION: the parent's hypothesis space divided into
        # alternatives that cannot both hold. If merge absorbs all but one
        # child, the parent was not partitioned -- it was narrowed, which is a
        # different alteration and must not be counted as a source of
        # diversity. Measured before this distinction existed: split+merge
        # netted -0.59 roots per firing across 22 firings, never once positive,
        # because merge ate the children on the step that made them.
        live = {h.root_id for h in out.hypotheses}
        for pid, roots in group_roots.items():
            surviving = roots & live
            if len(surviving) == 1:
                info['expanded'] += 1
                for h in out.hypotheses:
                    if h.root_id in surviving:
                        h.note_operator('expand')
            elif len(surviving) >= 2:
                info['split_net'] += 1
        info['net_new_roots'] = len(live - {h.root_id for h in hypotheses.hypotheses})
        return out, info

    def merge_similar(self, hypotheses):
        """Absorb near-duplicate particles. Threshold is a Jaccard percentile
        resolved from THIS step's own pairwise distribution -- a fixed cosine
        0.90 absorbed 26 of 34 known-DIFFERENT pairs on this data."""
        hyps = hypotheses.hypotheses
        n = len(hyps)
        if n < 3:
            return hypotheses, 0, []
        pairs = [jaccard_similarity(hyps[i].text, hyps[j].text)
                 for i in range(n) for j in range(i + 1, n)]
        pct = float(getattr(self.args, 'merge_percentile', 95.0))
        thr = trace_log.resolve_merge_threshold(pairs, pct)
        if thr is None:
            return hypotheses, 0, []
        absorbed, collapses = set(), []
        for i in range(n):
            if i in absorbed:
                continue
            for j in range(i + 1, n):
                if j in absorbed:
                    continue
                if jaccard_similarity(hyps[i].text, hyps[j].text) >= thr:
                    keep, drop = (i, j) if float(hypotheses.weights[i]) >= float(hypotheses.weights[j]) else (j, i)
                    # differing anchors reaching merge threshold = the anchor did
                    # not hold. Record it; do not silently discard one.
                    if hyps[keep].anchor != hyps[drop].anchor:
                        collapses.append({'kept': hyps[keep].anchor, 'lost': hyps[drop].anchor})
                    absorbed.add(drop)
        if not absorbed:
            return hypotheses, 0, []
        keep_idx = [i for i in range(n) if i not in absorbed]
        for i in absorbed:                      # absorbed commitments leave the population
            self.retire(hyps[i], hypotheses.weights[i])
        w = []
        for i in keep_idx:
            extra = sum(float(hypotheses.weights[j]) for j in absorbed
                        if jaccard_similarity(hyps[i].text, hyps[j].text) >= thr)
            w.append(float(hypotheses.weights[i]) + extra)
        tot = sum(w) or 1.0
        w = [x / tot for x in w]
        out = HypothesesSetV3(hypotheses.target_agent, hypotheses.contexts, hypotheses.perceptions,
                              [hyps[i].text for i in keep_idx], np.array(w),
                              parent_hypotheses=[hyps[i] for i in keep_idx],
                              anchors=[hyps[i].anchor for i in keep_idx],
                              accumulators=[hyps[i].raw_accumulator for i in keep_idx])
        for dst, i in zip(out.hypotheses, keep_idx):
            dst.root_id = hyps[i].root_id          # survivor keeps its root
            dst.note_operator('merge')
        return out, len(absorbed), collapses

    def retire(self, h, weight=0.0):
        """Remember a commitment that is leaving the population.

        Called from every path a hypothesis can exit by, not just expiry. That
        distinction is the whole feature: on v6_bb_Rodriguez expiry fired on
        1 step of 80 while merge fired on 19, split on 16 and perturb on 13, so
        a cache fed only by expiry stayed ~2% populated and revival never had
        anything to offer. Turnover happens through merge and replacement.

        Keyed by anchor, holding PEAK weight -- a commitment that once led and
        faded is a better revival candidate than one that never left the floor.

        Defensive about the attribute because several call sites build a Tracer
        without running the full __init__ (test_expiry.py among them).
        """
        if not hasattr(self, '_retired'):
            self._retired = {}
        if not getattr(h, 'anchor', None):
            return
        prev = self._retired.get(h.anchor, {})
        self._retired[h.anchor] = {
            'anchor': h.anchor, 'text': getattr(h, 'text', ''), 'root_id': getattr(h, 'root_id', None),
            'peak': max(float(prev.get('peak', 0.0)), float(weight or 0.0)),
            'revivals': int(prev.get('revivals', 0)),
        }
        # Bounded: the pool is scored on revival, so an unbounded one would grow
        # the prompt without bound. Keep the strongest.
        cap = int(getattr(self.args, 'retired_cap', 40))
        if len(self._retired) > cap:
            for a in sorted(self._retired, key=lambda x: self._retired[x]['peak'])[:len(self._retired) - cap]:
                del self._retired[a]

    def revive_retired(self, current_action, live_anchors, k=1, ctx=""):
        """Bring a retired commitment back, if it explains what just happened.

        Reviving is gated on the same absolute-fit test as minting: the retired
        candidates are ranked against a null, and one only returns if it beats
        "no particular commitment". Without that gate this would be an
        oscillator -- expiry retires the weak, revival restores them, forever.

        Cheaper than minting (one ranking call, no generation) and it returns the
        SAME root, so a commitment that goes quiet and comes back reads as one
        hypothesis with a gap rather than two that happen to agree.
        """
        if not hasattr(self, '_retired'):
            self._retired = {}
        pool = [v for a, v in self._retired.items() if a not in set(live_anchors or [])]
        if not pool or not current_action:
            return []
        pool.sort(key=lambda v: -float(v.get('peak', 0.0)))
        pool = pool[:8]
        n = len(pool)
        slate = [v['anchor'] for v in pool] + [
            f"{self.target_agent} has no particular commitment here beyond responding to "
            f"the immediate situation."]
        block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(slate))
        sys_p = (
            f"You judge which standing commitments, if any, would have PREDICTED the action "
            f"{self.target_agent} just took.\n\n"
            f"Rank all {n+1} from BEST to WORST predictor of that action, then score each 0-100.\n"
            f"Item {n+1} is the null: it wins when none of the others explains the action.\n\n"
            f"Answer in exactly this format:\n"
            f"RANKING\n1st: <number>\n...\n{n+1}th: <number>\n\n"
            f"ALLOCATION\n1: <0-100>\n...\n{n+1}: <0-100>\n\nREASONING\n<one sentence>")
        prompt = (f"<previous context>\n{(ctx or '')[-6000:]}\n</previous context>\n\n"
                  f"<previously held commitments>\n{block}\n</previously held commitments>\n\n"
                  f"<observed action>\n{current_action}\n</observed action>")
        raw = self.tracer_model.interact(prompt, system_prompt=sys_p, temperature=0,
                                         max_tokens=1024, stage='revive')
        alloc, err = parse_allocation(raw, n + 1)
        if alloc is None:
            return []
        order, _ = parse_ranking(raw, n + 1)
        scores = map_allocation_to_hypotheses(alloc, order)[0] if order else alloc
        baseline = float(scores[n])
        winners = [(float(scores[i]) - baseline, pool[i]) for i in range(n)
                   if float(scores[i]) > baseline]
        winners.sort(key=lambda t: -t[0])
        out = []
        for margin, v in winners[:k]:
            v = dict(v)
            v['margin'] = margin
            out.append(v)
            self._retired.pop(v['anchor'], None)
        if out:
            print(Panel("\n".join(f"{v['anchor']}  (peak {v['peak']:.2f}, margin +{v['margin']:.0f})"
                                  for v in out),
                        title="Revived from cache", style="green", box=box.SIMPLE_HEAD))
        return out

    def perturb_anchored(self, hypotheses, idxs, context_and_perception_str=None,
                         surprising_action=None):
        """PHASE 3e: replace a particle's commitment with a genuinely different one.

        The ONLY source term in the system. Roots are founded at initialization
        and thereafter only die (verified monotone across 5 clean runs), so
        without this the population decays 8 -> 1..4 in 28 steps and nothing
        can bend the slope -- seeding only sets the intercept.

        Mints a new anchor AND therefore a new root, via the anchor==root
        invariant. A paraphrase must NOT mint: it founds no new hypothesis, and
        minting on it would inflate the very metric that gates this operator.

        Acceptance tests COHERENCE with the observation history, never
        likelihood of the current action -- a likelihood floor would reject
        exactly the divergent candidates and turn repair into a fifth remover.
        """
        trace_log.set_stage('perturb')
        target_agent = hypotheses.target_agent
        live = [h.anchor for h in hypotheses.hypotheses if h.anchor]
        # 'rejected' = judged CONTRADICTS by the coherence gate.
        # 'unparsed'  = the proposal or the verdict did not parse.
        # Conflating them would corrupt the acceptance rate, which is the metric
        # that tells us whether the gate is testing anything -- the same class of
        # bug as parse failure aliasing onto bucket 'f'.
        results = {'accepted': [], 'rejected': [], 'unparsed': [], 'proposed': {}, 'revived': []}
        if not idxs:
            return results

        # REVIVE BEFORE MINTING. If a commitment this trace already held would
        # have predicted the action, bringing it back beats inventing a new one:
        # it is one ranking call instead of a generation plus a coherence gate,
        # and it restores the original root so the hypothesis reads as having
        # gone quiet rather than as a fresh idea that happens to match.
        if getattr(self.args, 'revive_retired', False) and surprising_action:
            live = [h.anchor for h in hypotheses.hypotheses if h.anchor]
            for v in self.revive_retired(surprising_action, live, k=max(1, len(idxs) // 2),
                                         ctx=context_and_perception_str or ""):
                if not idxs:
                    break
                i = idxs.pop(0)
                h = hypotheses.hypotheses[i]
                h.update_text(v.get('text') or h.text)
                h.update_anchor(v['anchor'], revision=True)
                if v.get('root_id'):
                    h.root_id = v['root_id']      # the SAME hypothesis returning
                h.note_operator('revive')
                results['revived'].append({'index': i, 'anchor': v['anchor'],
                                           'peak': v.get('peak'), 'margin': v.get('margin')})
                results['accepted'].append(i)
            if not idxs:
                return results

        exclude = "\n".join(f"- {a}" for a in dict.fromkeys(live)) or "- (none recorded)"
        k = len(idxs)
        # ONE call for all k replacements, not k independent calls.
        # Independent generation let the NEW commitments duplicate each other:
        # 7 replacements once yielded "maintain his public image and reputation",
        # "maintain his public image" and "preserve his reputation" -- about 3
        # distinct ideas. Each call avoided the commitments already in play but
        # could not see its siblings. Split gets this right by generating its
        # children together; perturbation now does the same.
        # ABDUCTIVE FRAME. "What does X want?" is answerable from the surface -- the
        # honest summary of a work thread is "define numerical thresholds", which
        # describes the message rather than a reason to send it. Asking what would have
        # to be TRUE for these messages to be worth sending forces a latent cause, and
        # the role prior gives the proposer somewhere to generate from other than the
        # words in front of it.
        use_prior = getattr(self.args, 'infer_motive', False)
        prior_block = ""
        if use_prior:
            rp = self.infer_role_prior()
            if rp:
                prior_block = (f"\n\nWhat is known about {target_agent}'s position, established "
                               f"from the whole record:\n{rp}\n"
                               f"Use this as your starting point. A commitment should follow from "
                               f"this SEAT and STAKE, not from the wording of any one message.")
        ask = (f"What would have to be TRUE about {target_agent} for the things they have said "
               f"and done here to be worth saying and doing? Propose {k} different answers."
               if use_prior else
               f"Given the observations {target_agent} has had, propose {k} different standing "
               f"commitments -- different things they are trying to achieve -- each still "
               f"consistent with everything they have observed.")
        sys_p = (
            f"You propose {k} ALTERNATIVE accounts of what {target_agent} wants.\n\n"
            f"{ask}{prior_block}\n\n"
            f"Rules:\n"
            f"- The {k} new commitments must be MUTUALLY EXCLUSIVE WITH EACH OTHER. "
            f"{target_agent} can hold at most one. Rewording the same goal is not a "
            f"different goal.\n"
            f"- They must also differ from every commitment already in play:\n{exclude}\n"
            f"- None may contradict anything {target_agent} demonstrably observed.\n"
            f"- Change what they WANT, not the wording of what they believe.\n"
            + (f"- A commitment is the REASON the messages are worth sending, not a description "
               f"of what they accomplish. Do NOT name the specific document, feature, number, "
               f"metric or object under discussion -- if your clause could only make sense in "
               f"this one thread, it is a restatement, not a motive.\n" if use_prior else "")
            +
            # Same form rule as split. Perturbation is the only source term, so
            # a malformed commitment here founds a root that nothing removes.
            f"- A COMMITMENT is what {target_agent} is trying to achieve, stated as "
            f"{target_agent}'s own aim. NEVER a question. NEVER a request or instruction "
            f"aimed at another character (\"Ask X to ...\", \"Tell X ...\"). NEVER a "
            f"description of what {target_agent} says or does -- name the END, not the move. "
            f"At most {COMMITMENT_MAX_WORDS} words.\n\n"
            # WHY A THIRD FIELD. Every other rule here asks what the target
            # WANTS. Two people can want the same thing and still disagree
            # about what would show they had got it, and that disagreement has
            # nowhere to go in a wants-only representation -- the proposer
            # rounds it to the nearest motive, and the nearest motive to a
            # senior person pressing a point is status. Measured: on the
            # Bloomfield purple dispute the filter built "Establish their
            # authority" 0.27 -> 0.67 out of four consecutive QUESTIONS,
            # because the real difference (a count is right when it matches
            # what we deliver / when two labellers reproduce it) is a
            # criterion, not a desire.
            + standard_rule(target_agent, getattr(self.args, 'standard_prompt', None)) +
            f"\n"
            f"Answer exactly {k} numbered lines:\n" +
            "\n".join(f"{j+1}. COMMITMENT: <short clause> | BELIEF: <one or two sentences> "
                       f"| STANDARD: <what would settle it for them>"
                       for j in range(k)))
        ctx = context_and_perception_str or ""
        block = "\n".join(
            f"{j+1}. currently: {hypotheses.hypotheses[i].anchor}"
            for j, i in enumerate(idxs))
        # On the surprise path the whole point is the action nothing explained,
        # and it is NOT in ctx (which stops at the previous step). Showing it and
        # naming the job is what turns a generic diversity repair into a targeted
        # one: the commitment being asked for is the one that would have
        # predicted THIS.
        action_block = ""
        if surprising_action:
            action_block = (
                f"\n\n<the action none of the current accounts explains>\n{surprising_action}\n"
                f"</the action none of the current accounts explains>\n\n"
                f"At least one of your {k} proposals must be a commitment that WOULD have "
                f"predicted that action. It may contradict the accounts listed above -- "
                f"{target_agent} may have changed what they want.")
        prompt = (f"<previous context>\n{ctx}\n</previous context>{action_block}\n\n"
                  f"<the {k} accounts to replace>\n{block}\n</the {k} accounts to replace>")
        raw_one = self.tracer_model.interact(
            prompt, system_prompt=sys_p, temperature=0.7, max_tokens=2048, stage='perturb')
        # STANDARD is optional in the pattern so a model that omits it still
        # yields a usable commitment rather than an unparsed slot.
        found = [(m.group(1), m.group(2), (m.group(3) or "").strip())
                 for m in re.finditer(
                     r"COMMITMENT\s*:\s*(.+?)\s*\|\s*BELIEF\s*:\s*(.+?)"
                     r"(?:\s*\|\s*STANDARD\s*:\s*(.+?))?\s*$", raw_one, re.I | re.M)]
        # drop any that duplicate an earlier one in this same batch
        uniq, seen_c = [], set()
        for c, b, sd in found:
            key = c.strip().lower().rstrip('.')
            ok, why = valid_commitment(c, enabled=not getattr(self.args,'legacy_form',False))
            if not ok:
                results.setdefault('rejected_commitments', []).append(
                    {'clause': c.strip(), 'reason': why})
                continue
            if key and key not in seen_c:
                seen_c.add(key)
                uniq.append((c.strip(), b.strip(), sd))
        results['proposed_raw'] = len(found)
        results['proposed_unique'] = len(uniq)
        results['proposed_with_standard'] = sum(1 for _, _, sd in uniq if sd)
        raws = [None] * len(idxs)
        for j in range(min(len(uniq), len(idxs))):
            raws[j] = (f"COMMITMENT: {uniq[j][0]}\nBELIEF: {uniq[j][1]}"
                       + (f"\nSTANDARD: {uniq[j][2]}" if uniq[j][2] else ""))

        cands = []
        for i, r in zip(idxs, raws):
            # a slot with no proposal: the batch returned fewer distinct
            # commitments than there were particles to replace. Keep the
            # original particle rather than inventing a duplicate for it.
            if not r:
                results['unparsed'].append(i)
                continue
            m_c = re.search(r"COMMITMENT\s*:\s*(.+)", r, re.I)
            m_b = re.search(r"BELIEF\s*:\s*(.+?)(?=\nSTANDARD\s*:|$)", r, re.I | re.S)
            m_s = re.search(r"STANDARD\s*:\s*(.+)", r, re.I | re.S)
            if not m_c or not m_b:
                results['unparsed'].append(i)
                continue
            cands.append((i, m_c.group(1).strip(), m_b.group(1).strip(),
                          m_s.group(1).strip() if m_s else None))
        if not cands:
            return results

        # coherence gate -- contradiction, not surprise
        gate_sys = (
            f"Decide whether each proposed account CONTRADICTS what {target_agent} has "
            f"demonstrably observed.\n\n"
            f"Reject ONLY for contradiction with the observed record. Do NOT reject an account "
            f"for being surprising, unlikely, or unflattering -- a genuinely different account "
            f"is the point.\n\nAnswer one line each:\n1: COHERENT or CONTRADICTS\n...")
        block = "\n".join(f"{n+1}. COMMITMENT: {c} | BELIEF: {b}"
                          + (f" | STANDARD: {sd}" if sd else "")
                          for n, (_, c, b, sd) in enumerate(cands))
        gate_raw = self.tracer_model.interact(
            f"<observed record>\n{ctx}\n</observed record>\n\n<proposed accounts>\n{block}\n"
            f"</proposed accounts>", system_prompt=gate_sys, temperature=0,
            max_tokens=1024, stage='perturb')
        verdicts = re.findall(r"^\s*\**\s*(\d+)\s*\**\s*[:.\)]\s*\**\s*(COHERENT|CONTRADICTS)",
                              gate_raw, re.I | re.M)
        vmap = {int(n): v.upper() for n, v in verdicts}
        results['gate_unparsed'] = sum(1 for n in range(1, len(cands) + 1) if n not in vmap)

        for pos, (i, commitment, belief, standard) in enumerate(cands, start=1):
            results['proposed'][i] = commitment
            verdict = vmap.get(pos)
            if verdict is None:
                # no verdict parsed: keep the original particle, but record it as
                # unparsed rather than as a coherence judgement either way
                results['unparsed'].append(i)
                continue
            if verdict == 'CONTRADICTS':
                results['rejected'].append(i)
                continue
            h = hypotheses.hypotheses[i]
            self.retire(h, hypotheses.weights[i])    # the commitment being replaced
            h.update_text(belief)
            # revision=True: this REPLACES a live particle's commitment, which is
            # what anchor_revisions is meant to count. Nothing passed it before,
            # so the counter read 0 everywhere and anchor staleness was
            # unmeasurable. Split is deliberately not a revision -- its children
            # are new hypotheses, not a changed mind.
            # the standard travels with the commitment it settles
            h.update_anchor(commitment, revision=True, standard=standard)
            h.note_operator('perturb')
            results['accepted'].append(i)
        # REBIRTH AT FAIR SHARE. A replacement inherited the weight of the
        # particle it replaced, and the expiry/collapse triggers select exactly
        # the particles sitting at the floor -- so every minted commitment was
        # born at ~eps and had to climb 8x just to reach parity. Measured on
        # exp_3: median birth mass 0.028 for perturbation's mints against 0.189
        # for split's children, 15 of 27 born below fair share. Under
        # w_t ~ w_{t-1}^alpha * L_t^beta a floor-level prior cannot recover
        # quickly however well the new commitment scores, so root-mass never
        # recovered, the trigger stayed hot, and the operator re-fired on
        # consecutive steps chasing a collapse its own mechanism guaranteed.
        #
        # A new hypothesis is not a weak version of the old one -- it is a new
        # claim that has not yet been tested. It enters at fair share, and the
        # mass comes proportionally from the rest of the population.
        #
        # MEASURED NET-NEGATIVE, SO THIS IS OFF BY DEFAULT. exp_4 vs exp_3 did
        # raise median birth mass 0.047 -> 0.113 as intended, but the mass has to
        # come from the established hypotheses, which lowers root-mass ESS across
        # the population and trips the collapse trigger MORE often: 9 -> 13
        # firings, 3 -> 6 consecutive re-fires, mint survival 59% -> 32%, argmax
        # churn 8 -> 5. The reasoning that floor-born mints kept the trigger hot
        # was wrong -- the fix fed the loop it was meant to break. Kept behind the
        # flag as the evidence.
        #
        # Deliberate, logged mass move -- NOT asserted on, same class as the
        # floor. sync_accumulator() re-keys the prior from these weights, so the
        # reset reaches the accumulator and is not undone on the next step.
        if results['accepted'] and getattr(self.args, 'rebirth_at_fair_share', False):
            before = [float(x) for x in hypotheses.weights]
            n = len(before)
            if n:
                fair = 1.0 / n
                w = list(before)
                for i in results['accepted']:
                    if i < n:
                        w[i] = fair
                tot = sum(w)
                if tot > 0:
                    w = [x / tot for x in w]
                    hypotheses.update_weights(np.array(w, dtype=float))
                    results['mass_moved_by_rebirth'] = round(
                        0.5 * sum(abs(a - b) for a, b in zip(w, before)), 5)
                    results['rebirth_weight'] = round(fair, 5)
        hypotheses.texts = [h.text for h in hypotheses.hypotheses]
        hypotheses.anchors = [h.anchor for h in hypotheses.hypotheses]
        return results

    def extract_anchors(self, hypotheses_texts, target_agent=None):
        """One call: distil each hypothesis to its distinguishing commitment.

        Kept separate from initialization so 3a and 3b stay independently
        measurable -- this works whether or not goal-level seeding is on.
        """
        trace_log.set_stage('anchor')
        target_agent = self.target_agent if target_agent is None else target_agent
        n = len(hypotheses_texts)
        block = "\n".join(f"{i+1}. {h.strip()}" for i, h in enumerate(hypotheses_texts))
        system_prompt = (
            f"Each numbered item is a hypothesis about {target_agent}'s mind. For each, state in "
            f"ONE short clause the single thing {target_agent} WANTS or is TRYING TO ACHIEVE under "
            f"that hypothesis -- its distinguishing commitment, not a summary.\n\n"
            f"Rules:\n- Name a goal, not a belief about facts.\n"
            f"- Make them as mutually distinguishable as the hypotheses allow.\n"
            f"- If two hypotheses share a goal, give them the same clause; do not invent a "
            f"difference that is not there.\n\n"
            # The weaker phrasing produced restatements: "Prevent mislabeling"
            # -> "No images are mislabeled", which is the aim's success
            # condition, not evidence, and carries no axis. The perturb path
            # got usable standards from the same model ("a reconciliation
            # report showing zero discrepancies between the dashboard's ratios
            # and the sensor logs"), so the difference is the instruction, not
            # the model. Name the failure explicitly and demand an artefact.
            + standard_rule(target_agent, getattr(self.args, 'standard_prompt', None)) +
            f"  Name who would produce it and what it would show. Two hypotheses may "
            f"share a goal and differ ONLY here.\n\n"
            f"Output exactly {n} lines:\n1. <clause> | STANDARD: <what would settle it>\n"
            f"...\n{n}. <clause> | STANDARD: <what would settle it>")
        # Thinking tokens count against max_output_tokens, so a reasoning model
        # spends this budget before writing anything: measured 1021 thinking
        # tokens and visible=None on gemini-2.5-pro at 1024, i.e. EVERY anchor
        # extraction returned empty and fell back silently. Same defect as
        # FINDINGS 1.2, which was about the likelihood scorer -- it recurs
        # anywhere a fixed budget meets a model that reasons first, and 2.5-pro
        # cannot disable thinking. Configurable so a caller using a reasoning
        # model can raise it without editing this line.
        raw = self.tracer_model.interact(
            block, system_prompt=system_prompt, temperature=0,
            max_tokens=int(getattr(self.args, 'anchor_max_tokens', 4096)),
            stage='anchor')
        parsed = capture_and_parse_ordered_list(raw)
        if len(parsed) != n:
            print(Panel(f"anchor extraction returned {len(parsed)} of {n}", style="red", box=box.SIMPLE_HEAD))
            parsed = (parsed + [None] * n)[:n]
        # Split the standard off the clause. The anchor must stay a bare
        # commitment -- it is the root's identity and is compared by string
        # elsewhere (merge, exclusion lists, the split parent block), so
        # leaving "| STANDARD: ..." on it would change what those compare.
        self._last_standards = []
        clean = []
        for c in parsed:
            if not c:
                clean.append(c); self._last_standards.append(None); continue
            m = re.split(r"\s*\|\s*STANDARD\s*:\s*", c, maxsplit=1, flags=re.I)
            clean.append(m[0].strip())
            self._last_standards.append(m[1].strip() if len(m) > 1 else None)
        parsed = clean
        return [p.strip() if p else None for p in parsed]

    def prompt_likelihood_comparative(self, existing_hypotheses: list, context_history: list, perception_history: list, action: str, target_agent: str = None):
        """Score all N hypotheses against the observed action in ONE call.

        Scoring each hypothesis in isolation lets "Very Likely" be defensible
        every time -- measured at 88% bucket 'a' on gold, with only 2 of 28
        steps discriminating at all. Here the hypotheses compete for a fixed
        budget, so the evaluator cannot say yes to everything.

        Two deliberate choices:

        * Ranking is forced, magnitudes are NOT. Requiring a minimum spread
          would manufacture the separation the Phase 1 gate measures, and the
          gate would then be reporting the constraint rather than the
          evaluator. A degenerate 26/25/25/24 allocation must remain *possible*
          so that normalized ESS can still detect it. probe_scorer.py is the
          guard: identical hypotheses must stay flat.
        * Verdict first, explanation after. A truncated response then still
          carries the answer, inverting the failure mode from "lost everything"
          to "lost the reasoning".
        """
        trace_log.set_stage('likelihood')
        target_agent = self.target_agent if target_agent is None else target_agent
        n = len(existing_hypotheses)

        pruned_context_history = deepcopy(context_history)
        pruned_perception_history = deepcopy(perception_history)
        pruned_context_history[-1]['action'] = None
        pruned_perception_history[-1]['action'] = None
        if len(pruned_context_history) > 1:
            context_and_perception_str = self.interleave_context_and_perception(pruned_context_history, pruned_perception_history)
        elif pruned_context_history[-1]['state']:
            context_and_perception_str = f"{pruned_context_history[-1]['state']}"
        else:
            context_and_perception_str = ""

        # BASELINE / NULL HYPOTHESIS. Likelihoods are normalized to sum to 1,
        # so "every live hypothesis is wrong" is structurally unrepresentable:
        # the best of a bad lot still receives ~1/n and keeps its mass. Measured
        # at the Katara reversal, where none of the five commitments was about
        # whether to kill Yon Rha: top likelihood 1.67x uniform, likelihood ESS
        # 0.763, root-mass ESS 0.759 -- every health metric green at the exact
        # step the filter missed the point.
        #
        # Adding a fixed null to the slate makes fit ABSOLUTE rather than merely
        # relative. Each hypothesis is then credited by how far it beats "no
        # particular commitment", which is the P(a|h)/P(a) correction the scorer
        # never had; and the null ranking at the top is the surprise signal that
        # nothing else in the system can produce.
        # TWO nulls, not one. The single null answers "does any live commitment
        # explain this action", which conflates two very different reasons for
        # no: the action CONTRADICTS the commitments (real surprise, mint), or
        # the action is routine traffic that reveals nothing about what the
        # target is pursuing (mint nothing). On a hand-picked span the second
        # case is rare; on the full 74-set blueberry_size topic only 8 sets
        # carry any sizing vocabulary at all, so surprise fired on consecutive
        # steps through ~250 turns of camera firmware and carnets and the filter
        # minted almost every step. The routine null is what separates them, and
        # it costs one extra slate item rather than another call.
        use_baseline = getattr(self.args, 'baseline_scorer', False)
        n_extra = 2 if use_baseline else 0
        n_sent = n + n_extra
        slate = list(existing_hypotheses)
        if use_baseline:
            slate = slate + [
                f"{target_agent} has no particular commitment here beyond responding to "
                f"the immediate situation.",
                f"This message is routine traffic -- scheduling, logistics, tooling, "
                f"acknowledgements -- and reveals nothing about what {target_agent} is "
                f"trying to achieve."]
        hypothesis_block = "\n".join(
            f"{i + 1}. {h.strip()}" for i, h in enumerate(slate))

        mode = getattr(self.args, 'scorer_mode', 'rank')
        if mode == 'rank':
            # Measured on one gold step where the other framings flatten
            # completely (independent 0-100 -> 90,90,90,90; "eliminate failures"
            # -> 100,100,100,100; both normalized ESS 1.000), ranking first then
            # scoring gives 10,20,50,20 -> 0.735.
            #
            # The difference is cognitive, not formatting: committing to a
            # strict order BEFORE any number forces a contrastive judgement,
            # where "score each" lets every hypothesis be evaluated on its own
            # and pass. Magnitudes stay free -- only the ordering is compelled --
            # so a degenerate near-uniform allocation remains possible and the
            # gate can still detect it. probe_scorer.py is the guard that this
            # does not simply manufacture order on identical inputs.
            system_prompt = (
                f"You compare competing hypotheses about {target_agent}'s mind against an action "
                f"{target_agent} actually took.\n\n"
                f"First rank all {n_sent} hypotheses from BEST to WORST predictor of that action. "
                f"Commit to a strict order -- no ties. Then score each 0-100 for how strongly it "
                f"predicts the action.\n\n"
                f"Rules:\n"
                f"- Judge only prediction of THIS action. Do not reward a hypothesis for being "
                f"detailed, well written, or generally plausible.\n"
                f"- A hypothesis that would equally well predict many other actions is a weak "
                f"predictor of this one.\n"
                f"- If the hypotheses genuinely are equivalent, say so in REASONING and score "
                f"them alike; do not invent a difference that is not there.\n\n"
                f"Answer in exactly this format, ranking FIRST:\n"
                f"RANKING\n1st: <hypothesis number>\n...\n{n_sent}th: <hypothesis number>\n\n"
                f"ALLOCATION\n1: <0-100>\n...\n{n_sent}: <0-100>\n\nREASONING\n"
                f"<what distinguishes the best from the worst>")
            prompt = (
                f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n"
                f"<candidate hypotheses about {target_agent}'s thoughts>\n{hypothesis_block}\n"
                f"</candidate hypotheses about {target_agent}'s thoughts>\n\n"
                f"<observed next action>\n{action}\n</observed next action>")
            raw = self.tracer_model.interact(prompt, system_prompt=system_prompt,
                                             temperature=0, max_tokens=2048, stage='likelihood')
            alloc, err = parse_allocation(raw, n_sent)
            # The allocation keys are ambiguous between rank position and
            # hypothesis index; the RANKING block settles it. Everything
            # downstream -- raw_scores, weights, raw_verdicts, likelihood_rank --
            # is in HYPOTHESIS order, so the resolution happens here and the
            # contract for callers is unchanged.
            order, rank_err = (None, None) if alloc is None else parse_ranking(raw, n_sent)
            if alloc is not None and order is not None:
                scores, keying, agreement = map_allocation_to_hypotheses(alloc, order)
            else:
                scores, keying, agreement = alloc, 'unresolved', None
            rank_info = {'ranking': order, 'alloc_keying': keying,
                         'rank_alloc_agreement': agreement,
                         'rank_unparsed': order is None, 'rank_parse_error': rank_err,
                         'allocation_raw': alloc}
            # Split the null off the slate and re-express every score as a
            # MARGIN over it. A hypothesis that merely redescribes the action
            # sits near the null and earns little; one that genuinely explains
            # it clears the null by a wide gap. SURPRISE is the null winning:
            # no live commitment beats "no particular commitment".
            if use_baseline and scores is not None and len(scores) == n + 2:
                baseline = float(scores[n])
                routine = float(scores[n + 1])
                raw_h = [float(x) for x in scores[:n]]
                margins = [x - baseline for x in raw_h]
                best = max(margins) if margins else 0.0
                # Off-topic: the routine reading beats every commitment AND the
                # null. Nothing to learn and nothing to mint -- the target simply
                # is not pursuing anything here.
                off_topic = (routine >= baseline and routine > max(raw_h or [0.0])) \
                    or is_low_content(action)
                rank_info['baseline_score'] = baseline
                rank_info['routine_score'] = routine
                rank_info['baseline_rank'] = 1 + sum(1 for x in raw_h if x > baseline)
                rank_info['margins'] = margins
                rank_info['best_margin'] = best
                rank_info['off_topic'] = off_topic
                rank_info['surprise'] = (best <= 0.0) and not off_topic
                if best <= 0.0:
                    # Nothing explains the action. The step carries no evidence
                    # about WHICH existing hypothesis is right, so it must not
                    # be allowed to reweight them -- staying flat is the honest
                    # reading, and the mint is what actually responds.
                    scores = [1.0] * n
                else:
                    # Blend a little uniform back in so a hypothesis sitting at
                    # the null is not annihilated outright: it is uninformative,
                    # not refuted, and a hard zero would make the floor the only
                    # thing keeping it alive.
                    blend = float(getattr(self.args, 'baseline_blend', 0.15))
                    pos = [max(m, 0.0) for m in margins]
                    tot = sum(pos) or 1.0
                    scores = [(1.0 - blend) * (x / tot) + blend / n for x in pos]
            elif use_baseline:
                rank_info['surprise'] = None
                rank_info['off_topic'] = None
            if alloc is not None and order is None:
                # Fall back to the positional read rather than inventing an
                # order, but make the fallback visible: it is the old behaviour
                # and it is wrong whenever the model keyed by rank.
                print(Panel(f"rank scorer: RANKING unparsed ({rank_err}) -- "
                            f"falling back to positional allocation",
                            style="yellow", box=box.SIMPLE_HEAD))
            if scores is not None and sum(scores) <= 0:
                return {'prompts': [prompt] * n, 'system_prompt': system_prompt,
                        'raw_predictions': raw,
                        'reasonings': re.split(r"REASONING", raw, flags=re.I)[-1].strip(),
                        'raw_scores': scores, 'weights': np.ones(n) / n,
                        'answers': [str(v) for v in scores], 'letters': [str(v) for v in scores],
                        'parse_mask': [False] * n, 'parse_failures': 0,
                        'allocation': scores, 'allocation_sum': 0.0, 'all_zero': True,
                        'ties': True, **rank_info}
            if scores is None:
                print(Panel(f"rank scorer unparsed ({err})", style="red", box=box.SIMPLE_HEAD))
                return {'prompts': [prompt] * n, 'system_prompt': system_prompt,
                        'raw_predictions': raw, 'reasonings': raw,
                        'raw_scores': [None] * n, 'weights': np.ones(n) / n,
                        'answers': [''] * n, 'letters': [None] * n,
                        'parse_mask': [True] * n, 'parse_failures': n,
                        'allocation': None, 'parse_error': err, **rank_info}
            total = float(sum(scores))
            weights = np.array([v / total for v in scores], dtype=float)
            return {'prompts': [prompt] * n, 'system_prompt': system_prompt,
                    'raw_predictions': raw,
                    'reasonings': re.split(r"REASONING", raw, flags=re.I)[-1].strip(),
                    'raw_scores': scores, 'weights': weights,
                    'answers': [str(v) for v in scores], 'letters': [str(v) for v in scores],
                    'parse_mask': [False] * n, 'parse_failures': 0,
                    'allocation': scores, 'allocation_sum': total,
                    'ties': len(set(scores)) != n, **rank_info}

        if mode == 'independent':
            # Sum-to-100 forces ordering but COMPRESSES magnitude: with n=4 the
            # model anchors near 25 each (measured: identical inputs -> 28/26/24/22,
            # normalized ESS 0.992; distinct inputs -> 0.974, barely different, and
            # worse than the isolated six-bucket scorer's 0.38 on the same inputs).
            # Here all hypotheses are still shown together -- the comparison is what
            # makes "very likely" hard to defend for every one -- but each is scored
            # on a free 0-100 scale, so the range the bucket scale allowed survives.
            system_prompt = (
                f"You judge competing hypotheses about {target_agent}'s mind against an action "
                f"{target_agent} actually took.\n\n"
                f"For each hypothesis, give the probability from 0 to 100 that {target_agent} "
                f"would take this exact action IF that hypothesis were true.\n\n"
                f"Rules:\n"
                f"- Score each hypothesis on its own merits. They need not sum to anything.\n"
                f"- A hypothesis that makes the action almost inevitable scores near 100. One "
                f"that leaves the action unexplained or contradicts it scores near 0. Use the "
                f"full range -- most hypotheses are not equally good.\n"
                f"- Judge only prediction of THIS action. Do not reward a hypothesis for being "
                f"detailed, well written, or generally plausible.\n\n"
                f"Answer in exactly this format, scores FIRST:\n"
                f"ALLOCATION\n1: <0-100>\n2: <0-100>\n...\n{n}: <0-100>\n\nREASONING\n"
                f"<one short line per hypothesis>")
            prompt = (
                f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n"
                f"<candidate hypotheses about {target_agent}'s thoughts>\n{hypothesis_block}\n"
                f"</candidate hypotheses about {target_agent}'s thoughts>\n\n"
                f"<observed next action>\n{action}\n</observed next action>\n\n"
                f"Score each of the {n} hypotheses 0-100 on how well it predicts the observed action.")
            raw = self.tracer_model.interact(prompt, system_prompt=system_prompt,
                                             temperature=0, max_tokens=2048, stage='likelihood')
            alloc, err = parse_allocation(raw, n)
            # An all-zero result is a VERDICT, not a parse failure: the evaluator
            # is saying no hypothesis predicts this action. Flat weights are the
            # right posterior (nothing distinguishes them) but the step must not
            # be recorded as unparsed, or a real unanimous judgement is thrown
            # away as noise.
            if alloc is not None and sum(alloc) <= 0:
                return {'prompts': [prompt] * n, 'raw_predictions': raw,
                        'reasonings': re.split(r"REASONING", raw, flags=re.I)[-1].strip(),
                        'raw_scores': alloc, 'weights': np.ones(n) / n,
                        'answers': [str(v) for v in alloc], 'letters': [str(v) for v in alloc],
                        'parse_mask': [False] * n, 'parse_failures': 0,
                        'allocation': alloc, 'allocation_sum': 0.0,
                        'all_zero': True, 'ties': True}
            if alloc is None:
                print(Panel(f"comparative scores unparsed ({err})", style="red", box=box.SIMPLE_HEAD))
                weights = np.ones(n) / n
                return {'prompts': [prompt] * n, 'raw_predictions': raw, 'reasonings': raw,
                        'raw_scores': [None] * n, 'weights': weights, 'answers': [''] * n,
                        'letters': [None] * n, 'parse_mask': [True] * n,
                        'parse_failures': n, 'allocation': None, 'parse_error': err}
            total = float(sum(alloc))
            weights = np.array([v / total for v in alloc], dtype=float)
            return {'prompts': [prompt] * n, 'raw_predictions': raw,
                    'reasonings': re.split(r"REASONING", raw, flags=re.I)[-1].strip(),
                    'raw_scores': alloc, 'weights': weights,
                    'answers': [str(v) for v in alloc], 'letters': [str(v) for v in alloc],
                    'parse_mask': [False] * n, 'parse_failures': 0,
                    'allocation': alloc, 'allocation_sum': total,
                    'ties': len(set(alloc)) != n}

        system_prompt = (
            f"You compare competing hypotheses about {target_agent}'s mind against an action "
            f"{target_agent} actually took.\n\n"
            f"Distribute exactly 100 points across the {n} hypotheses according to how well each "
            f"one predicts that action. A hypothesis that makes the action expected gets more "
            f"points; one that leaves it unexplained gets fewer.\n\n"
            f"Rules:\n"
            f"- Give every hypothesis a different number. No ties.\n"
            f"- The points must sum to 100.\n"
            f"- Judge only how well each hypothesis predicts THIS action. Do not reward "
            f"hypotheses for being detailed, well written, or generally plausible.\n\n"
            f"Answer in exactly this format, allocation FIRST:\n"
            f"ALLOCATION\n1: <points>\n2: <points>\n...\n{n}: <points>\n\nREASONING\n"
            f"<one short line per hypothesis>")

        prompt = (
            f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n"
            f"<candidate hypotheses about {target_agent}'s thoughts>\n{hypothesis_block}\n"
            f"</candidate hypotheses about {target_agent}'s thoughts>\n\n"
            f"<observed next action>\n{action}\n</observed next action>\n\n"
            f"Distribute 100 points across the {n} hypotheses by how well each predicts the "
            f"observed action.")

        raw = self.tracer_model.interact(prompt, system_prompt=system_prompt,
                                         temperature=0, max_tokens=2048, stage='likelihood')
        alloc, err = parse_allocation(raw, n)
        if alloc is not None and sum(alloc) <= 0:
            alloc, err = None, "allocation sums to zero"
        if alloc is None:
            # The whole step failed to parse: no verdict at all, so the step
            # carries no evidence. Flat weights, every particle flagged -- never
            # imputed as a confident low score.
            print(Panel(f"comparative allocation unparsed ({err})", style="red", box=box.SIMPLE_HEAD))
            weights = np.ones(n) / n
            return {'prompts': [prompt] * n, 'raw_predictions': raw,
                    'reasonings': raw, 'raw_scores': [None] * n, 'weights': weights,
                    'answers': [''] * n, 'letters': [None] * n,
                    'parse_mask': [True] * n, 'parse_failures': n,
                    'allocation': None, 'parse_error': err}

        total = float(sum(alloc))
        # Linear normalization: the weight ratios match the points the evaluator
        # was shown. softmax over these would distort them.
        weights = np.array([v / total for v in alloc], dtype=float)
        return {'prompts': [prompt] * n, 'raw_predictions': raw,
                'reasonings': re.split(r"REASONING", raw, flags=re.I)[-1].strip(),
                'raw_scores': alloc, 'weights': weights,
                'answers': [str(v) for v in alloc], 'letters': [str(v) for v in alloc],
                'parse_mask': [False] * n, 'parse_failures': 0,
                'allocation': alloc, 'allocation_sum': total,
                'ties': len(set(alloc)) != n}

    def propagate(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions: dict) -> HypothesesSetV3:
        """
        Propagate the hypotheses on the target agent using the context, which is the text that does not contain the target agent's actions.
        """
        trace_log.set_stage('propagate')
        prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions)
        target_agent = prop_info["target_agent"]
        context_history = prop_info["context_history"]
        perception_history = prop_info["perception_history"]
        context_and_perception_str = prop_info["context_and_perception_str"]
        new_context = prop_info["new_context"]

        # PHASE 3b: carry the anchor and instruct the model to hold it FIXED.
        # Propagation previously asked only "what did X believe?" given the new
        # context, which invites every particle toward the contextually obvious
        # answer -- convergence by construction. The anchor makes diversity
        # survive propagation rather than depend on luck.
        use_anchor = getattr(self.args, 'use_anchor', False)  # 3b: thread through propagation
        anchors = existing_hypotheses.anchors if use_anchor else [None] * len(existing_hypotheses.texts)
        # The question used to be, in full, "What did {target} believe?" -- asked
        # with <current context> holding what SOMEBODY ELSE just said. Given
        # that, "Katara believes that Aang is genuinely asking about the
        # situation" is a correct answer, and the population filled up with
        # readings of the interlocutor instead of the target's aim: measured
        # 49-78% of top beliefs were "X believes that <other character> ...",
        # and only 29-37% stated any want or intent at all.
        #
        # Note the asymmetry it introduced: `initialize` asks for the thoughts
        # "that led to the action" -- a causal frame that demands an aim -- and
        # propagation dropped that frame and never recovered it. So step 0
        # produced motive-shaped hypotheses and every later step produced scene
        # commentary.
        #
        # Asking for AIM before BELIEF restores the frame and forces the field
        # to exist. This is the same lever as rank-before-score in the scorer:
        # committing to a field before the prose is a cognitive change, not a
        # formatting one.
        # --legacy-form restores the pre-G2 propagation verbatim. It exists so the
        # context fix can be measured on its own: every run after G2 carries the
        # aim-first rewrite AND the 8-word commitment validator, which together
        # cut belief texts from ~74 words to ~33 and dropped millimetre-carrying
        # particles from 18% to 4%. Without a way to turn those off, "did the
        # context fix help" cannot be answered separately from "did the form
        # changes cost".
        legacy = getattr(self.args, 'legacy_form', False)
        system_prompt = (
            f"You are an expert assistant trying to predict {target_agent}'s thoughts."
        ) if legacy else (
            f"You are an expert assistant inferring why {target_agent} acts as they do.\n\n"
            f"Answer in exactly this form, AIM first:\n"
            f"AIM: <one sentence naming what {target_agent} is trying to achieve>\n"
            f"BELIEF: <one or two sentences on what {target_agent} now believes that bears "
            f"on that aim>\n\n"
            f"Rules:\n"
            f"- The AIM is {target_agent}'s own goal. Not a question, not a request aimed at "
            f"another character, not a description of what {target_agent} says or does.\n"
            f"- Do NOT simply restate what another character said or did. If another "
            f"character just spoke, say what it means FOR {target_agent}'S AIM.\n"
            f"- The BELIEF may be about anyone, but it must bear on the aim.")
        propagation_prompts = []
        for hypothesis, anchor in zip(existing_hypotheses.texts, anchors):
            anchor_block = ""
            if anchor:
                anchor_block = (f"<{target_agent}'s standing commitment under this hypothesis>\n{anchor}\n"
                                f"</{target_agent}'s standing commitment under this hypothesis>\n\n"
                                f"Update the belief in light of the new context, but HOLD THIS "
                                f"COMMITMENT FIXED. Your AIM must restate this commitment, not "
                                f"replace it. Do not drift toward a more obvious or more "
                                f"popular reading of the scene.\n\n")
            propagation_prompts.append(
                f"{self.trace_header}\n\n<previous context>\n{context_and_perception_str}\n</previous context>\n"
                f"<previous prediction regarding {target_agent}'s thoughts>\n{hypothesis}\n"
                f"</previous prediction regarding {target_agent}'s thoughts>\n\n{anchor_block}"
                f"<current context>{self.assumption}\n{new_context}\n</current context>\n\n"
                + (f"Question: What did {target_agent} believe?" if legacy else
                   f"Question: What is {target_agent} trying to achieve here, and what does "
                   f"{target_agent} now believe that bears on it?"))
        trace_log.set_stage('propagate')
        raw_texts = self.tracer_model.batch_interact(propagation_prompts, system_prompts=system_prompt, temperature=0, max_tokens=1024)
        propagated_texts = list(raw_texts) if legacy else [compose_aim_belief(r, target_agent) for r in raw_texts]
        # anchors pass through unchanged -> roots inherit, per the anchor==root invariant
        propagated_hypotheses = HypothesesSetV3(target_agent, context_history, perception_history, propagated_texts, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses, anchors=anchors if use_anchor else None)

        return propagated_hypotheses

    def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Rejuvenate hypotheses by paraphrasing the hypotheses
        """
        trace_log.set_stage('perturb')
        for h in existing_hypotheses.texts:
            print(Panel(h, title="Low Variance Hypotheses", style="red", box=box.SIMPLE_HEAD))

        if not self.args.use_perception_only:
            system_prompt = f"Your task is to paraphrase the following text. Make sure to keep the meaning of the text intact while rephrasing them. Do not add any additional comments."
            revision_prompts = [f"{hypothesis}" for hypothesis in existing_hypotheses.texts]
            revised_texts = self.tracer_model.batch_interact(revision_prompts, system_prompts=system_prompt, temperature=1, max_tokens=1024)
            existing_hypotheses.texts = revised_texts
        overall_text_diversity = 1 - overall_jaccard_similarity(existing_hypotheses.texts)
        print(Panel(f"Text diversity: {overall_text_diversity}", title="Diversity of the Jittered Hypotheses", style="blue", box=box.SIMPLE_HEAD))
        print(Panel("\n".join(existing_hypotheses.texts), title="Jittered hypotheses", style="blue", box=box.SIMPLE_HEAD))

        return existing_hypotheses

    def weighted_average_hypotheses(self, hypotheses: HypothesesSetV3, top_p: float = 0.9) -> dict:
        """
        Weighted mean estimate of the hypotheses.
        """
        target_agent = hypotheses.target_agent
        sorted_hypotheses = sorted(zip(hypotheses.texts, hypotheses.weights), key=lambda x: x[1], reverse=True)

        # only select up to cumulative weights of top_p
        top_hypotheses = []
        cumulative_weight = 0
        for hypothesis, weight in sorted_hypotheses:
            top_hypotheses.append((hypothesis, weight))
            cumulative_weight += weight
            if cumulative_weight >= top_p:
                break

        context_history = deepcopy(hypotheses.contexts)
        perception_history = deepcopy(hypotheses.perceptions)
        context_and_perception_str = self.interleave_context_and_perception(context_history, perception_history, target_agent)  
        hypotheses_str = f"{context_and_perception_str}\n\n<{target_agent}'s thoughts>\n"
        for idx, (hypothesis, weight) in enumerate(top_hypotheses):
            hypotheses_str += f"**Prediction {str(idx + 1)} (Weight: {weight:.2f}):**\n{hypothesis}\n\n\n"
        hypotheses_str = hypotheses_str.strip()
        hypotheses_str += f"\n</{target_agent}'s thoughts>\n\nQuestion: What did {target_agent} believe?"

        aggregated_hypothesis = self.tracer_model.interact(hypotheses_str, max_tokens=1234)
        final_hypothesis = f"<{target_agent}'s updated thoughts>\n{aggregated_hypothesis}\n</{target_agent}'s updated thoughts>"

        return {'text': final_hypothesis, 'likelihood': list(hypotheses.weights), 'aggregated': True, 'context': hypotheses.contexts[-1], 'perception': hypotheses.perceptions[-1], 'hypothesis': aggregated_hypothesis}


    def chain_weighted_average_trace(self, hypotheses_list: List[HypothesesSetV3]) -> dict:
        """
        Chain the trace of hypotheses using weighted average

        Args:
            hypotheses (HypothesesSet): 

        Returns:
            str: a summary of the hypothesis trace
        """
        target_agent = self.target_agent
        averaged_hypotheses_list = [self.weighted_average_hypotheses(hypotheses) for hypotheses in hypotheses_list]

        trace_str = ""
        for idx, h in enumerate(averaged_hypotheses_list):
            context_str = ""
            if h['context']['state']:
                context_str += f"{h['context']['state']}\n" # newly added
                context_str += f"<note>{h['perception']['state']}</note>\n"
            if h['context']['action']:
                context_str += f"{h['context']['action']}\n"
                if h['perception']['action']:
                    context_str += f"<note>{h['perception']['action']}</note>"
            trace_str += (f"<context {str(idx + 1)}>\n{context_str.strip()}\n\n"
                          f"<{target_agent}'s updated thoughts>{self._thoughts_block_prefix}{h['hypothesis']}"
                          f"</{target_agent}'s updated thoughts>\n</context {str(idx + 1)}>\n\n")
            
        return {'text': trace_str.strip(), 'aggregated': True}

    def _likelihood_ess(self, weight_results):
        """Gate metric for Phase 1, measured on the PARSED subset only.

        An imputed neutral weight is not a verdict and must not count toward
        separation, so the mask is applied here rather than at the call site.
        Subclasses whose scorer produces no parse mask override this.
        """
        mask = weight_results.get('parse_mask')
        return (trace_log.ess(weight_results['weights'], mask=mask),
                trace_log.ess_norm(weight_results['weights'], mask=mask))

    def _trace(self, text: str, target_agent=None):
        preprocessed_text = self.preprocess_input(text, target_agent)
        if preprocessed_text is None:
            print(cf.bold | cf.magenta("Failed to identify the target agent."))
            self.dump({'summary': ""}, [])
            return ""
        self.set_tracer_variables(preprocessed_text)

        trajectory = preprocessed_text['trajectory']
        perceptions_trajectory = preprocessed_text['perceptions']
        # recorded so the driver can tell a crashed run from a short one
        self._last_trajectory_len = len(trajectory)

        hypotheses_list = []
        context_history = []
        self._accum = {}   # accumulated prior is per-trace
        self._retired = {}
        self._low_mass_run = 0
        self._weak_run = {}
        for idx, (state_action, perceptions) in enumerate(zip(trajectory, perceptions_trajectory)):
            prop_ctx = None
            if idx == 0:
                new_hypotheses = self.initialize(state_action=state_action, perceptions=perceptions)
            else:
                existing_hypotheses = hypotheses_list[-1]
                new_hypotheses = self.propagate(existing_hypotheses, state_action=state_action, perceptions=perceptions)
                prop_ctx = self.interleave_context_and_perception(
                    existing_hypotheses.contexts, existing_hypotheses.perceptions)

            operators = []
            weight_results = None
            likelihood_ess = None
            likelihood_ess_norm = None
            accum_info = None
            perturb_info = None
            perturb_conditions = None
            cap_info = None
            expiry_info = None
            split_info = None
            pre_snapshot = []
            ess = None
            if state_action['action']:
                weight_results = self.weigh(new_hypotheses, state_action['action'], mode="prompting")
                # likelihood ESS is measured on the normalized likelihood vector
                # ALONE, before any accumulation/flooring/resampling. It is the
                # Phase 1 gate metric and must never be conflated with posterior ESS.
                likelihood_ess, likelihood_ess_norm = self._likelihood_ess(weight_results)
                # PHASE 2: combine with the carried prior instead of overwriting it.
                _L = [float(x) for x in weight_results['weights']]
                for _r, _i in enumerate(sorted(range(len(_L)), key=lambda j: -_L[j]), start=1):
                    if _i < len(new_hypotheses.hypotheses):
                        new_hypotheses.hypotheses[_i]._likelihood_rank = _r
                accum_info = self.accumulate(new_hypotheses, weight_results['weights'])
                new_hypotheses.weight_details = weight_results

                if self.args.n_hypotheses > 1:
                    ess = compute_ess(new_hypotheses)
                    # Snapshot before any operator rebuilds the population, so
                    # post-resample parent_ids resolve and roots reconstruct.
                    pre_snapshot = [
                        {'particle_id': h.particle_id, 'lineage_id': h.lineage_id,
                         'root_id': h.root_id, 'weight': float(w)}
                        for h, w in zip(new_hypotheses.hypotheses, new_hypotheses.weights)
                    ]
                    # Threshold against the ACTUAL population, not the configured N.
                    pop = len(new_hypotheses.hypotheses)
                    # Threshold placed from the MEASURED density, not convention.
                    # Pre-operator normalized ESS lives at 0.55-0.75 across 140
                    # steps; N/2 = 0.50 sits on that shoulder, so crossings were
                    # near-ties (a run fired at 0.49 vs 0.50) and resample count
                    # varied 1-2 on the same context. N/3 = 0.333 sits in an
                    # empty region: 0 steps within +/-0.05, while still catching
                    # both genuine deep dips (0.15, 0.40). N/4 captures the same
                    # two, so N/3 is the smaller change for the same effect.
                    ess_divisor = float(getattr(self.args, 'ess_divisor', 3.0))
                    if ess < pop / ess_divisor:
                        new_hypotheses = resample_hypotheses_with_other_info(new_hypotheses, ess)
                        operators.append('resample')
                        self.sync_accumulator(new_hypotheses)
                    # PHASE 2: `if`, not `elif`. Resampling duplicates particles and
                    # REDUCES diversity, so it used to short-circuit the one operator
                    # that could repair it. Confirmed on gold: step 2 had diversity
                    # 0.219 (below the 0.25 trigger) AND ESS below threshold, so the
                    # step that most needed jitter got 6 duplicates instead.
                    #
                    # Diversity is measured AFTER resampling now, so the post-resample
                    # collapse is visible rather than hidden behind the pre-resample read.
                    # PHASE 4: split (source) then merge (sink) in the SAME step.
                    if getattr(self.args, 'enable_split', False):
                        new_hypotheses, _sm = self.split_and_merge(new_hypotheses, prop_ctx)
                        split_info = _sm
                        if _sm.get('split_net'):
                            operators.append('split')
                        if _sm.get('expanded'):
                            operators.append('expand')
                        if _sm.get('merged'):
                            operators.append('merge')
                        if _sm.get('split_fired') or _sm.get('merged'):
                            new_hypotheses, _cap, _bound = self.enforce_population_cap(new_hypotheses)
                            if _bound:
                                operators.append('cap')
                            cap_info = (_cap, _bound)
                            self.sync_accumulator(new_hypotheses)
                    overall_text_diversity = 1 - overall_jaccard_similarity(new_hypotheses.texts)
                    if getattr(self.args, 'anchored_perturbation', False):
                        # PHASE 3e trigger: root-mass ESS, not Jaccard.
                        # Jaccard cannot see the failure -- measured particle ESS
                        # 0.96 (healthy) against root-mass 0.16 (~1.3 effective
                        # hypotheses of 8), with duplicates lexically identical
                        # only until propagation rewrites them.
                        rm = trace_log.root_mass_ess_over_n(new_hypotheses.hypotheses)
                        thr = float(getattr(self.args, 'root_mass_threshold', 0.5))
                        by_root = {}
                        for j, h in enumerate(new_hypotheses.hypotheses):
                            by_root.setdefault(h.root_id, []).append(j)
                        # TWO conditions, recorded separately. Mass alone is not
                        # collapse: root-mass ESS also falls when many distinct
                        # roots hold concentrated weight, which is the filter
                        # converging correctly and must not be perturbed.
                        mass_cond = rm is not None and rm < thr
                        collapse_cond = any(len(v) > 1 for v in by_root.values())
                        # Sustained-stagnation counter: how many consecutive
                        # steps root-mass has sat below threshold.
                        self._low_mass_run = (self._low_mass_run + 1) if mass_cond else 0
                        k = int(getattr(self.args, 'stagnation_steps', 3))
                        # TWO PATHS, both minting new roots:
                        #   collapse   -- duplicates exist; break the surplus copies apart
                        #   stagnation -- no duplicates, but root-mass has been low for k
                        #                 consecutive steps. This DECOUPLES repair from the
                        #                 resampler: with the production trigger at N/4 the
                        #                 resampler is a backstop and rarely fires, so tying
                        #                 perturbation to collapse would leave the repair
                        #                 machinery dormant. A transient dip is the filter
                        #                 converging and must not be touched; a SUSTAINED one
                        #                 is the hypothesis space going dead.
                        stagnation_cond = (mass_cond and not collapse_cond
                                           and self._low_mass_run >= k)
                        # SURPRISE. The other two paths both measure the SHAPE of
                        # the population -- how many roots, how the mass is spread.
                        # Neither can see that every commitment in a perfectly
                        # healthy-looking population is wrong, which is exactly
                        # what happens at a narrative reversal. This path fires on
                        # FIT: the null outranked every live commitment.
                        surprise_cond = bool(
                            getattr(self.args, 'surprise_perturb', False)
                            and weight_results is not None
                            and weight_results.get('surprise'))
                        # RESAMPLE-MOVE (Gilks & Berzuini). A classical particle
                        # filter gets divergence free: propagation adds noise, so
                        # duplicates separate immediately. Here propagation runs at
                        # temperature 0, so resampled copies stay BYTE-IDENTICAL --
                        # measured persisting 9 turns, three particles spending
                        # three LLM calls per turn computing the same thing.
                        #
                        # So the move must be applied explicitly, and it cannot be
                        # gated on a separate diversity threshold: 8 of 17 resamples
                        # fired while the root-mass condition was unmet and left
                        # duplicates in place. Whenever resampling creates copies,
                        # they get perturbed.
                        perturb_conditions = (mass_cond, collapse_cond,
                                              sum(len(v) - 1 for v in by_root.values() if len(v) > 1),
                                              self._low_mass_run,
                                              'surprise' if surprise_cond
                                              else ('resample_move' if (collapse_cond and 'resample' in operators and not mass_cond)
                                              else ('collapse' if (mass_cond and collapse_cond)
                                              else ('stagnation' if stagnation_cond else None))))
                        just_resampled = 'resample' in operators
                        # collapse repair fires whenever duplicates exist AND either
                        # the diversity threshold tripped or we just resampled.
                        if (collapse_cond and (mass_cond or just_resampled)) or stagnation_cond or surprise_cond:
                            # perturb only the OFFENDING particles: surplus copies
                            # of over-represented roots, weakest first.
                            idxs = []
                            if surprise_cond:
                                # Replace the lightest quarter. The leader is kept:
                                # it may still be right about the standing goal even
                                # when it fails to explain this one action, and the
                                # point is to ADD a reading, not to destroy one.
                                nrep = max(1, len(new_hypotheses.hypotheses) // 4)
                                idxs = sorted(range(len(new_hypotheses.hypotheses)),
                                              key=lambda j: float(new_hypotheses.weights[j]))[:nrep]
                            elif collapse_cond:
                                # PROTECT THE LEADER. Perturbation replaces only the
                                # SURPLUS copies of an over-represented root; the
                                # heaviest copy is always kept, so a hypothesis that
                                # is winning on evidence is never eliminated.
                                #
                                # Observed without this: a hypothesis holding 35% of
                                # belief mass at turn 3 was gone by turn 5 -- resampling
                                # concentrated copies onto it, and perturbation then
                                # treated that concentration as redundancy to break up.
                                # That works directly against the accumulator, whose
                                # purpose is to let evidence compound across steps.
                                protect = int(getattr(self.args, 'protect_leader', 1))
                                for _, members in sorted(by_root.items(), key=lambda kv: -len(kv[1])):
                                    if len(members) > protect:
                                        ranked = sorted(members, key=lambda j: -float(new_hypotheses.weights[j]))
                                        idxs.extend(ranked[protect:])
                            else:
                                # stagnation: replace the LIGHTEST particles. They hold
                                # effectively no mass, so nothing is lost, and a novel
                                # commitment can be introduced without the resampler
                                # having to destroy an old one first.
                                nrep = max(1, len(new_hypotheses.hypotheses) // 4)
                                idxs = sorted(range(len(new_hypotheses.hypotheses)),
                                              key=lambda j: float(new_hypotheses.weights[j]))[:nrep]
                                self._low_mass_run = 0
                            if idxs:
                                if surprise_cond:
                                    print(Panel(f"null outranked every commitment: minting {len(idxs)}",
                                                title="Surprise", style="yellow", box=box.SIMPLE_HEAD))
                                else:
                                    print(Panel(f"root-mass ESS {rm:.3f} < {thr}: perturbing {len(idxs)} particle(s)",
                                                title="Diversity Collapse", style="red", box=box.SIMPLE_HEAD))
                                # prop_ctx is built from the PREVIOUS contexts and
                                # excludes the action just scored -- so on the
                                # surprise path the mint would be blind to the very
                                # thing that triggered it. Pass it explicitly.
                                res = self.perturb_anchored(
                                    new_hypotheses, idxs, prop_ctx if idx > 0 else None,
                                    surprising_action=(state_action.get('action') if surprise_cond else None))
                                operators.append('perturb')
                                perturb_info = res
                                new_hypotheses, _cap, _bound = self.enforce_population_cap(new_hypotheses)
                                if _bound:
                                    operators.append('cap')
                                cap_info = (_cap, _bound)
                                self.sync_accumulator(new_hypotheses)
                    # WEIGHT-BASED EXPIRY -- independent of resampling and of
                    # the collapse trigger above. Runs last so a particle already
                    # replaced this step is not retired twice; its lineage is new
                    # and its counter starts at zero.
                    if getattr(self.args, 'enable_expiry', False):
                        dead, exp_info = self.expire_weak(new_hypotheses,
                                                  just_resampled='resample' in operators)
                        expiry_info = exp_info
                        if dead:
                            print(Panel(
                                f"{len(dead)} hypothesis(es) below {exp_info['expiry_threshold']:.4f} "
                                f"for {exp_info['expiry_steps']} consecutive turns: retiring",
                                title="Expiry", style="yellow", box=box.SIMPLE_HEAD))
                            res = self.perturb_anchored(new_hypotheses, dead,
                                                        prop_ctx if idx > 0 else None)
                            operators.append('expire')
                            if perturb_info is None:
                                perturb_info = res
                            new_hypotheses, _cap, _bound = self.enforce_population_cap(new_hypotheses)
                            if _bound:
                                operators.append('cap')
                            cap_info = (_cap, _bound)
                            self.sync_accumulator(new_hypotheses)
                    elif overall_text_diversity < 0.25:
                        print(Panel(f"Text diversity: {overall_text_diversity}", title="Low Variance Hypotheses", style="red"))
                        new_hypotheses = self.rejuvenate_hypotheses(new_hypotheses)
                        operators.append('perturb')
                        self.sync_accumulator(new_hypotheses)
            else:
                pass

            self._log_step(idx, new_hypotheses, weight_results, operators,
                           likelihood_ess=likelihood_ess, ess_value=ess,
                           likelihood_ess_norm=likelihood_ess_norm,
                           accum_info=accum_info, state_action=state_action,
                           pre_snapshot=pre_snapshot, perturb_info=perturb_info,
                           perturb_conditions=perturb_conditions, cap_info=cap_info,
                           split_info=split_info, expiry_info=expiry_info)
            hypotheses_list.append(new_hypotheses)

            # update history
            if state_action['state']:
                context_history.append({'text': state_action['state'], 'action': False})
            if state_action['action']:
                context_history.append({'text': state_action['action'], 'action': True})

        traced_thoughts = self.chain_weighted_average_trace(hypotheses_list)
        trace_text = f"{self.trace_header}\n\n{traced_thoughts['text']}"

        self.dump(traced_thoughts, hypotheses_list)
        return trace_text

    def trace(self, input_text, target_agent=None):
        # check if the input text, target_character is cached 
        if target_agent is None:
            target_agent = self.identify_target(input_text)
        if self.args.dataset != "mmtom":
            context = input_text.split("\nQuestion:")[0].strip()
        else:
            context = input_text
        if self.args.dataset == "fantom":
            context = context.split("\n\nTarget:")[0].strip()
            context = context.split("\n\nInformation:")[0].strip()
        elif self.args.dataset == "confaide":
            context = context.split("Meeting:")[-1].strip()

        # check if the input text is already traced in self.cache_db
        if context in self.cache_db and target_agent in self.cache_db[context]:
            print(Panel(f">>> Using cached trace for {target_agent}!", style="yellow"))
            return self.cache_db[context][target_agent]
        
        print(Panel(f">>> Tracing {target_agent}'s thoughts!", style="blue"))
        trace_result = self._trace(context, target_agent)

        # cache the trace result for the same input text and character
        if context not in self.cache_db:
            self.cache_db[context] = {}
            self.cache_db[context][target_agent] = trace_result
        else:
            self.cache_db[context][target_agent] = trace_result

        return trace_result

class MultiTracer(Tracer):
    def identify_target(self, input_text: str) -> str:
        """
        Identify the target agent that we have to trace by looking at the question.

        Args:
            input_text (str): The context text.

        Returns:
            str: The target agent.
        """
        question = extract_question(input_text)
        target_identification_prompt = f"'{question}'\n\nMain question: Who is the subject of the above question? Whose perspective is this question primarily about? Provide the names of the individual, their title, or the group. If the subject of the question is not related to a person or a group, state 'none'. If the question is asking to list the names, state 'all'.\nThe concise answer to the main question is (e.g, name):"

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o-2024-11-20', run_id=self.args.run_id)
            output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
        else:
            output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
        target_agent = output.split("\n")[0].split(":")[-1].strip().strip(".").replace("*", "")

        if target_agent == 'all':
            print(Panel(f"Identified target agent as 'all'!", style="yellow"))
            context = input_text.split(question)[0].split("\n\n")[0]
            target_identification_prompt = f"{context}\n\nName all the people or groups who are in the context. Use commas. Do not include any additional comments."
            if self.args.use_helper_llm:
                llm = load_model('gpt-4o-2024-11-20', run_id=self.args.run_id)
                output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
            else:
                output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
            target_agent = output

        return target_agent

    def trace(self, input_text, target_agent=None):
        # check if the input text, target_character is cached 
        if target_agent is None:
            target_agent = self.identify_target(input_text)
        context = input_text.split("\nQuestion:")[0].strip()
        if self.args.dataset == 'fantom':
            context = context.split("\n\nTarget:")[0].strip()
            context = context.split("\n\nInformation:")[0].strip()

        if "," in target_agent:
            _target_agents = target_agent.split(",")
            target_agents = [a.strip() for a in _target_agents]
            trace_result = ""
            for idx, ta in enumerate(target_agents):
                if context in self.cache_db and ta in self.cache_db[context]:
                    print(Panel(f">>> Using cached trace for {ta}!", style="yellow"))
                    individual_trace = self.cache_db[context][ta]
                else:
                    print(Panel(f">>> Tracing {ta}'s thoughts!", style="blue"))
                    individual_trace = self._trace(context, ta)
                if idx == 0:
                    _individual_trace = individual_trace.replace("To answer this question,", "To answer this question, first,").strip()
                elif idx > 0 and idx < len(target_agents) - 1:
                    _individual_trace = individual_trace.replace("To answer this question", "Next").strip()
                else:
                    _individual_trace = individual_trace.replace("To answer this question", "Finally").strip()
                trace_result += _individual_trace + "\n\n"

                # cache the trace result for the same input text and character
                if context not in self.cache_db:
                    self.cache_db[context] = {}
                    self.cache_db[context][ta] = individual_trace
                else:
                    self.cache_db[context][ta] = individual_trace
        else:
            # check if the input text is already traced in self.cache_db
            if context in self.cache_db and target_agent in self.cache_db[context]:
                print(Panel(f">>> Using cached trace for {target_agent}!", style="yellow"))
                return self.cache_db[context][target_agent]
            
            print(Panel(f">>> Tracing {target_agent}'s thoughts!", style="blue"))
            trace_result = self._trace(context, target_agent)

            # cache the trace result for the same input text and character
            if context not in self.cache_db:
                self.cache_db[context] = {}
                self.cache_db[context][target_agent] = trace_result
            else:
                self.cache_db[context][target_agent] = trace_result

        return trace_result.strip()

class TracerLight(Tracer):
    """
    TracerLight is a simplified version of Tracer that does propagation and likelihood calculation of all hypotheses at once in a single prompt.
    This leads to much worse performance.
    """
    def initialize(self, state_action, perceptions):
        trace_log.set_stage('initialize')
        context_input = ""
        state, action = state_action['state'], state_action['action']
        if state:
            if self.args.input_is_chat:
                context_input += f"{state}\n"
            else:
                agent_state = self.get_agent_state(self.target_agent, state)
                context_input += f"<state>\n{agent_state.strip()}\n</state>\n"
            context_input += f"<note>{perceptions['state']}</note>\n\n"

        n_hypotheses_str = str(self.args.n_hypotheses)

        if self.args.n_hypotheses > 1:
            # PHASE 3a: seed at the GOAL level, not the belief level.
            #
            # Two particles can differ in believed facts and still predict the
            # same next utterance, which is a likelihood the evaluator cannot
            # separate -- measured at 88% bucket 'a' with diversity 0.743, i.e.
            # textually various, predictively identical. Goal-level differences
            # are what produce divergent action predictions downstream.
            #
            # PREDICTED NULL: this raises the INTERCEPT (more distinct starting
            # roots) and cannot touch the SLOPE (~0.375-0.50 decay per resample).
            # A higher ending root count is the trivial consequence of a higher
            # start and must not be read as success.
            goal_seeded = getattr(self.args, 'goal_seeding', False)
            if goal_seeded:
                axis = (f"Each hypothesis must propose a DIFFERENT thing {self.target_agent} "
                        f"wants or is trying to achieve -- not merely a different belief about "
                        f"the facts. Two hypotheses that share a goal but differ in wording are "
                        f"the same hypothesis. Make the goals mutually exclusive where possible.")
            else:
                axis = ""
            if action:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what were {self.target_agent}'s thoughts (e.g., beliefs, intent) that led to the action above. {axis}Do not add any additional comments."
            else:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what {self.target_agent} will be thinking (e.g., beliefs). {axis}Do not add any additional comments."
            _hypotheses_list = prompting_for_ordered_list(self.tracer_model, prompt=belief_query, n=self.args.n_hypotheses)
            hypotheses_list = [hypothesis.strip() for hypothesis in _hypotheses_list]
        else:
            belief_query = f"{context_input}\n\nQuestion: What will {self.target_agent} be thinking now?"
            hypothesis = self.tracer_model.interact(belief_query, temperature=0, max_tokens=1024)
            hypotheses_list = [hypothesis]

        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)
        initial_hypotheses = HypothesesSetV3(target_agent=self.target_agent, contexts=[state_action], perceptions=[perceptions], texts=hypotheses_list, weights=weights)

        return initial_hypotheses

    def propagate(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions: dict) -> HypothesesSetV3:
        """
        Propagate the hypotheses on the target agent using the context, which is the text that does not contain the target agent's actions.
        """

        prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions) # get interleaved context and perception
        target_agent = prop_info["target_agent"]
        context_history = prop_info["context_history"]
        perception_history = prop_info["perception_history"]
        context_and_perception_str = prop_info["context_and_perception_str"]
        new_context = prop_info["new_context"]

        system_prompt = f"You are an expert assistant trying to predict {target_agent}'s thoughts. Update the previous {self.args.n_hypotheses} predictions based on the new context and the new action. Try to make the predictions diverse to cover a wide range of possibilities even low probability ones. Output {self.args.n_hypotheses} updated predictions in an ordered list. Do not add any additional comments."
        ordered_hypotheses_string = '\n'.join([f"{index + 1}. {item}" for index, item in enumerate(existing_hypotheses.texts)])

        propagation_prompt = f"{self.trace_header}\n\n<previous context>\n{context_and_perception_str}\n</previous context>\n<previous predictions regarding {target_agent}'s thoughts>\n{ordered_hypotheses_string}\n</previous predictions regarding {target_agent}'s thoughts>\n\n<current context>{self.assumption}\n{new_context}\n</current context>\n\nQuestion: What did {target_agent} believe?"
        temperature = 0
        while True:
            propagated_text = self.tracer_model.interact(propagation_prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=2024)
            propagated_hypotheses_str_list = capture_and_parse_ordered_list(propagated_text)
            if len(propagated_hypotheses_str_list) == self.args.n_hypotheses:
                break
            else:
                print(Panel(f"Number of propagated hypotheses is not equal to {self.args.n_hypotheses}! Retrying...", box=box.SIMPLE_HEAD, style="red"))
                temperature += 0.3
                if temperature > 1.2:
                    break

        propagated_hypotheses = HypothesesSetV3(target_agent, context_history, perception_history, propagated_hypotheses_str_list, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses)

        return propagated_hypotheses

    def prompt_likelihood(self, existing_hypotheses: list, context_history: list, perception_history: list, action: str, target_agent: str = None):

        trace_log.set_stage('likelihood')
        target_agent = self.target_agent if target_agent is None else target_agent
        # TODO: maybe trim the last action and its perception because the thought actually comes before the action.
        pruned_context_history = deepcopy(context_history)
        pruned_perception_history = deepcopy(perception_history)
        if len(pruned_context_history) > 1:
            context_and_perception_str = self.interleave_context_and_perception(pruned_context_history[:-1], pruned_perception_history[:-1]) # to remove the last action and its perception, because we will be evaluating the thought before the action
        else:
            if pruned_context_history[-1]['state']:
                context_and_perception_str = f"{pruned_context_history[-1]['state']}"
            else:
                context_and_perception_str = ""

        system_prompt = f"Your job is to rate the probability (0-1) of actions/utterance under a list of given different hypotheses. Use common sense: for instance, if someone is searching for an item, they are likely to take it once they find it rather than merely observing it. If they don't take it and just sees it, it indicates a lack of interest and that was not what they were looking for. For each hypothesis, briefly explain the probability of the action/utterance under each hypothesis first. Then, at the end of your response, aggregate the answer for each hypothesis in a simple ordered list with prefix 'Final Answer:'"
        question = f"Question: Rate the probability (0-1) of the <next action> or <next response> described above under each given hypothesis. Let's think step by step and give the final answer."
        hypothesis_str = '\n'.join([f"Hypothesis {index + 1}. {item}" for index, item in enumerate(existing_hypotheses)])
        if self.args.input_is_chat:
            likelihood_prompt = f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis_str}\n</{target_agent}'s thoughts>\n\n<next response>\n{action}\n</next response>\n\n{question}"
        else:
            likelihood_prompt = f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis_str}\n</{target_agent}'s thoughts>\n\n<next action>{action}</next action>\n<note>{perception_history[-1]['action']}</note>\n\n{question}"
        raw_predictions = self.tracer_model.interact(likelihood_prompt, temperature=0, system_prompt=system_prompt, max_tokens=2048)
        if "Final Answer:" in raw_predictions:
            reasoning, answer = raw_predictions.split("Final Answer:")[0], raw_predictions.split("Final Answer:")[-1]
        elif "Final Answer**" in raw_predictions:
            reasoning, answer = raw_predictions.split("Final Answer**")[0], raw_predictions.split("Final Answer**")[-1]
        else:
            reasoning = answer = raw_predictions
        probs = capture_and_parse_ordered_list(answer)
        converted_probs = [float(prob.split(":")[-1].strip().strip("*").strip()) for prob in probs]

        # normalize the probabilities to sum to 1
        converted_probs = converted_probs / np.sum(converted_probs)
        weights = converted_probs

        results = {
            'prompts': [likelihood_prompt] * len(existing_hypotheses),
            'raw_predictions': raw_predictions,
            'reasonings': reasoning,
            'raw_scores': probs,
            'weights': weights
        }

        return results

    def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Rejuvenate hypotheses by paraphrasing the hypotheses
        """
        trace_log.set_stage('perturb')
        for h in existing_hypotheses.texts:
            print(Panel(h, title="Low Variance Hypotheses", style="red", box=box.SIMPLE_HEAD))

        system_prompt = f"Your task is to paraphrase the following list of texts. Make sure to keep the meaning of the texts intact while rephrasing them. If there are identical texts, try to make them slightly different. Do not add any additional comments and output the revised texts in an ordered list."
        hypotheses_list = [f"{index + 1}. {item}" for index, item in enumerate(existing_hypotheses.texts)]
        hypotheses_list_str = "\n".join(hypotheses_list)
        revised_text = self.tracer_model.interact(hypotheses_list_str, system_prompt=system_prompt, temperature=1, max_tokens=1024)
        revised_hypotheses_list = capture_and_parse_ordered_list(revised_text)
        existing_hypotheses.texts = revised_hypotheses_list
        overall_text_diversity = 1 - overall_jaccard_similarity(existing_hypotheses.texts)
        print(Panel(f"Text diversity: {overall_text_diversity}", title="Diversity of the Jittered Hypotheses", style="blue", box=box.SIMPLE_HEAD))
        print(Panel("\n".join(existing_hypotheses.texts), title="Jittered hypotheses", style="blue", box=box.SIMPLE_HEAD))

        return existing_hypotheses

    def weighted_average_hypotheses(self, hypotheses: HypothesesSetV3, top_p: float = 0.9) -> dict:
        """
        Weighted mean estimate of the hypotheses.
        """
        
        target_agent = hypotheses.target_agent
        sorted_hypotheses = sorted(zip(hypotheses.texts, hypotheses.weights), key=lambda x: x[1], reverse=True)

        # only select up to cumulative weights of top_p
        top_hypotheses = []
        cumulative_weight = 0
        for hypothesis, weight in sorted_hypotheses:
            top_hypotheses.append((hypothesis, weight))
            cumulative_weight += weight
            if cumulative_weight >= top_p:
                break

        hypotheses_str = ""
        for idx, (hypothesis, weight) in enumerate(top_hypotheses):
            hypotheses_str += f"**Prediction {str(idx + 1)} (Weight: {weight:.2f}):**\n{hypothesis}\n\n"
        hypotheses_str = hypotheses_str.strip()
        final_hypothesis = f"<{target_agent}'s updated thoughts>\n{hypotheses_str}\n</{target_agent}'s updated thoughts>"

        return {'text': final_hypothesis, 'likelihood': list(hypotheses.weights), 'aggregated': True, 'context': hypotheses.contexts[-1], 'perception': hypotheses.perceptions[-1], 'hypothesis': hypotheses_str}

    _thoughts_block_prefix = "\n"

    def _likelihood_ess(self, weight_results):
        """TracerLight's scorer emits no parse mask, so the gate is measured on
        the full likelihood vector and the normalized variant is not reported."""
        return trace_log.ess(weight_results['weights']), None


class MultiTracerLight(TracerLight, MultiTracer):
    pass

def load_tracer_model(args):
    if args.tracer_type == 'tracer':
        tracing_model = Tracer(args)
    elif args.tracer_type == 'multi-tracer':
        tracing_model = MultiTracer(args)
    elif args.tracer_type == 'tracer-light':
        tracing_model = TracerLight(args)
    elif args.tracer_type == 'multi-tracer-light':
        tracing_model = MultiTracerLight(args)
    else:
        raise NotImplementedError

    return tracing_model