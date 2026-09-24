"""P(alternative | portfolio) for the choice-point filter, behind one interface.

THE PLANNED PRIMARY IS UNAVAILABLE. The design scored a single option letter and
read top-k log-probs at that position, giving an exact normalised distribution
for one call. PREREG Phase B records that every Google route refuses it -- two
credential formats, two SDKs, the OpenAI-compatibility endpoint, ~15 models --
so `LogprobScorer` is written against the interface and cannot be confirmed here.

THIS FILE IMPLEMENTS THE PRE-REGISTERED FALLBACK: a coarse DECLARED RANKING.

    The model is asked to rank the alternatives, not to invent numbers.
    A fixed monotone map turns a rank into a probability, and the map's one
    free parameter is FITTED ON DEV.

Why a ranking rather than asking for percentages directly: the design rejects
"invent a number" for the same reason v1's scorer failed (P-14) -- an LLM asked
for calibrated numerics produces confident noise. A rank is a judgement the model
can actually make; the calibration then lives in the rank->probability map, where
it is fitted against observed frequencies and can be audited. The model supplies
the ORDER, the dev set supplies the SPREAD.

Why sampled frequency is NOT the fallback, restated because it keeps looking
tempting: at eps_frac=0.12 and n=6 the epsilon floor is ~0.02, so a resolution of
1/K needs K > n/eps_frac ~ 50 samples per particle per choice point, or sampling
noise rather than the floor binds the weight update.

FORMAT COMPLIANCE IS PART OF `aligned`, NOT AN ASSUMPTION INSIDE IT. Probing the
backend found gemini-2.5-flash answering "The question asks Alex" to a
one-letter instruction at max_output_tokens=4, because thinking tokens consumed
the budget. Thinking is disabled explicitly and the parse rate is reported.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

# Rank -> probability: p(r) proportional to THETA^(r-1), renormalised over the
# alternatives offered. One parameter, fitted on dev, reported with the fit.
# THETA=1 is uniform (the model's order carries nothing); THETA->0 is argmax
# (the order is treated as certainty). The dev fit is what stops either being
# assumed.
DEFAULT_THETA = 0.45

LETTERS = "ABCDEFGHIJKLM"          # A-M; Diplomacy offers up to 13 (PREREG)


def options_block(alternatives: Sequence[str], order: Sequence[int] = None) -> str:
    """Render the offered alternatives as lettered lines.

    `order` is a permutation index; the same choice point is rendered under two
    different orders and the induced distribution over ACTIONS must agree, or a
    positional prior on 'A' reads as a motive effect.
    """
    idx = list(order) if order is not None else list(range(len(alternatives)))
    return "\n".join(f"  {LETTERS[i]}  {alternatives[j]}" for i, j in enumerate(idx))


def forecast_prompts(prefix: str, person: str, situation: str,
                     alternatives: Sequence[str], portfolio_text: str = None,
                     order: Sequence[int] = None):
    """The (system, user) pair. MODULE LEVEL AND PURE, like rank_scorer_prompts.

    audit_vacuity.py re-scores a LOGGED slate by parsing the prompt and
    rebuilding the rest verbatim; audit_forecast.py does the same with a
    substituted portfolio. A rebuild that reconstructed this wording by hand
    would be measuring a prompt the filter never used, so the block markers
    below are a contract with the audit as much as with the model.
    """
    n = len(alternatives)
    system = (
        f"You forecast what {person} does next.\n\n"
        f"Rank ALL {n} options from MOST to LEAST likely. Commit to a strict "
        f"order -- no ties, every letter used exactly once.\n\n"
        f"Rules:\n"
        f"- Judge only what {person} is likely to do here. Do not reward an "
        f"option for being reasonable, fair, or well phrased.\n"
        f"- Use only what is shown. Do not assume anything said later.\n\n"
        f"Answer with exactly one line:\n"
        f"RANKING: <letter>, <letter>, ... ({n} letters, best first)")
    belief = f"\n<what we believe about {person}>\n{portfolio_text}\n</what we believe about {person}>\n" \
        if portfolio_text else ""
    user = (f"<conversation so far>\n{prefix or '(nothing yet)'}\n</conversation so far>\n"
            f"{belief}\n<situation>\n{situation}\n</situation>\n\n"
            f"<options>\n{options_block(alternatives, order)}\n</options>")
    return system, user


RANKING = re.compile(r"RANKING\s*:\s*([A-M][\sA-M,;>\-]*)", re.I)


def parse_ranking(raw: str, n: int):
    """The permutation the reply commits to, as indices into the RENDERED order.

    Returns None rather than a partial order when the reply does not name every
    letter exactly once. A partial ranking silently completed by the parser is
    the parser's forecast, not the model's, and it would be scored as if it were
    evidence -- counted as an unparsed call instead.
    """
    if not raw:
        return None
    m = RANKING.search(raw)
    body = m.group(1) if m else raw
    seen, out = set(), []
    for ch in body.upper():
        if ch in LETTERS[:n] and ch not in seen:
            seen.add(ch)
            out.append(LETTERS.index(ch))
    return out if len(out) == n else None


def rank_to_distribution(rank_positions: Sequence[int], n: int,
                         theta: float = DEFAULT_THETA) -> Dict[int, float]:
    """Geometric decay over declared rank, renormalised. Keys are rendered slots."""
    w = {}
    for r, slot in enumerate(rank_positions):
        w[slot] = theta ** r
    z = sum(w.values()) or 1.0
    return {k: v / z for k, v in w.items()}


def to_actions(dist_by_slot: Dict[int, float], alternatives: Sequence[str],
               order: Sequence[int] = None) -> Dict[str, float]:
    """Map rendered slots back to action labels, undoing the permutation."""
    idx = list(order) if order is not None else list(range(len(alternatives)))
    return {alternatives[idx[slot]]: p for slot, p in dist_by_slot.items()}


def fit_theta(records, grid=None) -> float:
    """Choose THETA on DEV by log-score. Reported with the value it resolved to.

    `records` are (rank_positions, actual_slot, n). Fitting on dev is what makes
    the map a calibration rather than a guess -- and it is the only free
    parameter, so the fit cannot quietly absorb a bad ordering.
    """
    grid = grid or [round(x, 2) for x in [0.05 * i for i in range(1, 20)]]
    best, best_ll = None, -1e18
    for th in grid:
        ll = 0.0
        for ranks, actual, n in records:
            d = rank_to_distribution(ranks, n, th)
            ll += math.log(max(d.get(actual, 0.0), 1e-12))
        if ll > best_ll:
            best, best_ll = th, ll
    return best
