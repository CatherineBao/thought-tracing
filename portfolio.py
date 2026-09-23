"""The portfolio particle: a small set of co-existing motives in priority order.

WHY A PORTFOLIO AND NOT A SINGLE COMMITMENT. v1 gave each particle one anchor and
made co-existing motives compete for one pool of weight. People hold several
motives at once, so different turns favoured different TRUE motives and the
leader kept flipping -- argmax churn 55.6 per 100 steps, zero runs at zero
(PRECHECKS P-7). A portfolio moves the competition up a level: particles are
rival ACCOUNTS of the person, and the motives inside one account co-exist rather
than fight. The priority order is itself a hypothesis -- "timeline beats accuracy
when they conflict" is the kind of claim a single-anchor particle cannot state.

TWO IDENTITY PLANES, DELIBERATELY DISTINCT. hypothesis.py already documents the
particle_id / lineage_id split; this module adds a second plane below it.

    particle plane   particle_id / lineage_id / root_id
                     identity of THIS PORTFOLIO COMPOSITION -- the hypothesis.
                     Unchanged semantics, so expire_weak, ancestral_mass,
                     root_mass_ess_over_n, argmax_churn and the
                     baseline_metrics.json comparisons keep working untouched.

    motive plane     motive_id / motive_root_id
                     identity of ONE COMMITMENT. This is where
                     hypothesis.update_anchor's ANCHOR IDENTITY == ROOT IDENTITY
                     rule now lives: revising a motive's anchor founds a new
                     motive root. The rule is not weakened, only relocated one
                     level down.

Overloading root_id onto the motive set was considered and rejected: a set-hash
root stops being unique in time, so a portfolio that dies and is independently
re-founded reuses its id and ancestral_mass resurrects a dead series instead of
starting a new one.

PRIORITY ORDER IS NOT PART OF ROOT IDENTITY. A reorder is a claim about how
conflicts resolve, not about content. Making it root-founding would mint a root
on an operator v2 exists to ENCOURAGE, reinstating exactly the birth/death churn
the portfolio is meant to remove -- and merge_similar's same-root guard would
then refuse to compare two order-variants, which are the pairs most likely to be
genuinely redundant. Reorders are counted in priority_revisions instead, mirroring
anchor_revisions.

MARGINAL MASS IS THE STABILITY METRIC, AND IT IS A SNAPSHOT. A motive's marginal
is the total weight of every particle holding it, keyed on motive_root_id. It is
churn-invariant: it does not move when two particles holding one motive trade
places, which is the churn-invariant measure the original briefing asked for.

It is NOT an accumulator. accumulate_weights keys on lineage_id, so a motive
migrating between particles carries no accumulated evidence with it. "Marginal
mass rose over the run" is therefore not readable across a resample the way
ancestral_mass is, and no caller may treat it that way.

THE PINNED PARTICLE IS EXCLUDED FROM THE PRIMARY MARGINAL. Every population
carries one particle whose portfolio is the person's own stated reasons, and it
can lose weight but never expires. Because it never dies its motives always carry
at least its weight, which makes a total that includes it non-comparable across
population sizes. marginals() therefore drops it by default and reports it
separately.

This module imports trace_log (for id minting) and nothing else. It must never
import tracer: the audits and the eval harness have to be able to read a
portfolio without constructing an API client.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from trace_log import new_particle_id


# A portfolio holds at most three motives. Above three the forecast prompt stops
# being a claim about a person and becomes a list, and the population needs
# 4-8 distinct ACCOUNTS rather than one account carrying everything.
MAX_MOTIVES = 3

# Population floor. split_candidates returns [] below n=4 (trace_log) and
# merge_similar returns early below n=3 (tracer), so with one pinned slot a
# 4-particle arm leaves 3 free and essentially no operator can fire -- it would
# measure propagation plus reweighting and nothing else. A run below this floor
# is a deliberate no-operator control and must say so.
MIN_POPULATION = 6

_WS = re.compile(r'\s+')


def new_motive_id() -> str:
    """64-bit id, same width as particle ids.

    trace_log.new_particle_id was widened from 32 to 64 bits because 32 gives
    ~25% collision at 50k ids, which in a durable store silently merges two
    people's histories under one root. Motive roots go into the same store, so
    they need the same width.
    """
    return new_particle_id()


def normalize_anchor(anchor: Optional[str]) -> str:
    """The canonical form used for identity comparisons.

    Whitespace and case only. Deliberately NOT stemmed: PRECHECKS P-3 measured
    that stemming merges 2,054 of 3,272 distinct anchors (63%) -- `[to maintain]`
    absorbs both "maintain a facade of impartiality" and "maintain his public
    image". Semantic identity is a model's job (merge_equivalent_anchors), never
    a lexical rule's.
    """
    return _WS.sub(' ', (anchor or '')).strip().lower()


class Motive:
    """One commitment inside a portfolio.

    motive_id      this motive as carried by this particle. Minted once at birth
                   and inherited on copy, so a motive that moves between
                   particles is traceable.
    motive_root_id the COMMITMENT's identity. Follows update_anchor's rule: a new
                   anchor text founds a new motive root. Marginal mass is keyed
                   on this, not on motive_id, so two particles that independently
                   hold the same commitment contribute to one marginal.
    born_at        the choice point id at which this commitment was first minted.
                   Carried so a timeline can be drawn without replaying the log.
    """

    __slots__ = ('motive_id', 'motive_root_id', 'anchor', 'standard', 'method',
                 'born_at', 'anchor_revisions')

    def __init__(self, anchor: str, standard: str = None, method: str = None,
                 born_at: str = None, motive_id: str = None,
                 motive_root_id: str = None, anchor_revisions: int = 0):
        self.anchor = anchor
        self.standard = standard
        self.method = method
        self.born_at = born_at
        self.motive_id = motive_id or new_motive_id()
        self.motive_root_id = motive_root_id or self.motive_id
        self.anchor_revisions = anchor_revisions

    # -- identity ---------------------------------------------------------

    @property
    def key(self) -> str:
        return normalize_anchor(self.anchor)

    def copy(self) -> 'Motive':
        """A deep copy that keeps BOTH identities.

        Used by resample. A resampled duplicate is a COPY of a hypothesis, not a
        new one, so it keeps the commitment it was copying -- regenerating
        diversity is perturbation's job. The copy must be deep because motives
        are mutable objects: a shallow copy would let an edit to one duplicate
        mutate the other, which is the class of bug that
        HypothesesSetV3.texts/.anchors already produced once.
        """
        return Motive(self.anchor, self.standard, self.method, self.born_at,
                      motive_id=self.motive_id, motive_root_id=self.motive_root_id,
                      anchor_revisions=self.anchor_revisions)

    def revise(self, new_anchor: str, standard: str = None, method: str = None,
               born_at: str = None) -> 'Motive':
        """Return the motive with a new commitment. A NEW anchor founds a NEW root.

        Returns a new object rather than mutating, so a caller cannot half-apply
        a revision and leave motive_root_id pointing at the previous commitment.

        The standard/method asymmetry is inherited verbatim from
        HypothesisV3.update_anchor and for the same measured reason: a stale
        settling condition is actively WRONG because it settles a commitment the
        particle no longer holds, while a stale method label is merely
        uninformative -- and revive restores a commitment without knowing its
        method, so clobbering method would erase the provenance revive exists to
        bring back.
        """
        if new_anchor is None or normalize_anchor(new_anchor) == self.key:
            return self
        return Motive(
            new_anchor,
            standard=standard,
            method=method if method is not None else self.method,
            born_at=born_at if born_at is not None else self.born_at,
            motive_id=self.motive_id,
            motive_root_id=new_motive_id(),
            anchor_revisions=self.anchor_revisions + 1,
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'motive_id': self.motive_id,
            'motive_root_id': self.motive_root_id,
            'anchor': self.anchor,
            'standard': self.standard,
            'method': self.method,
            'born_at': self.born_at,
            'anchor_revisions': self.anchor_revisions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Motive':
        return cls(d.get('anchor'), d.get('standard'), d.get('method'),
                   d.get('born_at'), motive_id=d.get('motive_id'),
                   motive_root_id=d.get('motive_root_id'),
                   anchor_revisions=d.get('anchor_revisions', 0))

    def __repr__(self) -> str:
        return f"Motive({self.anchor!r}, root={self.motive_root_id[:8]})"


class PortfolioError(ValueError):
    """Raised on an invariant violation.

    Loud rather than silent, deliberately. The repo has already paid for one
    silent desync (HypothesesSetV3.texts drifting from the objects) and three
    silent no-ops in the memory stage; a portfolio whose priority no longer
    covers its motives would corrupt every marginal downstream without anything
    printing.
    """


class Portfolio:
    """A full account of the person: 1-3 motives plus the order they resolve in.

    priority is a list of motive_ids, most-important first, and must be a
    permutation of exactly the motives held. `note` records what the order means
    in words ("m1 wins when they conflict") and is carried into the forecast
    prompt.

    pinned marks the stated-reason particle. It can lose weight but cannot be
    expired, capped away, absorbed by a merge, or dropped by a resample.
    """

    __slots__ = ('motives', 'priority', 'note', 'pinned', 'priority_revisions')

    def __init__(self, motives: Sequence[Motive], priority: Sequence[str] = None,
                 note: str = None, pinned: bool = False,
                 priority_revisions: int = 0):
        self.motives = list(motives)
        self.priority = list(priority) if priority is not None else [m.motive_id for m in self.motives]
        self.note = note
        self.pinned = bool(pinned)
        self.priority_revisions = priority_revisions
        self.validate()

    # -- invariants -------------------------------------------------------

    def validate(self) -> 'Portfolio':
        if not self.motives:
            raise PortfolioError("a portfolio holds at least one motive")
        if len(self.motives) > MAX_MOTIVES:
            raise PortfolioError(
                f"{len(self.motives)} motives exceeds MAX_MOTIVES={MAX_MOTIVES}")
        ids = [m.motive_id for m in self.motives]
        if len(set(ids)) != len(ids):
            raise PortfolioError("duplicate motive_id within one portfolio")
        if sorted(self.priority) != sorted(ids):
            raise PortfolioError(
                "priority must be a permutation covering every motive: "
                f"priority={self.priority} motives={ids}")
        return self

    # -- views ------------------------------------------------------------

    @property
    def ordered(self) -> List[Motive]:
        """Motives in priority order."""
        by_id = {m.motive_id: m for m in self.motives}
        return [by_id[mid] for mid in self.priority]

    @property
    def primary(self) -> Motive:
        """The priority-1 motive.

        HypothesisV3.anchor becomes a derived view of this, which is what keeps
        merge_similar's jaccard, the weight charts, memory.py, dedupe_anchors and
        every audit that reads `anchor` working unchanged on a portfolio run.
        """
        return self.ordered[0]

    @property
    def motive_root_ids(self) -> List[str]:
        return [m.motive_root_id for m in self.motives]

    def holds(self, motive_root_id: str) -> bool:
        return any(m.motive_root_id == motive_root_id for m in self.motives)

    def canonical_key(self) -> Tuple[str, ...]:
        """Identity of the motive SET, order-independent.

        The portfolio analogue of HypothesesSetV3.__init__'s founding
        canonicalisation, which collapses two founding particles carrying the
        same commitment onto one root. Measured there: 5 distinct anchors across
        6 particles founded 6 roots, overstating diversity by one before the run
        started. Order-independent because two founding particles differing only
        in priority are one account explored twice at seed time.
        """
        return tuple(sorted(m.key for m in self.motives))

    # -- edits: every one returns a NEW portfolio -------------------------

    def copy(self) -> 'Portfolio':
        return Portfolio([m.copy() for m in self.motives], list(self.priority),
                         self.note, self.pinned, self.priority_revisions)

    def reorder(self, new_priority: Sequence[str], note: str = None) -> 'Portfolio':
        """Change the conflict order. Does NOT change the motive set."""
        return Portfolio([m.copy() for m in self.motives], list(new_priority),
                         note if note is not None else self.note, self.pinned,
                         self.priority_revisions + 1)

    def expand(self, motive: Motive, note: str = None) -> 'Portfolio':
        """Add a motive. Only legal below MAX_MOTIVES; appended last in priority."""
        if len(self.motives) >= MAX_MOTIVES:
            raise PortfolioError(f"cannot expand past MAX_MOTIVES={MAX_MOTIVES}")
        motives = [m.copy() for m in self.motives] + [motive]
        return Portfolio(motives, list(self.priority) + [motive.motive_id],
                         note if note is not None else self.note, self.pinned,
                         self.priority_revisions)

    def replace(self, motive_id: str, motive: Motive, note: str = None) -> 'Portfolio':
        """Swap one motive for another, keeping its rank. Perturbation's edit."""
        if motive_id not in [m.motive_id for m in self.motives]:
            raise PortfolioError(f"no such motive to replace: {motive_id}")
        motives = [motive if m.motive_id == motive_id else m.copy() for m in self.motives]
        priority = [motive.motive_id if mid == motive_id else mid for mid in self.priority]
        return Portfolio(motives, priority,
                         note if note is not None else self.note, self.pinned,
                         self.priority_revisions)

    def drop(self, motive_root_id: str, note: str = None) -> Optional['Portfolio']:
        """Remove a motive by its ROOT id. Returns None if nothing would remain.

        Keyed on the root rather than the instance id because motive expiry is a
        population-wide decision -- a marginal fell to nothing, so every particle
        holding that commitment drops it. A particle reduced to zero motives is
        retired by the caller; this returns None rather than raising so that
        decision stays with the operator.
        """
        motives = [m.copy() for m in self.motives if m.motive_root_id != motive_root_id]
        if not motives:
            return None
        if len(motives) == len(self.motives):
            return self
        keep = {m.motive_id for m in motives}
        return Portfolio(motives, [mid for mid in self.priority if mid in keep],
                         note if note is not None else self.note, self.pinned,
                         self.priority_revisions)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'motives': [m.to_dict() for m in self.motives],
            'priority': list(self.priority),
            'note': self.note,
            'pinned': self.pinned,
            'priority_revisions': self.priority_revisions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Portfolio':
        return cls([Motive.from_dict(m) for m in d.get('motives', [])],
                   d.get('priority'), d.get('note'), d.get('pinned', False),
                   d.get('priority_revisions', 0))

    def __len__(self) -> int:
        return len(self.motives)

    def __iter__(self):
        return iter(self.ordered)

    def __repr__(self) -> str:
        pin = ' PINNED' if self.pinned else ''
        return f"Portfolio({[m.anchor for m in self.ordered]}{pin})"


# --------------------------------------------------------------------------
# marginal mass
# --------------------------------------------------------------------------

def marginals(portfolios: Iterable[Portfolio], weights: Sequence[float],
              include_pinned: bool = False) -> Dict[str, float]:
    """Total weight of every particle holding each motive, keyed on motive_root_id.

    THE PRIMARY METRIC EXCLUDES THE PINNED PARTICLE. It never dies, so its
    motives always carry at least its weight and a total including it is not
    comparable across population sizes. Callers that want the secondary figure
    pass include_pinned=True and report it as such.

    Weights are used as given and are NOT renormalised after dropping the pinned
    particle. Renormalising would inflate every remaining marginal by
    1/(1-w_pinned), which moves with a quantity that has nothing to do with the
    motives, and would make two steps incomparable whenever the pinned particle's
    weight changed.
    """
    out: Dict[str, float] = {}
    for portfolio, weight in zip(portfolios, weights):
        if portfolio is None:
            continue
        if portfolio.pinned and not include_pinned:
            continue
        for root in set(portfolio.motive_root_ids):
            out[root] = out.get(root, 0.0) + float(weight)
    return out


def marginal_leader(marginal: Dict[str, float]) -> Optional[str]:
    """The heaviest motive root, ties broken by id so the series is deterministic.

    A tie broken by dict order would make churn depend on insertion order, which
    changes under every operator that rebuilds the population -- and churn is the
    exact quantity being measured.
    """
    if not marginal:
        return None
    return max(sorted(marginal), key=lambda r: marginal[r])


def marginal_churn(series: Sequence[Dict[str, float]], per: int = 100) -> Optional[float]:
    """Changes of the leading motive, per `per` steps.

    Defined the same way trace_log.argmax_churn is defined on roots, so the two
    are read side by side rather than swapped for one another. They are NOT
    interchangeable and no gate may compare one against the other's baseline:
    argmax churn is over PARTICLES, this is over MOTIVES, and a new metric is
    always free to look better than an old one.

    Returns None below two steps -- there is no transition to count, and
    reporting 0.0 would read as perfect stability.
    """
    steps = [s for s in series if s]
    if len(steps) < 2:
        return None
    leaders = [marginal_leader(s) for s in steps]
    changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)
    return per * changes / (len(leaders) - 1)


def motive_timeline(series: Sequence[Dict[str, float]]) -> Dict[str, List[float]]:
    """Per-motive marginal across the run, zero-filled where absent.

    Zero-filled rather than sparse so every series has the same length and a
    caller cannot silently align two motives by index when one was born later.
    """
    roots = sorted({r for s in series for r in s})
    return {r: [float(s.get(r, 0.0)) for s in series] for r in roots}
