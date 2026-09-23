"""Mass conservation for the PORTFOLIO operators.

test_phase4_mass.py already pins the two invariants for single-anchor split and
merge, and those stay green -- content-split refines one motive into two
mutually exclusive ones, so it founds roots exactly the way the old split did.
This file adds what portfolios change, as POSITIVE assertions rather than by
weakening the existing test:

  order-split      forks the priority order, NOT the motive set. So it conserves
                   weight and root mass like any split, but distinct_roots stays
                   PUT -- invariant 4 of test_phase4_mass (`distinct_roots +=
                   children - 1`) deliberately does not hold for it. Asserting
                   that explicitly is the point: if a reorder ever started
                   founding roots it would mint births on an operator v2 exists
                   to encourage, and the churn series would silently refill with
                   exactly the noise the portfolio was built to remove.

  motive expiry    drops one commitment from every particle holding it. Particle
                   weight is untouched, so total mass is conserved -- but the
                   dropped motive's MARGINAL must go to exactly zero. Marginal
                   mass is not conserved as a measure over motives, and that is
                   by design; the test states the accounting rather than leaving
                   it to be discovered.

  merge            the survivor keeps every marginal it held; a motive only the
                   absorbed particle carried goes to zero.
"""
import numpy as np

import portfolio as P
from portfolio import Motive, Portfolio
from hypothesis import HypothesesSetV3

TOL = 1e-9


def _mass_by_root(hyps):
    out = {}
    for h, w in zip(hyps.hypotheses, hyps.weights):
        out[h.root_id] = out.get(h.root_id, 0.0) + float(w)
    return out


def _marginals(hyps, include_pinned=False):
    return P.marginals(hyps.portfolios, hyps.weights, include_pinned=include_pinned)


def build(portfolios, weights):
    return HypothesesSetV3('Target', [], [], [f"t{i}" for i in range(len(portfolios))],
                           np.array(weights, dtype=float), portfolios=portfolios)


# -- order split -----------------------------------------------------------

def test_order_split_conserves_mass_and_founds_no_root():
    a, b = Motive('hit the deadline'), Motive('get it right')
    keep = Portfolio([a, b], [a.motive_id, b.motive_id],
                     note='deadline wins when they conflict')
    other = Portfolio([Motive('stay out of it')])
    hyps = build([keep, other], [0.6, 0.4])

    before_roots = _mass_by_root(hyps)
    before_distinct = len(set(hyps.root_ids))
    before_marg = _marginals(hyps)

    # the fork: same motives, opposite order, parent mass split evenly
    forked = keep.reorder([b.motive_id, a.motive_id],
                          note='getting it right wins when they conflict')
    out = build([keep, other, forked], [0.3, 0.4, 0.3])

    assert abs(sum(float(w) for w in out.weights) - 1.0) < TOL, "total mass moved"

    after_roots = _mass_by_root(out)
    assert abs(sum(after_roots.values()) - sum(before_roots.values())) < TOL

    # THE assertion: a reorder is not a new account, so no root was founded.
    assert len(set(out.root_ids)) == before_distinct, \
        "priority reorder must NOT found a root -- see set_portfolio"

    # and the motives themselves are untouched, so every marginal is unchanged
    assert set(_marginals(out)) == set(before_marg)
    for root, mass in before_marg.items():
        assert abs(_marginals(out)[root] - mass) < TOL, \
            "reordering redistributes nothing: the same particles still hold the same motives"


def test_order_split_changes_which_motive_leads():
    """Conserving mass is not enough -- the fork has to actually differ."""
    a, b = Motive('hit the deadline'), Motive('get it right')
    keep = Portfolio([a, b], [a.motive_id, b.motive_id])
    forked = keep.reorder([b.motive_id, a.motive_id])
    assert keep.primary.anchor != forked.primary.anchor
    assert keep.canonical_key() == forked.canonical_key()


# -- content split ---------------------------------------------------------

def test_content_split_founds_a_root_and_conserves_mass():
    """The portfolio analogue of the old split, and invariant 4 DOES hold."""
    broad = Motive('protect the project')
    parent = Portfolio([broad])
    other = Portfolio([Motive('stay out of it')])
    hyps = build([parent, other], [0.6, 0.4])
    before_distinct = len(set(hyps.root_ids))

    kid_a = parent.replace(broad.motive_id, Motive('protect the schedule'))
    kid_b = parent.replace(broad.motive_id, Motive('protect the budget'))
    out = build([kid_a, other, kid_b], [0.3, 0.4, 0.3])

    assert abs(sum(float(w) for w in out.weights) - 1.0) < TOL
    assert len(set(out.root_ids)) == before_distinct + 1, \
        "a content split founds one extra root (children - 1)"
    marg = _marginals(out)
    assert broad.motive_root_id not in marg, "the broad motive is gone"
    assert abs(marg[kid_a.primary.motive_root_id] - 0.3) < TOL
    assert abs(marg[kid_b.primary.motive_root_id] - 0.3) < TOL


# -- motive expiry ---------------------------------------------------------

def test_motive_expiry_conserves_particle_weight_and_zeroes_the_marginal():
    dying = Motive('impress the new hire')
    living = Motive('ship the release')
    p1 = Portfolio([dying, living])
    p2 = Portfolio([dying.copy(), Motive('avoid blame')])
    hyps = build([p1, p2], [0.5, 0.5])

    before = _marginals(hyps)
    assert abs(before[dying.motive_root_id] - 1.0) < TOL

    survivors = [pf.drop(dying.motive_root_id) for pf in hyps.portfolios]
    out = build(survivors, [0.5, 0.5])

    assert abs(sum(float(w) for w in out.weights) - 1.0) < TOL, \
        "dropping a motive moves no particle weight"
    after = _marginals(out)
    assert dying.motive_root_id not in after, "the expired marginal must be exactly zero"
    assert abs(after[living.motive_root_id] - before[living.motive_root_id]) < TOL, \
        "an untouched motive keeps its marginal exactly"


def test_motive_expiry_is_not_marginal_mass_conservation():
    """Stated as an assertion so nobody later 'fixes' the total back to 1.0.

    Marginal mass is a measure over MOTIVES and is only conserved when the
    motive sets are equal. A drop removes mass from the motive plane while the
    particle plane is untouched; that asymmetry is the design, not a leak.
    """
    m = Motive('a')
    pf = Portfolio([m, Motive('b')])
    hyps = build([pf], [1.0])
    total_before = sum(_marginals(hyps).values())
    out = build([pf.drop(m.motive_root_id)], [1.0])
    total_after = sum(_marginals(out).values())
    assert total_after < total_before
    assert abs(sum(float(w) for w in out.weights) - 1.0) < TOL


def test_expiring_the_last_motive_retires_the_particle():
    solo = Motive('only aim')
    pf = Portfolio([solo])
    assert pf.drop(solo.motive_root_id) is None, \
        "drop returns None rather than raising: retiring is the operator's call"


# -- merge -----------------------------------------------------------------

def test_merge_keeps_survivor_marginals_and_zeroes_absorbed_only_ones():
    shared = Motive('keep the customer')
    survivor_only = Motive('keep the schedule')
    absorbed_only = Motive('keep the peace')
    survivor = Portfolio([shared, survivor_only])
    absorbed = Portfolio([shared.copy(), absorbed_only])
    hyps = build([survivor, absorbed], [0.5, 0.5])
    before = _marginals(hyps)

    # absorb: the survivor takes the absorbed particle's weight
    out = build([survivor], [1.0])
    after = _marginals(out)

    assert abs(sum(float(w) for w in out.weights) - 1.0) < TOL, "merge conserves total mass"
    assert abs(after[shared.motive_root_id] - before[shared.motive_root_id]) < TOL, \
        "a motive both particles held keeps its marginal exactly"
    assert after[survivor_only.motive_root_id] > before[survivor_only.motive_root_id], \
        "the survivor's own motive gains the absorbed weight"
    assert absorbed_only.motive_root_id not in after, \
        "a motive only the absorbed particle carried goes to zero"


# -- the pinned particle ---------------------------------------------------

def test_pinned_weight_is_not_redistributed_into_the_primary_marginal():
    pinned = Portfolio([Motive('what he says he wants')], pinned=True)
    free = Portfolio([Motive('what he actually wants')])
    hyps = build([pinned, free], [0.75, 0.25])
    primary = _marginals(hyps)
    secondary = _marginals(hyps, include_pinned=True)
    assert abs(primary[free.primary.motive_root_id] - 0.25) < TOL, \
        "the primary metric must not renormalise -- it would move with w_pinned"
    assert abs(secondary[pinned.primary.motive_root_id] - 0.75) < TOL


if __name__ == '__main__':
    import sys
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  ok   {n}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
