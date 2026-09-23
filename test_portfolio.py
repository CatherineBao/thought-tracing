"""Offline invariants for the portfolio particle. No LLM calls, no API key.

Two things these tests exist to stop, both of which this repo has already paid
for once:

  1. A silent identity drift. HypothesesSetV3.texts drifting from the objects it
     was supposed to mirror cost a real bug, and the weight chart drew "8 lines
     for 33 distinct roots" because one series held two different commitments.
     Every identity rule below is asserted rather than commented.

  2. A metric that cannot detect its own failure. Marginal mass is the headline
     stability number, so the tests pin what it does when two particles holding
     one motive TRADE PLACES (nothing) and when a motive is dropped (falls to
     exactly zero), not merely that it runs.
"""
import portfolio as P
from portfolio import Motive, Portfolio, PortfolioError


def _m(anchor, **kw):
    return Motive(anchor, standard=kw.pop('standard', f"settled by {anchor}"),
                  method=kw.pop('method', 'anomaly'), born_at=kw.pop('born_at', 'cp-000'), **kw)


def _pf(*anchors, **kw):
    return Portfolio([_m(a) for a in anchors], **kw)


# -- invariants -----------------------------------------------------------

def test_at_most_three_motives():
    _pf('a', 'b', 'c')                       # the cap itself is legal
    try:
        _pf('a', 'b', 'c', 'd')
    except PortfolioError:
        return
    raise AssertionError("four motives must raise")


def test_portfolio_needs_at_least_one_motive():
    try:
        Portfolio([])
    except PortfolioError:
        return
    raise AssertionError("empty portfolio must raise")


def test_priority_must_cover_every_motive():
    ms = [_m('a'), _m('b')]
    Portfolio(ms, [ms[1].motive_id, ms[0].motive_id])       # a permutation is fine
    try:
        Portfolio(ms, [ms[0].motive_id])                    # missing one
    except PortfolioError:
        pass
    else:
        raise AssertionError("short priority must raise")
    try:
        Portfolio(ms, [ms[0].motive_id, ms[0].motive_id])   # not a permutation
    except PortfolioError:
        return
    raise AssertionError("duplicated priority entry must raise")


def test_priority_defaults_to_declaration_order():
    pf = _pf('first', 'second')
    assert [m.anchor for m in pf.ordered] == ['first', 'second']
    assert pf.primary.anchor == 'first'


def test_anchor_is_the_priority_one_motive():
    """`.anchor` on the particle becomes a view of this, which is what keeps
    merge_similar, the charts, memory.py and every audit working unchanged."""
    ms = [_m('low'), _m('high')]
    pf = Portfolio(ms, [ms[1].motive_id, ms[0].motive_id])
    assert pf.primary.anchor == 'high'


# -- motive identity: anchor == root, one level down -----------------------

def test_revising_an_anchor_founds_a_new_motive_root():
    m = _m('gather information')
    r = m.revise('protect reputation', standard='s', method='devil')
    assert r.motive_root_id != m.motive_root_id, "new commitment must found a new root"
    assert r.motive_id == m.motive_id, "the motive keeps its instance identity"
    assert r.anchor_revisions == m.anchor_revisions + 1


def test_revising_to_the_same_anchor_is_a_no_op():
    m = _m('gather information')
    assert m.revise('gather information') is m
    assert m.revise('  Gather   Information ') is m, "identity is whitespace/case-insensitive"


def test_revision_clobbers_standard_but_preserves_method_when_unknown():
    """Inherited verbatim from HypothesisV3.update_anchor, same measured reason:
    a stale standard settles a commitment no longer held (actively wrong); a
    stale method is merely uninformative, and revive restores a commitment
    without knowing its method."""
    m = _m('a', standard='old standard', method='silence')
    r = m.revise('b', standard='new standard', method=None)
    assert r.standard == 'new standard'
    assert r.method == 'silence', "method must survive a revision that does not name one"


def test_motive_copy_keeps_both_identities():
    m = _m('a')
    c = m.copy()
    assert (c.motive_id, c.motive_root_id) == (m.motive_id, m.motive_root_id)
    assert c is not m


def test_portfolio_copy_is_deep():
    """Shallow copies are how a resample duplicate silently mutates its source.
    Motives are objects, so the copy has to reach them."""
    pf = _pf('a', 'b')
    c = pf.copy()
    assert all(x is not y for x, y in zip(pf.motives, c.motives))


# -- edits -----------------------------------------------------------------

def test_reorder_changes_no_motive_identity():
    """Priority order is NOT part of root identity -- making it root-founding
    would mint roots on an operator v2 exists to encourage."""
    pf = _pf('a', 'b')
    before = sorted(pf.motive_root_ids)
    after = pf.reorder(list(reversed(pf.priority)))
    assert sorted(after.motive_root_ids) == before
    assert after.priority_revisions == pf.priority_revisions + 1
    assert after.primary.anchor == 'b'


def test_expand_appends_and_respects_the_cap():
    pf = _pf('a')
    grown = pf.expand(_m('b'))
    assert len(grown) == 2 and grown.priority[-1] == grown.motives[-1].motive_id
    full = _pf('a', 'b', 'c')
    try:
        full.expand(_m('d'))
    except PortfolioError:
        return
    raise AssertionError("expand past the cap must raise")


def test_replace_keeps_rank():
    pf = _pf('a', 'b')
    target = pf.motives[0].motive_id
    new = _m('c')
    out = pf.replace(target, new)
    assert out.priority[0] == new.motive_id, "the replacement inherits the rank"
    assert [m.anchor for m in out.ordered] == ['c', 'b']


def test_drop_removes_by_root_and_returns_none_when_empty():
    pf = _pf('a', 'b')
    root = pf.motives[0].motive_root_id
    out = pf.drop(root)
    assert len(out) == 1 and not out.holds(root)
    assert out.priority == [out.motives[0].motive_id], "priority must stay a permutation"
    solo = _pf('only')
    assert solo.drop(solo.motives[0].motive_root_id) is None


def test_canonical_key_is_order_independent():
    a = _pf('x', 'y')
    b = _pf('y', 'x')
    assert a.canonical_key() == b.canonical_key()


# -- marginal mass ---------------------------------------------------------

def test_marginals_sum_particle_weights():
    shared = _m('shared')
    p1 = Portfolio([shared, _m('only-one')])
    p2 = Portfolio([shared.copy(), _m('only-two')])
    out = P.marginals([p1, p2], [0.6, 0.4])
    assert abs(out[shared.motive_root_id] - 1.0) < 1e-12
    assert abs(out[p1.motives[1].motive_root_id] - 0.6) < 1e-12
    assert abs(out[p2.motives[1].motive_root_id] - 0.4) < 1e-12


def test_marginal_is_churn_invariant_when_holders_trade_places():
    """The whole point of the metric. Two particles holding one motive swapping
    weight must not move that motive's marginal."""
    shared = _m('shared')
    p1 = Portfolio([shared])
    p2 = Portfolio([shared.copy()])
    before = P.marginals([p1, p2], [0.7, 0.3])
    after = P.marginals([p1, p2], [0.3, 0.7])
    assert before == after


def test_a_motive_counted_once_per_particle():
    """Even if two motives in one portfolio somehow share a root, the particle's
    weight is counted once -- otherwise a marginal could exceed total mass."""
    m = _m('a')
    twin = Motive('a-variant', motive_root_id=m.motive_root_id)
    pf = Portfolio([m, twin])
    out = P.marginals([pf], [1.0])
    assert abs(out[m.motive_root_id] - 1.0) < 1e-12


def test_pinned_excluded_by_default_included_on_request():
    pinned = Portfolio([_m('stated reason')], pinned=True)
    free = _pf('hidden aim')
    assert P.marginals([pinned, free], [0.5, 0.5]) == {
        free.motives[0].motive_root_id: 0.5}
    both = P.marginals([pinned, free], [0.5, 0.5], include_pinned=True)
    assert len(both) == 2


def test_marginals_are_not_renormalised_after_dropping_the_pinned_particle():
    """Renormalising would inflate every marginal by 1/(1-w_pinned), which moves
    with a quantity unrelated to the motives and makes two steps incomparable."""
    pinned = Portfolio([_m('stated')], pinned=True)
    free = _pf('hidden')
    out = P.marginals([pinned, free], [0.8, 0.2])
    assert abs(out[free.motives[0].motive_root_id] - 0.2) < 1e-12


# -- churn -----------------------------------------------------------------

def test_marginal_churn_counts_leader_changes():
    a, b = 'root-a', 'root-b'
    series = [{a: 0.9, b: 0.1}, {a: 0.9, b: 0.1}, {a: 0.1, b: 0.9}]
    assert P.marginal_churn(series) == 100 * 1 / 2


def test_marginal_churn_is_none_below_two_steps():
    """None, not 0.0 -- there is no transition to count and 0.0 would read as
    perfect stability."""
    assert P.marginal_churn([]) is None
    assert P.marginal_churn([{'a': 1.0}]) is None


def test_marginal_leader_breaks_ties_deterministically():
    tied = {'bbb': 0.5, 'aaa': 0.5}
    assert P.marginal_leader(tied) == P.marginal_leader(dict(reversed(list(tied.items()))))


def test_timeline_is_zero_filled():
    series = [{'a': 1.0}, {'a': 0.5, 'b': 0.5}]
    tl = P.motive_timeline(series)
    assert tl == {'a': [1.0, 0.5], 'b': [0.0, 0.5]}


# -- serialisation ---------------------------------------------------------

def test_roundtrip_preserves_every_identity():
    pf = _pf('a', 'b', pinned=True, note='a wins when they conflict')
    back = Portfolio.from_dict(pf.to_dict())
    assert back.pinned and back.note == pf.note
    assert [m.motive_id for m in back.ordered] == [m.motive_id for m in pf.ordered]
    assert back.motive_root_ids == pf.motive_root_ids


def test_population_floor_is_stated():
    """Below this the operators cannot fire and a run measures propagation plus
    reweighting only -- split_candidates returns [] under 4, merge under 3."""
    assert P.MIN_POPULATION >= 6


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
