"""The four clauses of the choice definition, where they are measurable.

PREREG defines a choice as:

    a moment, identifiable from what came before, where the person had at least
    two genuinely feasible and distinguishable courses of action, and where
    plausible motives would favour different ones.

Clauses 1 and 2 -- identifiable from the prefix, genuinely feasible -- are
properties of each corpus's extraction rule and are checked by the converters
and by test_hindsight.py. Clauses 3 and 4 are properties of the MOMENT and can
only be read once forecasts exist, so they are measured here.

THE POINT OF CLAUSE 4. Phase CP's whole apparatus landed on a constant. One
reading is that the motives were empty; another is that the moments were, and
nothing in that run could tell them apart. Disagreement separates the two: if
lift concentrates on the choice points where the competing accounts actually
forecast differently, the motives are doing discriminating work. If lift is flat
across the strata, they are not -- whatever the pooled number says.

DISAGREEMENT MUST NEVER BE A FILTER. Dropping low-disagreement points is
selection on the model's own output: it discards exactly the moments the
population found uninformative and reports the rest as if it were the task.
That is the same error as conditioning on "the majority action was not taken".
"""
import math

import prequential as q


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# -- clause 4: do plausible motives favour different options? ---------------

def test_identical_accounts_disagree_by_nothing():
    """The definition's whole point: a moment where every account predicts the
    same thing is not a choice, however dramatic the transcript."""
    same = [{'A': 0.7, 'B': 0.2, 'C': 0.1}] * 5
    assert _close(q.forecast_disagreement(same), 0.0)


def test_maximally_opposed_accounts_disagree_by_one_bit():
    opposed = [{'A': 1.0, 'B': 0.0}, {'A': 0.0, 'B': 1.0}]
    assert _close(q.forecast_disagreement(opposed), 1.0)


def test_disagreement_is_bounded_by_the_weight_entropy():
    """D <= H(w): a population cannot express more disagreement than it has
    spread. This is why the normalised form exists -- the raw figure is not
    comparable across populations of different size or concentration."""
    dists = [{'A': 1.0}, {'B': 1.0}, {'C': 1.0}, {'D': 1.0}]
    w = [0.7, 0.1, 0.1, 0.1]
    d = q.forecast_disagreement(dists, w)
    hw = -sum(x * math.log(x, 2) for x in w)
    assert d <= hw + 1e-9
    assert _close(q.forecast_disagreement(dists, w, normalise=True), d / hw)


def test_concentrated_population_disagrees_less_than_a_spread_one():
    """Two accounts that contradict each other buy nothing if one holds almost
    all the weight -- there is effectively one account in play."""
    opposed = [{'A': 1.0, 'B': 0.0}, {'A': 0.0, 'B': 1.0}]
    even = q.forecast_disagreement(opposed, [0.5, 0.5])
    skewed = q.forecast_disagreement(opposed, [0.95, 0.05])
    assert skewed < even


def test_disagreement_never_negative_and_zero_below_two_accounts():
    assert q.forecast_disagreement([]) == 0.0
    assert q.forecast_disagreement([{'A': 1.0}]) == 0.0
    noisy = [{'A': 0.3333333333, 'B': 0.6666666667}] * 3
    assert q.forecast_disagreement(noisy) >= 0.0


def test_disagreement_ignores_option_labels_it_never_sees():
    """Options absent from an account's distribution are read as zero mass, not
    as missing data -- otherwise an account that simply never mentioned an
    option would look like it disagreed about it."""
    a = [{'A': 0.5, 'B': 0.5}, {'A': 0.5, 'B': 0.5, 'C': 0.0}]
    assert _close(q.forecast_disagreement(a), 0.0)


def test_normalisation_uses_the_tight_bound_not_the_weight_entropy_alone():
    """D <= min(H(w), log2 k). Dividing by H(w) alone understates a saturated
    point: four accounts maximally opposed over TWO options have H(w) = 2 bits
    but can only ever reach 1, so H(w) alone would report 0.5 for a population
    that is as split as the option set allows."""
    d = [{'A': 1.0, 'B': 0.0}, {'A': 0.0, 'B': 1.0},
         {'A': 1.0, 'B': 0.0}, {'A': 0.0, 'B': 1.0}]
    assert _close(q.forecast_disagreement(d), 1.0)
    assert _close(q.forecast_disagreement(d, normalise=True), 1.0)


def test_more_options_can_score_higher_which_is_why_type_matters():
    """The confound the within-type stratification exists for: a choice type
    offering more options can reach a higher raw score for free."""
    two = [{'A': 1.0, 'B': 0.0}, {'A': 0.0, 'B': 1.0}]
    four = [{'A': 1.0}, {'B': 1.0}, {'C': 1.0}, {'D': 1.0}]
    assert q.forecast_disagreement(four) > q.forecast_disagreement(two)


def test_stratification_can_be_resolved_within_choice_type():
    """A single corpus-wide cut on a raw statistic would sort partly BY TYPE."""
    pts = [{'kind': 'a', 'disagreement': 0.1}, {'kind': 'a', 'disagreement': 0.9},
           {'kind': 'b', 'disagreement': 1.6}, {'kind': 'b', 'disagreement': 1.9}]
    within = q.stratify_by_disagreement(pts, {'a': 0.5, 'b': 1.7}, within='kind')
    assert within['n_low'] == 2 and within['n_high'] == 2
    globalcut = q.stratify_by_disagreement(pts, 0.5)
    assert globalcut['n_high'] == 3, "a global cut puts both of type b in high"
    assert within['n_low'] + within['n_high'] == len(pts), "still not a filter"


# -- clause 3: are the listed options distinguishable? ----------------------

def test_separability_reads_the_max_over_accounts_not_the_mean():
    """One account that separates a pair makes the pair a real fork; a mean
    would let a crowd of indifferent accounts hide it."""
    dists = [{'A': 0.5, 'B': 0.5}, {'A': 0.5, 'B': 0.5}, {'A': 0.9, 'B': 0.1}]
    sep = q.option_separability(dists)
    assert _close(sep[('A', 'B')], 0.8)


def test_options_no_account_separates_score_zero():
    """A pair nothing separates is, as far as anything here can tell, one action
    written twice -- a prompt to look at the option set by hand."""
    dists = [{'A': 0.4, 'B': 0.4, 'C': 0.2}] * 4
    assert _close(q.option_separability(dists)[('A', 'B')], 0.0)
    assert q.option_separability(dists)[('A', 'C')] > 0.0


# -- the stratification ----------------------------------------------------

def test_cut_is_frozen_on_dev_and_is_a_rule_not_a_number():
    dev = [0.0, 0.1, 0.2, 0.8, 0.9]
    cut = q.disagreement_cut(dev, quantile=0.5)
    assert cut == 0.2
    # the same rule on a different sample gives a different number, which is the
    # point: re-cutting on test would let the boundary move to wherever the lift
    # happened to be
    assert q.disagreement_cut([0.5, 0.6, 0.7], quantile=0.5) == 0.6


def test_stratify_keeps_every_point_in_exactly_one_stratum():
    """Nothing is dropped. Both strata are reported."""
    pts = [{'disagreement': v} for v in (0.0, 0.1, 0.2, 0.8, 0.9)]
    s = q.stratify_by_disagreement(pts, cut=0.2)
    assert s['n_low'] + s['n_high'] == len(pts), "stratification is not a filter"
    assert s['n_low'] == 3 and s['n_high'] == 2


def test_points_with_no_disagreement_recorded_fall_in_the_low_stratum():
    """A missing value is uninformative, and treating it as high would quietly
    promote unmeasured points into the stratum the claim rests on."""
    s = q.stratify_by_disagreement([{'x': 1}], cut=0.0)
    assert s['n_low'] == 1


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
