"""Every gate's SIGN, asserted against a fixture that fails it.

WHY THIS FILE EXISTS. Three separate inverted-direction bugs were caught in
review while this design was being written, the last one in a gate that would
have killed CaSiNo before Milestone 1 by requiring a full-transcript reader to
be no better than a constant. A direction stated in prose is not checked by
anything. Here each gate is driven with a value that must FAIL it and a value
that must PASS it, so an inversion cannot survive.

The two arms that look alike and point opposite ways:

    options_only     must NOT EXCEED  majority + delta   (above it, phrasing leaks)
    context_neutral  must NOT FALL BELOW majority - delta (below it, the bar is
                                                           worse than a constant)

The regression fixture is Phase CP's own numbers. Applied retrospectively the
gate must reproduce that run's diagnosis: the option phrasing did NOT leak
(blind 0.622 lost to majority 0.656), and the bar WAS below a constant
(obvious 0.533). If this file ever stops reproducing that, the gate has drifted
away from the failure it was built from.
"""
import label_gate as g


# -- the registry is the contract ------------------------------------------

def test_every_gate_declares_a_direction():
    for name, spec in g.GATES.items():
        assert spec['better'] in ('lower', 'higher'), (name, spec)
        assert spec.get('asks'), f"{name} must say what question it answers"


def test_the_two_lookalike_arms_point_opposite_ways():
    """The single most important assertion in this file."""
    assert g.GATES['options_only']['better'] == 'lower'
    assert g.GATES['context_neutral_sanity']['better'] == 'higher'


# -- options_only: must not BEAT the class prior ---------------------------

def test_options_only_fails_when_it_beats_the_prior():
    """A blind arm that outscores the class prior can only be reading the
    option wording, which is the leak the tripwire exists for."""
    r = g.check_options_only(0.80, majority_acc=0.50)
    assert not r['pass'], "options_only ABOVE majority+delta must FAIL"
    assert 'VOID' in r['verdict_on_fail']


def test_options_only_passes_when_it_merely_recovers_the_prior():
    """Recovering the class prior is free and is NOT leakage. This is the
    correction Phase CP's outcome records as owed: the original rule compared
    blind to CHANCE and voided a run for something a blind guesser gets for
    nothing."""
    assert g.check_options_only(0.50, majority_acc=0.50)['pass']
    assert g.check_options_only(0.20, majority_acc=0.50)['pass'], \
        "well below the prior is fine -- the gate is one-sided"


def test_options_only_boundary_is_inclusive():
    assert g.check_options_only(0.55, majority_acc=0.50, delta=0.05)['pass']
    assert not g.check_options_only(0.5501, majority_acc=0.50, delta=0.05)['pass']


# -- context_neutral sanity: must not be WORSE than the class prior ---------

def test_context_neutral_fails_when_worse_than_a_constant():
    """Phase CP's actual failure: obvious 0.533 against majority 0.656."""
    r = g.check_context_neutral(0.533, majority_acc=0.656)
    assert not r['sanity']['pass'], "a bar below a constant must FAIL"
    assert not r['pass']


def test_context_neutral_sanity_passes_when_better_than_a_constant():
    r = g.check_context_neutral(0.70, majority_acc=0.50, ceiling=1.0)
    assert r['sanity']['pass'] and r['pass']


def test_context_neutral_is_not_the_options_only_gate_inverted():
    """Drive one value through both. It must pass one and fail the other --
    if a refactor ever collapsed them, this catches it."""
    high = 0.95
    assert not g.check_options_only(high, majority_acc=0.50)['pass']
    assert g.check_context_neutral(high, majority_acc=0.50, ceiling=1.0)['sanity']['pass']


# -- context_neutral headroom ----------------------------------------------

def test_headroom_fails_at_ceiling():
    """The plausible CaSiNo outcome: negotiators state their priorities aloud,
    a context-only reader sits at ceiling, and a motive has nothing to add."""
    r = g.check_context_neutral(0.99, majority_acc=0.40, ceiling=1.0, headroom=0.05)
    assert r['sanity']['pass'], "it is not worse than a constant"
    assert not r['headroom']['pass'], "but there is no room left"
    assert not r['pass'], "the arm passes only if BOTH hold"
    assert 'silent subgroup' in r['headroom']['verdict_on_fail']


# -- the no-model statistics ------------------------------------------------

def test_majority_share_direction():
    lo = g.label_stats([{'actual': a, 'person': 'p', 'corpus': 'c', 'n_alternatives': 4}
                        for a in ['A', 'B', 'C', 'D']])
    hi = g.label_stats([{'actual': 'A', 'person': 'p', 'corpus': 'c', 'n_alternatives': 4}
                        for _ in range(4)])
    assert g.evaluate(lo)['checks'][0]['pass'], "a balanced label set must PASS"
    assert not g.evaluate(hi)['checks'][0]['pass'], "a constant must FAIL"


def test_entropy_is_normalised_by_alternatives_offered_not_observed():
    """Normalising by observed classes would score a corpus that only ever uses
    two of seven options a perfect 1.0."""
    labels = ['A', 'B'] * 20
    assert g.normalised_entropy(labels, k=2) == 1.0
    assert g.normalised_entropy(labels, k=7) < 0.4


def test_loo_constant_is_defined_at_one_point_per_person():
    """Per-person majority share reads 1.0 for anyone with a single point and
    fails the corpus on day one; leave-one-out reads it as a miss instead."""
    singles = {('c', f'p{i}'): ['A'] for i in range(10)}
    assert g.loo_constant_accuracy(singles) == 0.0
    repeaters = {('c', 'p'): ['A'] * 10}
    assert g.loo_constant_accuracy(repeaters) == 1.0, "a true constant is detected"


def test_loo_is_pooled_not_averaged_over_people():
    """Somebody with 30 points must not weigh the same as somebody with one."""
    mixed = {('c', 'big'): ['A'] * 30, ('c', 'small'): ['B']}
    acc = g.loo_constant_accuracy(mixed)
    assert acc > 0.9, f"the 30-point constant should dominate, got {acc}"


def test_majority_limit_is_a_formula_not_a_table():
    """majority <= 1/k + SLACK*(1 - 1/k): a constant may sit at most 30% of the
    way from chance to certainty.

    The first version was a flat 0.45, which is ARITHMETICALLY UNREACHABLE at
    k=2 -- a binary label's majority share is at least 0.5 -- so it rejected a
    perfect coin flip. The repair was a k>=3/k==2 table, and a table needs a new
    exception whenever an odd k arrives. One formula does not.
    """
    assert abs(g.majority_limit(2) - 0.65) < 1e-9, "reproduces the binary case"
    assert g.majority_limit(2) > 0.5, "an unreachable limit is not a gate"
    # monotone: more options means a constant must do less well
    lims = [g.majority_limit(k) for k in (2, 3, 4, 7, 8, 15)]
    assert all(a > b for a, b in zip(lims, lims[1:])), lims
    # stricter than the old flat 0.45 wherever there are many options
    assert g.majority_limit(7) < 0.45 and g.majority_limit(8) < 0.45


def test_k_is_what_is_offered_not_the_label_vocabulary():
    """Diplomacy's signed labels span 15 values corpus-wide, but a player is
    offered (2 x powers in contact) + NONE, measured at mean 8.45. Scored
    against the vocabulary it fails by 0.001; against what is offered it passes.
    """
    assert 0.3478 > g.majority_limit(15), "vocabulary-k would reject it"
    assert 0.3478 <= g.majority_limit(8.45), "offered-k accepts it"


def test_the_gate_gives_the_right_verdict_on_every_measured_corpus():
    """One table, so a change to SLACK has to face every corpus at once."""
    cases = [("phase_cp", 0.6560, 7, False),
             ("casino_deal_response", 0.8510, 4, False),
             ("casino_allocation", 0.2337, 7, True),
             ("diplomacy_pair", 0.7594, 3, False),
             ("diplomacy_attack_only", 0.4395, 8, False),
             ("diplomacy_signed", 0.3478, 8.45, True),
             ("avalon_party_vote", 0.7083, 2, False),
             ("avalon_include_exclude", 0.5203, 2, True)]
    for name, maj, k, should_pass in cases:
        got = maj <= g.majority_limit(k)
        assert got == should_pass, f"{name}: majority {maj} at k={k} -> {got}"


def test_the_formula_vindicates_the_signed_diplomacy_label_independently():
    """Signed was adopted on semantic grounds -- `Germany` cannot mean both
    attacking and supporting Germany. The formula reaches the same place from
    the label distribution alone: attack-only now fails, signed passes."""
    assert 0.4395 > g.majority_limit(8), "attack-only fails the formula"
    assert 0.3478 <= g.majority_limit(8.45), "signed passes it"


def test_entropy_alone_cannot_carry_a_binary_choice_type():
    """Normalised by log k, a 0.708/0.292 split scores 0.871 and sails through.
    That is why Avalon's party vote needs the majority check to catch it."""
    labels = ['yes'] * 708 + ['no'] * 292
    assert g.normalised_entropy(labels, k=2) > 0.75, "entropy would pass it"
    assert g.majority_share(labels) > g.majority_limit(2), "majority must fail it"


# -- the regression fixture -------------------------------------------------

def test_phase_cp_numbers_reproduce_their_own_diagnosis():
    points = ([{'actual': 'HOLD', 'person': f'p{i%12}', 'corpus': 'bloomfield',
                'n_alternatives': 7} for i in range(70)]
              + [{'actual': 'CONCEDE', 'person': f'p{i%12}', 'corpus': 'bloomfield',
                  'n_alternatives': 7} for i in range(30)])
    res = g.evaluate(g.label_stats(points), options_only_acc=0.622,
                     context_neutral_acc=0.533, majority_acc=0.656)
    by = {c['gate']: c for c in res['checks']}
    assert not res['pass'], "Phase CP's label distribution must not pass this gate"
    assert not by['majority_share']['pass'], "70% HOLD is the finding"
    assert by['options_only']['pass'], \
        "blind 0.622 LOST to majority 0.656 -- not leakage, which is the correction"
    assert not by['context_neutral']['sanity']['pass'], \
        "obvious 0.533 was below a constant -- the bar was the problem"


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
