"""The rank-mode scorer's ALLOCATION block is ambiguously keyed, and both
readings occur in real output.

The prompt asks for RANKING ("1st: <hypothesis>") and then ALLOCATION
("1: <0-100>"). Following the ordinals, the model usually treats the allocation
keys as RANK POSITIONS. parse_allocation read them as HYPOTHESIS INDICES, so
the top likelihood went to whatever sat at index 0 -- measured at 1847 of 2068
scored steps across musing_out, with the parser's top pick disagreeing with the
model's own stated 1st place on 1527 of them.

Both fixtures below are verbatim scorer output from
musing_out/traces/tracer-...runid-ep_Katara_nhypotheses-8.jsonl, chosen because
they use OPPOSITE conventions with nothing in the text to distinguish them.
That is the whole difficulty, and it is why the resolution is inferred from
agreement with the RANKING block rather than from a format rule.
"""
import numpy as np

from tracer import (map_allocation_to_hypotheses, order_agreement,
                    parse_allocation, parse_ranking)

N = 8

# Step 36 -- the episode's reversal ("But as much as I hate you ... I just
# can't do it."). The model ranked hypothesis 2 best; the allocation descends
# monotonically, so it is keyed by RANK.
STEP36 = (
    "RANKING\n1st: 2\n2nd: 4\n3rd: 1\n4th: 6\n5th: 8\n6th: 3\n7th: 5\n8th: 7\n\n"
    "ALLOCATION\n1: 95\n2: 90\n3: 85\n4: 80\n5: 60\n6: 55\n7: 50\n8: 10\n\n"
    "REASONING\nThe best hypotheses predict Katara's internal conflict and her "
    "ultimate decision not to kill Yon Rha.")

# Step 0 -- same run, same prompt, keyed by HYPOTHESIS. The allocation is not
# monotone and its argmax already sits on the hypothesis ranked 1st.
STEP0 = (
    "RANKING\n1st: 2\n2nd: 5\n3rd: 1\n4th: 6\n5th: 4\n6th: 7\n7th: 3\n8th: 8\n\n"
    "ALLOCATION\n1: 70\n2: 95\n3: 20\n4: 40\n5: 85\n6: 35\n7: 30\n8: 10\n\n"
    "REASONING\nHypothesis 2 is the most specific predictor of the response.")


def resolve(raw, n=N):
    """The production path: parse both blocks, resolve the keying."""
    alloc, err = parse_allocation(raw, n)
    assert alloc is not None, err
    order, rank_err = parse_ranking(raw, n)
    assert order is not None, rank_err
    return map_allocation_to_hypotheses(alloc, order)


def test_ranking_parses_to_a_permutation():
    order, err = parse_ranking(STEP36, N)
    assert err is None
    assert order == [1, 3, 0, 5, 7, 2, 4, 6]        # 0-based, best first
    assert sorted(order) == list(range(N))


def test_allocation_ignores_the_ranking_block():
    """parse_allocation must not read "1st: 2" as an allocation pair."""
    alloc, err = parse_allocation(STEP36, N)
    assert err is None
    assert alloc == [95, 90, 85, 80, 60, 55, 50, 10]


def test_rank_keyed_allocation_credits_the_hypothesis_the_model_ranked_first():
    scores, keying, agreement = resolve(STEP36)
    assert keying == 'rank'
    assert agreement == 1.0
    # hypothesis 2 (index 1) was ranked 1st and must receive the top score.
    assert scores[1] == 95
    assert int(np.argmax(scores)) == 1
    # The old positional read handed 95 to index 0. That is the bug.
    assert int(np.argmax(parse_allocation(STEP36, N)[0])) == 0


def test_hypothesis_keyed_allocation_is_left_alone():
    scores, keying, agreement = resolve(STEP0)
    assert keying == 'hypothesis'
    assert scores == [70, 95, 20, 40, 85, 35, 30, 10]
    # Still hypothesis 2 (index 1), reached by the other convention.
    assert int(np.argmax(scores)) == 1
    assert agreement > 0.9


def test_resolution_agrees_with_the_stated_ranking_in_both_cases():
    """Whichever convention was used, the argmax lands on the model's 1st."""
    for raw in (STEP36, STEP0):
        order, _ = parse_ranking(raw, N)
        scores, _, _ = resolve(raw)
        assert int(np.argmax(scores)) == order[0]


def test_order_agreement_bounds():
    order = [0, 1, 2, 3]
    assert order_agreement([4, 3, 2, 1], order) == 1.0      # exact
    assert order_agreement([1, 2, 3, 4], order) == 0.0      # reversed
    assert order_agreement([1, 1, 1, 1], order) == 0.5      # all ties


def test_partial_ranking_refuses_rather_than_guessing():
    """A half-parsed order would mis-key every score, so it is not returned."""
    raw = ("RANKING\n1st: 2\n2nd: 4\n\n"
           "ALLOCATION\n1: 95\n2: 90\n3: 85\n4: 80\n5: 60\n6: 55\n7: 50\n8: 10\n")
    order, err = parse_ranking(raw, N)
    assert order is None and 'expected 8 ranks' in err
    # The allocation itself still parses; the caller falls back positionally.
    assert parse_allocation(raw, N)[0] is not None


def test_non_permutation_ranking_is_rejected():
    raw = ("RANKING\n1st: 2\n2nd: 2\n3rd: 1\n4th: 6\n5th: 8\n6th: 3\n7th: 5\n8th: 7\n\n"
           "ALLOCATION\n1: 95\n2: 90\n3: 85\n4: 80\n5: 60\n6: 55\n7: 50\n8: 10\n")
    order, err = parse_ranking(raw, N)
    assert order is None and 'permutation' in err


def test_missing_ranking_block_does_not_crash():
    raw = "ALLOCATION\n1: 95\n2: 90\n3: 85\n4: 80\n5: 60\n6: 55\n7: 50\n8: 10\n"
    order, err = parse_ranking(raw, N)
    assert order is None and err
    assert parse_allocation(raw, N)[0] == [95, 90, 85, 80, 60, 55, 50, 10]


def test_formatting_variants():
    raw = ("RANKING\n**1st**: H2\n2) 4\n3. 1\n4th - 6\n5th: 8\n6th: 3\n7th: 5\n8th: 7\n\n"
           "ALLOCATION\n**1**: 95\n2. 90\n3 - 85\n4: 80\n5: 60\n6: 55\n7: 50\n8: 10\n")
    order, err = parse_ranking(raw, N)
    assert err is None and order == [1, 3, 0, 5, 7, 2, 4, 6]
    assert parse_allocation(raw, N)[0] == [95, 90, 85, 80, 60, 55, 50, 10]


def test_identical_scores_stay_flat():
    """A genuinely undiscriminating verdict must not be given an order.

    probe_scorer.py's guard depends on this: identical hypotheses should
    produce a flat likelihood, and the resolution must not manufacture one.
    """
    raw = ("RANKING\n1st: 1\n2nd: 2\n3rd: 3\n4th: 4\n\n"
           "ALLOCATION\n1: 50\n2: 50\n3: 50\n4: 50\n")
    scores, _, _ = resolve(raw, 4)
    assert scores == [50, 50, 50, 50]


def test_all_zero_allocation_survives_resolution():
    raw = ("RANKING\n1st: 3\n2nd: 1\n3rd: 2\n4th: 4\n\n"
           "ALLOCATION\n1: 0\n2: 0\n3: 0\n4: 0\n")
    scores, _, _ = resolve(raw, 4)
    assert sum(scores) == 0      # caller treats this as 'no hypothesis predicts this'


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'FAILED' if fails else 'all passed'} ({fails} failure(s))")
    sys.exit(1 if fails else 0)
