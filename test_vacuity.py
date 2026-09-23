"""The vacuity probe's cut points, all of which mis-attribute silently if wrong.

audit_vacuity re-scores a LOGGED slate against a substituted action. Every step
of that rests on cutting a logged prompt into three blocks and cutting the
middle block into hypotheses, and both cuts failed on real dumps in ways that
produced plausible-looking numbers rather than an error:

  * a hypothesis is free text and routinely spans several paragraphs, so
    splitting the slate on newlines reported n_sent=14 for an 8-particle
    population and mis-indexed every rank in the step;
  * a hypothesis body often contains its OWN numbered list, so splitting on
    every "N. " returned item numbers [1,2,3,4,5,6,1,2,3,4,7,8].

Both fixtures below are shaped after the Oppenheimer dumps where those
occurred. The third cut -- <observed next action> -- is the one the whole
experiment turns on: if it is mis-placed, the true action stays inside the
context block, the substitution is a no-op, and the result reads as perfect
vacuity on every hypothesis at once.
"""
import audit_vacuity as av
from tracer import rank_scorer_prompts


def make(slate_items, action="Groves: Let's go.", ctx="<context 1>\nsomething\n</context 1>"):
    block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(slate_items))
    _, prompt = rank_scorer_prompts("Groves", len(slate_items), ctx, block, action)
    return prompt


# --------------------------------------------------------------------------
# cutting the prompt
# --------------------------------------------------------------------------

def test_single_line_hypotheses_parse():
    p = make(["He wants secrecy.", "He wants speed.", "He wants credit."])
    got = av.parse_prompt(p, expect_n=3)
    assert got is not None
    assert got["slate"] == ["He wants secrecy.", "He wants speed.", "He wants credit."]
    assert got["action"] == "Groves: Let's go."


def test_multi_paragraph_hypothesis_is_one_item():
    body = "Groves believed the delay was political.\n\nHe also believed Oppenheimer knew it."
    p = make([body, "Groves wanted the schedule held."])
    got = av.parse_prompt(p, expect_n=2)
    assert got is not None, "a multi-paragraph hypothesis must not become two items"
    assert len(got["slate"]) == 2
    assert got["slate"][0] == body


def test_nested_numbered_list_does_not_steal_a_boundary():
    # the measured failure: the body's own 1..4 restarted the numbering
    body = ("Groves saw several reasons:\n"
            "1.  Reduce overhead\n2.  Maintain morale\n3.  Use available talent\n"
            "4.  Preserve secrecy")
    p = make(["Groves wanted the wives employed.", body, "Groves wanted Condon gone."])
    got = av.parse_prompt(p, expect_n=3)
    assert got is not None
    assert len(got["slate"]) == 3, f"nested list stole a boundary: {len(got['slate'])} items"
    assert got["slate"][1] == body


def test_expect_n_mismatch_refuses_rather_than_guessing():
    p = make(["a", "b", "c"])
    assert av.parse_prompt(p, expect_n=4) is None
    assert av.parse_prompt(p, expect_n=2) is None


def test_action_block_is_cut_off_from_the_context():
    p = make(["a", "b"], action="Groves: Fire him.", ctx="Groves: Fire him.")
    got = av.parse_prompt(p, expect_n=2)
    # the same sentence appears in both blocks; the action cut must still be
    # the LAST block, or substitution silently leaves the true action in place
    assert got["action"] == "Groves: Fire him."
    assert got["ctx"] == "Groves: Fire him."


def test_a_prompt_that_is_not_the_rank_prompt_is_refused():
    assert av.parse_prompt("just some text", expect_n=3) is None
    assert av.parse_prompt("", expect_n=3) is None


def test_rebuild_round_trips():
    """The substitution rebuilds the prompt; anything but byte equality is a
    different experiment from the one the filter ran."""
    items = ["Groves wanted secrecy.\n\nAnd speed.", "Groves wanted Condon gone."]
    p = make(items)
    got = av.parse_prompt(p, expect_n=2)
    _, rebuilt = rank_scorer_prompts("Groves", got and len(got["slate"]),
                                     got["ctx"], got["slate_block"], got["action"])
    assert rebuilt == p


def test_substitution_changes_only_the_action():
    p = make(["a", "b"], action="TRUE ACTION")
    got = av.parse_prompt(p, expect_n=2)
    _, swapped = rank_scorer_prompts("Groves", 2, got["ctx"], got["slate_block"], "DECOY")
    assert "DECOY" in swapped and "TRUE ACTION" not in swapped
    assert av.parse_prompt(swapped, expect_n=2)["slate"] == got["slate"]
    assert av.parse_prompt(swapped, expect_n=2)["ctx"] == got["ctx"]


# --------------------------------------------------------------------------
# ranks
# --------------------------------------------------------------------------

def test_ranks_follow_the_stated_ranking_not_the_array_index():
    """Rank-keyed allocation, the majority convention. Hypothesis 2 is 1st."""
    raw = ("RANKING\n1st: 2\n2nd: 4\n3rd: 1\n4th: 3\n\n"
           "ALLOCATION\n1: 95\n2: 80\n3: 60\n4: 10\n")
    ranks = av.ranks_from(raw, 4)
    assert ranks[1] == 1, f"hypothesis 2 should rank 1st, got {ranks}"
    assert ranks[2] == 4


def test_unparseable_response_yields_no_ranks():
    assert av.ranks_from("the answer is unclear", 4) is None
    assert av.ranks_from("", 4) is None


# --------------------------------------------------------------------------
# decoys
# --------------------------------------------------------------------------

def test_a_restatement_is_not_a_decoy():
    a = "Katara: The Fire Nation can't separate our family again!"
    assert av.overlap(a, a) == 1.0
    assert av.overlap(a, "Katara: The Fire Nation cannot separate our family!") > 0.5
    assert av.overlap(a, "Sokka: I found the tunnel entrance.") < 0.3


def test_stopwords_do_not_manufacture_overlap():
    assert av.overlap("it is the one that was", "they are the ones that were") == 0.0


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
