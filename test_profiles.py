"""Guards for the external baseline profile (profiles.py + its two call sites).

Offline: the model is stubbed and no API client is constructed.

The first test is the one that matters. Every measurement in musing_out was
taken against the prompts as they stand, so a one-character drift in the
default path destroys more evidence than this feature can generate. Same rule
test_methods.py protects for methods.py, for the same reason.

  .venv/bin/python test_profiles.py
"""
import json
import os
import sys
import types

import profiles
import tracer


# --------------------------------------------------------------------------
# the default path must not move
# --------------------------------------------------------------------------

class _StubModel:
    def __init__(self):
        self.prompts = []

    def interact(self, prompt, **kw):
        self.prompts.append(prompt)
        return "\n".join(f"{i + 1}. Wolf believes thing {i}" for i in range(8))

    def batch_interact(self, prompts, **kw):
        return [self.interact(p) for p in prompts]


def _seed(**over):
    t = tracer.Tracer.__new__(tracer.Tracer)
    args = dict(n_hypotheses=4, methods=None, goal_seeding=True,
                infer_motive=False, extract_anchors_flag=False, use_anchor=False,
                dataset='musing', use_cot=False, input_is_chat=True,
                target_perceptions='sight', use_helper_llm=False,
                character_profile=False, profiles=None, profile_roster=None)
    args.update(over)
    t.args = types.SimpleNamespace(**args)
    t.target_agent = 'Wolf'
    t.assumption = ''
    t.tracer_model = _StubModel()
    t._last_standards = []
    return t


def _run_seed(t):
    tracer.Tracer.initialize(t, {'state': 'ctx', 'action': 'Wolf speaks.'},
                             {'state': '', 'action': ''})
    return t.tracer_model.prompts[0]


def test_default_path_prompt_is_byte_identical():
    """Flag off, and no _character_profile attribute at all -- which is the
    state of every tracer constructed before this feature existed."""
    off = _run_seed(_seed())
    assert 'prior working history' not in off
    assert 'SEAT' not in off

    # and with the attribute explicitly present but empty
    t = _seed()
    t._character_profile = None
    assert _run_seed(t) == off


def test_profile_reaches_the_seed_prompt():
    t = _seed(character_profile=True)
    t._character_profile = "SEAT: runs the customer side\nSTAKE: keep the accounts confident"
    prompt = _run_seed(t)
    assert 'prior working history' in prompt
    assert 'runs the customer side' in prompt
    # the sentence that makes the block usable travels with it
    assert 'Let the hypotheses follow from that SEAT and STAKE' in prompt


def test_profile_replaces_the_inferred_prior_rather_than_stacking():
    """Two SEAT/STAKE blocks in one prompt is the same claim twice, and it
    makes the two arms inseparable."""
    called = []
    t = _seed(character_profile=True, infer_motive=True)
    t._character_profile = "SEAT: runs the customer side"
    t.infer_role_prior = lambda: called.append(1) or "SEAT: inferred seat"
    prompt = _run_seed(t)
    assert not called, "paid for a role-prior call the profile had already answered"
    assert 'inferred seat' not in prompt
    assert prompt.count('Let the hypotheses follow from that SEAT and STAKE') == 1


def test_infer_motive_still_works_when_no_profile_is_present():
    t = _seed(infer_motive=True)
    t._character_profile = None
    t.infer_role_prior = lambda: "SEAT: inferred seat"
    assert 'inferred seat' in _run_seed(t)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

FIX = {
    "_README": "fabricated fixture",
    "A": {"seat": "seat A", "stake": "stake A", "pressure": "pressure A",
          "stance": ["blunt", "fast"], "standing": "standing A",
          "relations": {"B": "leans on them"}, "quotes": ["quoted words"],
          "read_confidence": 0.9, "source": "fabricated"},
    "B": {"seat": "seat B", "source": "fabricated"},
    "C": {"seat": "seat C", "source": "fabricated"},
}


def test_metadata_keys_are_not_people():
    assert set(profiles.people(FIX)) == {"A", "B", "C"}


def test_quotes_are_withheld_by_default():
    assert "quoted words" not in profiles.render_profile(FIX, "A", ["B"])
    assert "quoted words" in profiles.render_profile(FIX, "A", ["B"], include_quotes=True)


def test_relation_is_rendered_with_the_target_as_subject():
    out = profiles.render_profile(FIX, "A", ["B"])
    assert "- B -- A leans on them" in out, out


def test_roster_excludes_the_target_and_the_unprofiled():
    out = profiles.render_profile(FIX, "A", ["A", "B", "ZZZ"])
    assert "ZZZ" not in out
    assert out.count("- A --") == 0


def test_roster_is_capped():
    big = {"T": {"seat": "seat T"}}
    big.update({f"P{i}": {"seat": f"seat {i}"} for i in range(20)})
    out = profiles.render_profile(big, "T", [f"P{i}" for i in range(20)], roster_limit=3)
    assert out.count("\n- ") == 3


def test_unknown_target_renders_nothing_rather_than_raising():
    assert profiles.render_profile(FIX, "Nobody", ["A"]) == ""
    assert profiles.render_profile(None, "A", ["B"]) == ""


def test_corpus_without_profiles_loads_as_none():
    assert profiles.load_profiles("atla") is None


# --------------------------------------------------------------------------
# the shipped fixture
# --------------------------------------------------------------------------

def test_bloomfield_profiles_are_stamped_fabricated():
    p = profiles.load_profiles("bloomfield")
    assert p, "bloomfield profiles missing"
    recs = profiles.people(p)
    for tok, rec in recs.items():
        assert rec.get("source") == "fabricated", f"{tok} not stamped"


def test_bloomfield_covers_the_traced_targets():
    recs = profiles.people(profiles.load_profiles("bloomfield"))
    for tok in ("Wolf", "Lyubovsky", "Rovani"):
        assert tok in recs
        for key in ("seat", "stake", "pressure"):
            assert recs[tok].get(key), f"{tok} missing {key}"


def test_profile_tokens_match_the_speaker_tokens():
    """A token that is not a real speaker is a profile that can never be used."""
    with open(os.path.join(profiles.DATA_DIR, "bloomfield_speakers.json"),
              encoding="utf-8") as fh:
        speakers = set(json.load(fh))
    extra = set(profiles.people(profiles.load_profiles("bloomfield"))) - speakers
    assert not extra, f"profiles for non-speakers: {sorted(extra)}"


def test_rendered_profile_is_not_thread_bound():
    """The prior must transport to a different week.

    restatement.BOUND is the repo's own test for a clause that could only be
    uttered in this particular conversation. A prior that trips it is seeding
    the exact failure this POC exists to measure -- the commitments would score
    as motives while being handed their thread-boundness by the prompt.
    """
    import restatement
    p = profiles.load_profiles("bloomfield")
    roster = ["Wolf", "Lyubovsky", "Rovani", "McLafferty"]
    for tok in ("Wolf", "Lyubovsky", "Rovani"):
        block = profiles.render_profile(p, tok, roster)
        for line in block.split("\n"):
            # Names are unavoidable in a roster and in RELATIONS; the rest of
            # the pattern (quoted categories, versions, artefacts, numbers) is
            # what must not appear.
            stripped = restatement.BOUND.sub(
                lambda m: "" if m.group(0) in roster else m.group(0), line)
            hit = restatement.BOUND.search(stripped)
            assert not hit, f"{tok}: thread-bound prior text {hit.group(0)!r} in {line!r}"


# --------------------------------------------------------------------------

def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0



# --------------------------------------------------------------------------
# audit_profile: the order-swap control
#
# Appended after main() deliberately -- main() collects from globals() at call
# time, so these are picked up.
# --------------------------------------------------------------------------

class _H2HStub:
    """Answers with a fixed policy so the swap logic can be checked exactly.

    `policy` is called with (first_clause, second_clause) and returns the letter
    the judge would say.
    """
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def interact(self, prompt, **kw):
        import re as _re
        a = _re.search(r"<account A[^>]*>\n(.*?)\n</account A>", prompt, _re.S).group(1)
        b = _re.search(r"<account B[^>]*>\n(.*?)\n</account B>", prompt, _re.S).group(1)
        self.calls.append((a, b))
        return f"WINNER: {self.policy(a, b)}\nWHY: stub"


def test_head_to_head_judges_both_orders():
    import audit_profile
    m = _H2HStub(lambda a, b: "A")   # always picks whatever is printed first
    w, flipped, _ = audit_profile.head_to_head(m, "t", "X", "profile clause", "other clause")
    assert len(m.calls) == 2, "did not judge both orders"
    # position-only judge: forward says profile, reverse says other -> disagree
    assert flipped is True
    assert w == "TIE", "a pure position bias was scored as a win"


def test_head_to_head_content_winner_survives_the_swap():
    import audit_profile
    m = _H2HStub(lambda a, b: "A" if a == "good" else "B")
    w, flipped, _ = audit_profile.head_to_head(m, "t", "X", "good", "bad")
    assert w == "A" and flipped is False
    # and the same content wins when it is the SECOND argument
    w2, flipped2, _ = audit_profile.head_to_head(m, "t", "X", "bad", "good")
    assert w2 == "B" and flipped2 is False


def test_head_to_head_tie_is_reported_as_tie():
    import audit_profile
    m = _H2HStub(lambda a, b: "TIE")
    w, flipped, _ = audit_profile.head_to_head(m, "t", "X", "p", "o")
    assert w == "TIE" and flipped is False


def test_unparseable_verdict_does_not_crash():
    import audit_profile
    class _Junk:
        def interact(self, prompt, **kw):
            return "the model rambled and never said WINNER"
    w, flipped, _ = audit_profile.head_to_head(_Junk(), "t", "X", "p", "o")
    assert w == "TIE"


def test_earned_rate_is_none_when_nothing_was_judged():
    """0.000 would read as a measured failure rather than a missing
    measurement, and the two must not print the same."""
    import audit_profile
    ev = [{"class": "hidden"}, {"class": "noise"}]
    assert audit_profile.rates(ev)["earned_rate"] is None
    ev[0]["judge"] = "NOT STATED"
    r = audit_profile.rates(ev)
    assert r["earned_rate"] == 0.5 and r["earned"] == 1

def test_single_seed_has_no_floor():
    """A floor of 0.0 would make every gap 'significant'. One seed measures
    nothing, and the report has to say so rather than imply a result."""
    import audit_profile
    assert audit_profile.spread([0.5]) is None
    assert audit_profile.spread([]) is None
    assert abs(audit_profile.spread([0.4, 0.6]) - 0.2) < 1e-9


def test_profile_echo_catches_parroting():
    """The judge is blind to this by construction: profile content is not in
    the transcript, so reading the prior back scores NOT STATED."""
    import audit_profile
    prior = "SEAT: runs the customer-facing side STAKE: keeping accounts confident"
    parrot = audit_profile.profile_echo("keeping accounts confident", prior)
    independent = audit_profile.profile_echo("prevent mislabeling of fruit", prior)
    assert parrot > 0.9, parrot
    assert independent == 0.0, independent
    # degenerate inputs must not raise
    assert audit_profile.profile_echo("", prior) == 0.0
    assert audit_profile.profile_echo("anything", "") == 0.0


# --------------------------------------------------------------------------
# the scramble control
# --------------------------------------------------------------------------

def test_scramble_is_a_derangement():
    """Nobody may receive their own record, or the control is not a control."""
    import audit_profile
    for person, given in audit_profile.SCRAMBLE.items():
        assert person != given, f"{person} was handed their own profile"
    # and it is a permutation of the same set, so every record is used once
    assert sorted(audit_profile.SCRAMBLE) == sorted(audit_profile.SCRAMBLE.values())


def test_scramble_hands_over_the_other_persons_record():
    p = profiles.load_profiles("bloomfield")
    roster = ["Wolf", "Lyubovsky", "Rovani", "McLafferty"]
    wrong = profiles.render_profile(p, "Lyubovsky", roster, exclude={"Wolf"})
    right = profiles.render_profile(p, "Wolf", roster)
    assert wrong != right
    assert "senior engineer on the vision side" in wrong
    assert "customer-facing side of deployments" not in wrong


def test_scramble_scrubs_the_traced_agent_from_the_roster():
    """A record that names the person it is supposedly describing is a tell."""
    p = profiles.load_profiles("bloomfield")
    roster = ["Wolf", "Lyubovsky", "Rovani", "McLafferty"]
    out = profiles.render_profile(p, "Lyubovsky", roster, exclude={"Wolf"})
    assert "Wolf" not in out, out


def test_scramble_meta_flags_itself():
    p = profiles.load_profiles("bloomfield")
    m = profiles.profile_meta(p, "Lyubovsky", ["Wolf", "Rovani"], traced="Wolf")
    assert m["profile_scrambled"] is True
    assert m["profile_target"] == "Lyubovsky"
    assert "Wolf" not in m["profile_roster"]
    straight = profiles.profile_meta(p, "Wolf", ["Lyubovsky"], traced="Wolf")
    assert straight["profile_scrambled"] is False


def test_tracer_renders_the_scrambled_record():
    t = _seed(character_profile=True, profiles=profiles.load_profiles("bloomfield"),
              profile_roster=["Wolf", "Lyubovsky", "Rovani"], profile_as="Lyubovsky")
    t.trace_base_header = "regarding [target agent]'s thoughts."
    tracer.Tracer.set_tracer_variables(
        t, {"question": "q", "context": "c", "target_agent": "Wolf"})
    assert t._character_profile
    assert "senior engineer on the vision side" in t._character_profile
    assert "customer-facing side of deployments" not in t._character_profile


def test_pecho_matrix_separates_supplied_from_true():
    """The whole scramble reads off this table, so it has to point the right
    way on data where the answer is known."""
    import audit_profile
    bundles = {
        ("profile", "Wolf", 0): {"events": [{"clause": "keep accounts confident"}],
                                 "meta": {}},
        ("scram", "Wolf", 0): {"events": [{"clause": "definitions precise enough"}],
                               "meta": {}},
    }
    ptexts = {"Wolf": "keep accounts confident about delivery",
              "Lyubovsky": "definitions precise enough that two people agree"}
    m = audit_profile.pecho_matrix(bundles, ["Wolf", "Lyubovsky"],
                                   ["profile", "scram"], ptexts)
    # a correct record is echoed by the arm that was given it
    assert m[("profile", "Wolf")]["Wolf"] > m[("profile", "Wolf")]["Lyubovsky"]
    # and a WRONG record is echoed by the scramble arm -- the steering signal
    assert m[("scram", "Wolf")]["Lyubovsky"] > m[("scram", "Wolf")]["Wolf"]


if __name__ == "__main__":
    sys.exit(main())
