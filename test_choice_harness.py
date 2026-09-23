"""Offline tests for the choice-point harness. No LLM calls, no API key.

What they protect, each from a way this harness could report a result it has
not got:

  1. The arms differ in ONE thing. `obvious` and a hypothesis arm must share
     the transcript, the situation and the option order exactly; if the motive
     block is not the only difference, the lift is measuring formatting.
  2. The blind arm is blind. If the person, the situation or the transcript
     leaks into it, the tripwire stops being one.
  3. `role` is not `habit` wearing a hat. It must exclude the person's own
     earlier choices, or the two baselines rise together and neither controls
     the other.
  4. The statistics are the paired ones. McNemar on hand-worked counts, BH
     monotone, permutation agreeing with the exact test on a clean case.
  5. Under a hash model every arm sits at chance and the verdict is NO EFFECT.
     A pipeline that finds an effect in noise finds one anywhere.
"""
import json
import os
import shutil
import sys
import tempfile

import choice_forecast as F
import choice_motives as MO
import choice_points as CP
from choice_llm import FakeModel, extract_json


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _cp(i, person, side, actual="HOLD", acts=("HOLD", "CONCEDE", "ESCALATE")):
    return {
        "cp_id": f"c:ch:{person}:{i}", "corpus": "c", "channel": "ch",
        "person": person, "at_turn": i, "set_id": f"c-{i:04d}", "date": f"2024-01-{i:02d}",
        "title": "t", "side": side,
        "situation": "a threshold is disputed", "why_real": "two options open",
        "alternatives": [{"action": a, "gist": f"gist {a}", "gives_up": f"cost {a}"}
                         for a in acts],
        "actual": actual, "n_alternatives": len(acts), "action_text": "what they wrote",
    }


class _Ctx:
    def prefix(self, cp, n=40):
        return f"[0] {cp['person']}: earlier turn"

    def turn(self, cp):
        return {}


# --------------------------------------------------------------------------

def test_only_the_motive_differs_between_obvious_and_hypothesis():
    cp, ctx = _cp(1, "Alice", "test"), _Ctx()
    plain = F.forecast_prompt(cp, ctx, None)
    withm = F.forecast_prompt(cp, ctx, {"motive": "protect the grower's read",
                                        "because": "gave up instrument fidelity twice"})
    assert len(withm) > len(plain)
    # everything before the belief block, and the options block, are identical
    head_p = plain.split("Options open to them:")[0]
    head_w = withm.split("\nWhat we believe about")[0]
    assert head_p.rstrip().startswith(head_w.rstrip()) or head_w.rstrip() in plain
    assert plain.split("Options open to them:")[1] == withm.split("Options open to them:")[1]


def test_option_order_is_fixed_and_shared():
    cp = _cp(1, "Alice", "test")
    a, _ = F.options_block(cp)
    b, _ = F.options_block(json.loads(json.dumps(cp)))
    assert [x["action"] for x in a] == [x["action"] for x in b]
    # and not simply the order the extractor wrote, which puts nothing first
    # by accident but would put the taken option at a fixed index if it did
    assert set(x["action"] for x in a) == {"HOLD", "CONCEDE", "ESCALATE"}


def test_blind_prompt_leaks_nothing():
    cp = _cp(1, "Alice", "test")
    b = F.blind_prompt(cp)
    for leak in ("Alice", cp["situation"], "Conversation", "#ch"):
        assert leak not in b, f"blind prompt leaked {leak!r}"


def test_parse_answer_only_accepts_offered_options():
    allowed = {"HOLD", "CONCEDE"}
    assert F.parse_answer("ANSWER: HOLD\nWHY: x", allowed) == "HOLD"
    assert F.parse_answer("ANSWER: [CONCEDE]", allowed) == "CONCEDE"
    assert F.parse_answer("ANSWER: re-raise", allowed) is None   # not offered
    assert F.parse_answer("I think they will concede here", allowed) == "CONCEDE"
    assert F.parse_answer("", allowed) is None


def test_role_excludes_the_persons_own_choices():
    dev = [_cp(1, "Alice", "dev", "HOLD"), _cp(2, "Alice", "dev", "HOLD"),
           _cp(3, "Alice", "dev", "HOLD"), _cp(4, "Bob", "dev", "CONCEDE")]
    chan, chan_person = F.role_table(dev)
    rest = F.others_in_channel(chan, chan_person, "c", "ch", "Alice")
    assert rest == F.collections.Counter({"CONCEDE": 1}), rest
    assert F.pick_from_counter(rest, ["HOLD", "CONCEDE"], "x") == "CONCEDE"


def test_role_subtracts_per_channel_not_per_person():
    # Alice acts in two channels. Removing her GLOBAL count from one channel
    # drives that channel negative and can hand `role` an action nobody in it
    # ever took. Subtraction has to be per channel.
    dev = [_cp(1, "Alice", "dev", "HOLD"), _cp(2, "Alice", "dev", "HOLD")]
    dev[1]["channel"] = "other"
    dev.append(_cp(3, "Bob", "dev", "HOLD"))
    chan, chan_person = F.role_table(dev)
    rest = F.others_in_channel(chan, chan_person, "c", "ch", "Alice")
    assert rest == F.collections.Counter({"HOLD": 1}), rest   # Bob survives
    assert all(v > 0 for v in rest.values())


def test_majority_is_a_constant_from_the_earlier_choices_only():
    dev = ([_cp(i, "Alice", "dev", "HOLD") for i in range(1, 8)]
           + [_cp(i, "Bob", "dev", "CONCEDE") for i in range(8, 11)])
    assert F.majority_action(dev) == "HOLD"
    # test-side labels must not reach it
    dev_plus_test = dev + [_cp(i, "Cara", "test", "TRADE") for i in range(11, 40)]
    assert F.majority_action([c for c in dev_plus_test if c["side"] == "dev"]) == "HOLD"


def test_habit_is_restricted_to_the_options_actually_offered():
    counter = F.collections.Counter({"IGNORE": 9, "HOLD": 1})
    assert F.pick_from_counter(counter, ["HOLD", "CONCEDE"], "x") == "HOLD"


def test_mcnemar_is_one_sided_and_exact():
    # 5 items the arm gets right and the reference does not, 0 the other way
    b, c, p = F.mcnemar([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert (b, c) == (5, 0) and abs(p - 1 / 32) < 1e-12
    # reversed: the arm is reliably WORSE and must not register
    b, c, p = F.mcnemar([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    assert (b, c) == (0, 5) and p == 1.0
    # no discordant pairs is no evidence
    assert F.mcnemar([1, 0], [1, 0]) == (0, 0, 1.0)


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = F.benjamini_hochberg([0.001, 0.02, 0.2, 0.9])
    assert all(0 <= x <= 1 for x in q)
    assert q == sorted(q), q
    assert q[0] >= 0.001 and q[-1] <= 1.0


def test_permutation_agrees_with_the_exact_test():
    arm, ref = [1] * 6, [0] * 6
    p = F.permutation_p(arm, ref, iters=4000, seed=1)
    assert p < 0.05, p
    assert F.permutation_p([1, 0, 1, 0], [1, 0, 1, 0], iters=2000) > 0.2


def test_clustered_permutation_is_conservative_where_pairs_repeat():
    # Three choice points, each carrying 13 hypotheses that all beat the
    # reference. Treated as 39 independent wins that is overwhelming; treated
    # honestly it is three coin flips, which cannot clear 0.05.
    clusters = {f"cp{i}": [(1, 0)] * 13 for i in range(3)}
    p_clustered = F.permutation_p_clustered(clusters, iters=8000, seed=3)
    flat_arm = [1] * 39
    flat_ref = [0] * 39
    _, _, p_mcnemar = F.mcnemar(flat_arm, flat_ref)
    assert p_mcnemar < 1e-10, p_mcnemar
    assert p_clustered > 0.05, p_clustered
    # and with enough independent choice points it still detects a real effect
    many = {f"cp{i}": [(1, 0)] * 13 for i in range(10)}
    assert F.permutation_p_clustered(many, iters=8000, seed=3) < 0.05


def test_clustered_permutation_sees_nothing_in_a_tie():
    clusters = {f"cp{i}": [(1, 1), (0, 0)] for i in range(12)}
    assert F.permutation_p_clustered(clusters, iters=4000, seed=1) > 0.2


def test_extract_json_prefers_whichever_bracket_comes_first():
    assert extract_json('[{"a":1},{"b":2}]') == [{"a": 1}, {"b": 2}]
    assert extract_json('```json\n{"a":[1,2]}\n```') == {"a": [1, 2]}
    assert extract_json("no json here") is None


def test_validate_rejects_the_ways_extraction_goes_wrong():
    chunk = [{"speaker": "Alice", "text": "x", "set_id": "c-1", "date": "d", "title": "t"},
             {"speaker": "Bob", "text": "y", "set_id": "c-1", "date": "d", "title": "t"}]
    people = {"Alice", "Bob"}
    good = {"person": "Alice", "at_turn": 0, "situation": "s", "why_real": "w",
            "alternatives": [{"action": a, "gist": "g", "gives_up": "u"}
                             for a in ("HOLD", "CONCEDE", "DROP")],
            "actual": "HOLD"}
    assert CP.validate(good, chunk, 0, people)[0] is not None
    for mutate, expect in (
            (lambda d: d.update(person="Nobody"), "person not a participant"),
            (lambda d: d.update(at_turn=1), "at_turn is not that person's turn"),
            (lambda d: d.update(at_turn=99), "at_turn outside the window"),
            (lambda d: d.update(actual="TRADE"), "actual not among the alternatives"),
            (lambda d: d.update(alternatives=good["alternatives"][:2]),
             "fewer than 3 alternatives"),
    ):
        d = json.loads(json.dumps(good))
        mutate(d)
        cp, why = CP.validate(d, chunk, 0, people)
        assert cp is None and why == expect, (why, expect)


def test_duplicate_action_labels_are_rejected():
    chunk = [{"speaker": "Alice", "text": "x", "set_id": "c-1", "date": "d", "title": "t"}]
    d = {"person": "Alice", "at_turn": 0, "situation": "s", "why_real": "w",
         "alternatives": [{"action": "HOLD", "gist": "a", "gives_up": "b"},
                          {"action": "HOLD", "gist": "c", "gives_up": "d"},
                          {"action": "DROP", "gist": "e", "gives_up": "f"}],
         "actual": "HOLD"}
    assert CP.validate(d, chunk, 0, {"Alice"})[1] == "bad or duplicate action label"


def test_revealed_preference_ledger_hides_the_prose():
    pts = [_cp(1, "Alice", "dev", "HOLD")]
    shown = MO.ledger(pts, with_text=False)
    assert "gist HOLD" not in shown and "threshold is disputed" not in shown
    assert "cost HOLD" in shown and "TOOK" in shown
    # and the full ledger does show it, so the two arms differ only in that
    full = MO.ledger(pts, with_text=True)
    assert "gist HOLD" in full and "threshold is disputed" in full


def test_pipeline_under_a_hash_model_finds_no_effect():
    tmp = tempfile.mkdtemp(prefix="choice-")
    try:
        pts = ([_cp(i, "Alice", "dev", "HOLD") for i in range(1, 7)]
               + [_cp(i, "Bob", "dev", "CONCEDE") for i in range(7, 13)]
               + [_cp(i, "Alice", "test", "HOLD") for i in range(13, 19)]
               + [_cp(i, "Bob", "test", "ESCALATE") for i in range(19, 25)])
        store = os.path.join(tmp, "c.jsonl")
        with open(store, "w", encoding="utf-8") as fh:
            for cp in pts:
                fh.write(json.dumps(cp) + "\n")
        motives = [{"hyp_id": f"c:{p}:{m}", "corpus": "c", "person": p, "method": m,
                    "family": "record", "motive": f"{m} aim", "because": "b",
                    "n_dev_choices": 6, "n_test_choices": 6}
                   for p in ("Alice", "Bob") for m in ("anomaly", "revealed")]
        mpath = os.path.join(tmp, "_motives.jsonl")
        with open(mpath, "w", encoding="utf-8") as fh:
            for h in motives:
                fh.write(json.dumps(h) + "\n")

        old_load, old_motives, old_ctx = CP.load, MO.load_motives, F.Context
        F.load_points = lambda c: [json.loads(l) for l in open(store)]
        F.load_motives = lambda: [json.loads(l) for l in open(mpath)]
        F.Context = lambda corpora: _Ctx()
        try:
            res = F.run(["c"], FakeModel(), iters=2000, out=os.path.join(tmp, "o.json"))
        finally:
            F.load_points, F.load_motives, F.Context = old_load, old_motives, old_ctx

        assert res["n_test_choices"] == 12
        assert not res["tripwire"]["leaks"], res["tripwire"]
        assert res["verdict"].startswith("NO EFFECT"), res["verdict"]
        assert res["pooled"]["mcnemar"]["p"] > 0.05
        # the surprise list must not be sold as a finding under noise
        assert res["surprises"]["caught_by_a_credible_hypothesis"] == 0
    finally:
        shutil.rmtree(tmp)


def test_swap_pairing_never_points_at_the_same_person():
    test = [_cp(i, p, "test") for p in ("Alice", "Bob", "Cara") for i in range(3)]
    _, swap = F.build_jobs(test, _Ctx(), [], 40)
    assert len(swap) == 3
    for src, dst in swap.items():
        assert src != dst and src[0] == dst[0]


if __name__ == "__main__":
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
