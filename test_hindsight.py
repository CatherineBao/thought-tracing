"""The cut rule reads only the input, and nothing after a choice reaches its prefix.

PREREG registers this as an obligation rather than a courtesy, because CaSiNo's
whole case rests on one number produced by the cut: the silent subgroup is 0.182
under the full cut and 0.629 under the early cut. If the cut could see the
submitted deal, that 0.629 is not selection on the input and the subgroup is not
a legitimate test.

Three separate things are checked, because they fail differently:

  1. the CUT RULE is a function of the prose alone -- perturbing `task_data`
     must not move it;
  2. the PREFIX stops before the cut -- no turn at or after it survives;
  3. ANSWER-KEY content never appears in a prefix, in particular `value2reason`,
     which is keyed High/Medium/Low and would hand over the ground-truth order.

No LLM calls.
"""
import copy
import json
import os

import casino_ingest as C

HERE = os.path.dirname(os.path.abspath(__file__))


def _dialogue(chat, info=None):
    default = {
        "mturk_agent_1": {
            "value2issue": {"High": "Firewood", "Medium": "Food", "Low": "Water"},
            "value2reason": {"High": "we brought a wood stove",
                             "Medium": "large group",
                             "Low": "creek nearby"},
            "outcomes": {"points_scored": 19},
        },
        "mturk_agent_2": {
            "value2issue": {"High": "Water", "Medium": "Food", "Low": "Firewood"},
            "value2reason": {"High": "no creek", "Medium": "kids", "Low": "gas stove"},
            "outcomes": {"points_scored": 12},
        },
    }
    return {"dialogue_id": 7, "chat_logs": chat,
            "participant_info": info or default}


def _chat():
    return [
        {"id": "mturk_agent_1", "text": "Hello! how is your trip going", "task_data": {}},
        {"id": "mturk_agent_2", "text": "Good thanks. What do you need most?", "task_data": {}},
        {"id": "mturk_agent_1", "text": "I could really use 2 firewood", "task_data": {}},
        {"id": "mturk_agent_2", "text": "That works, I mainly want water", "task_data": {}},
        {"id": "mturk_agent_1", "text": "Submit-Deal",
         "task_data": {"issue2youget": {"Firewood": "3", "Food": "1", "Water": "0"},
                       "issue2theyget": {"Firewood": "0", "Food": "2", "Water": "3"}}},
        {"id": "mturk_agent_2", "text": "Accept-Deal", "task_data": {}},
    ]


# -- 1. the cut rule is a function of the prose alone ----------------------

def test_cut_ignores_task_data_entirely():
    chat = _chat()
    base = C.first_allocation_turn(chat)
    # rewrite the submitted allocation to its exact opposite
    flipped = copy.deepcopy(chat)
    flipped[4]["task_data"]["issue2youget"] = {"Firewood": "0", "Food": "0", "Water": "3"}
    assert C.first_allocation_turn(flipped) == base, \
        "the cut moved when only task_data changed -- it is reading the outcome"
    # and removing task_data outright must not move it either
    stripped = copy.deepcopy(chat)
    stripped[4]["task_data"] = {}
    assert C.first_allocation_turn(stripped) == base


def test_cut_lands_on_the_first_prose_allocation_not_the_submission():
    chat = _chat()
    at = C.first_allocation_turn(chat)
    assert at == 2, f"expected the '2 firewood' turn, got {at}"
    assert at < 4, "the cut must precede the Submit-Deal, or the prefix is the whole negotiation"


def test_cut_falls_back_to_the_submission_when_no_prose_allocation_exists():
    chat = [c for c in _chat() if "firewood" not in (c["text"] or "").lower()]
    at = C.first_allocation_turn(chat)
    assert at is not None and chat[at]["text"] == "Submit-Deal"


# -- 2. the prefix stops before the cut ------------------------------------

def test_prefix_contains_no_turn_at_or_after_the_cut():
    _, points, _, _ = C.convert([_dialogue(_chat())])
    assert points, "no choice point produced"
    for p in points:
        texts = [t["text"] for t in p["prefix_turns"]]
        assert "That works, I mainly want water" not in texts, \
            "a turn after the cut reached the prefix"
        assert len(p["prefix_turns"]) <= p["at_turn"]


def test_no_special_token_reaches_a_prefix():
    """Submit-Deal / Accept-Deal in a prefix would announce that the choice has
    already happened."""
    _, points, _, _ = C.convert([_dialogue(_chat())])
    for p in points:
        for t in p["prefix_turns"]:
            assert t["text"] not in C.SPECIAL, t


def test_silent_flag_is_computed_from_the_prefix_only():
    """The 0.629 figure depends on this. A priority stated AFTER the cut must
    still leave the person silent."""
    chat = _chat()
    chat.insert(3, {"id": "mturk_agent_1",
                    "text": "honestly I need firewood most of all", "task_data": {}})
    _, points, _, _ = C.convert([_dialogue(chat)])
    a = [p for p in points if p["person"] == "A"][0]
    assert not any("most of all" in t["text"] for t in a["prefix_turns"]), \
        "a post-cut statement leaked into the prefix"


# -- 3. answer-key content never appears in a prefix ------------------------

def test_value2reason_never_reaches_a_prefix():
    """It is keyed High/Medium/Low, so putting it in a prompt hands over the
    ground-truth priority order outright."""
    d = _dialogue(_chat())
    _, points, keys, _ = C.convert([d])
    secrets = [r for pp in keys["casino-0007"]["participants"].values()
               for r in pp["value2reason"].values()]
    assert secrets, "fixture has no reasons to leak"
    for p in points:
        blob = " ".join(t["text"] for t in p["prefix_turns"]).lower()
        for s in secrets:
            assert s.lower() not in blob, f"answer-key reason leaked: {s!r}"


def test_answer_key_is_written_outside_the_corpus_tree():
    """A prompt builder walks data/musing. The key must not be reachable there."""
    assert "musing" not in C.KEY_DIR.replace(os.sep, "/").split("/data/")[-1].split("/")[0]
    for name in ("casino_dialogue.json", "casino_speakers.json", "casino_gaps.json"):
        path = os.path.join(C.MUSING, name)
        if not os.path.exists(path):
            continue
        blob = open(path, encoding="utf-8").read().lower()
        assert "value2issue" not in blob and "value2reason" not in blob, \
            f"{name} carries answer-key fields"


def test_ties_are_labels_and_nothing_is_discarded():
    """Excluding ties would be selection on the outcome and would delete a third
    of the corpus; breaking them invents a preference the proposal does not make."""
    assert "Firewood+Food" in C.LABELS and "Firewood+Food+Water" in C.LABELS
    assert "OTHER" in C.LABELS, "an unlisted actual action is scored, never dropped"
    assert C.claim_label({"Firewood": "2", "Food": "2", "Water": "1"}) == "Firewood+Food"
    assert C.claim_label({"Firewood": "1", "Food": "1", "Water": "1"}) == "Firewood+Food+Water"


# -- Avalon ----------------------------------------------------------------

import avalon_ingest as A


def _game():
    def msg(pl, t, q=1, tn=1):
        return {"player": pl, "msg": t, "mid": f"m{abs(hash(t))%9999}", "quest": q, "turn": tn}
    msgs = [msg("system", "game started!"),
            msg("player-1", "i'm good"),
            msg("system", "player-1 proposed a party: player-1, player-2"),
            msg("player-3", "i don't trust player-2"),
            msg("system", "party vote outcome: player-1: yes, player-2: yes, player-3: no, "
                          "player-4: yes, player-5: no, player-6: yes"),
            msg("system", "vote succeeded! initiating quest vote!"),
            msg("system", "quest succeeded!"),
            msg("system", "good won for now, but the assassin..."),
            msg("player-4", "player-1 was obviously merlin the whole time"),
            msg("system", "the assassin didn't find merlin, thus the forces of good win!")]
    return {"users": {str(i): {"name": f"player-{i}", "index": i, "role": r}
                      for i, r in enumerate(["merlin", "percival", "morgana",
                                             "assassin", "servant-1", "servant-2"], 1)},
            "messages": {str(i + 1): m for i, m in enumerate(msgs)},
            "beliefs": {"1": {"player": "player-6", "about_player": "player-4",
                              "belief": "morgana", "quest": 1, "turn": 2}},
            "persuasion": {"1": {"mid": msgs[1]["mid"], "persuasion": "assertion",
                                 "deception": "lie"}}}


def test_avalon_endgame_never_reaches_a_prefix_or_the_corpus():
    """The assassination phase and post-game chat discuss roles openly."""
    st_set, points, _, _ = A.convert_game(_game(), 0)
    assert "assassin didn" not in st_set["full_context"].lower()
    assert "forces of good" not in st_set["full_context"].lower()
    for p in points:
        blob = " ".join(t["text"] for t in p["prefix_turns"]).lower()
        assert "assassin didn" not in blob and "forces of good" not in blob
        assert "obviously merlin the whole time" not in blob,             "post-game role talk leaked into a prefix"


def test_avalon_prefix_stops_before_its_own_proposal():
    _, points, _, _ = A.convert_game(_game(), 0)
    inc = [p for p in points if p["kind"] == "include"]
    assert inc
    for p in inc:
        blob = " ".join(t["text"] for t in p["prefix_turns"])
        assert "proposed a party" not in blob, "the proposal being forecast is in its own prefix"
        assert "party vote outcome" not in blob, "its own vote outcome is in the prefix"


def test_avalon_vote_points_are_evidence_only():
    """They fail the gate at 0.708 as a scoring target. The filter may learn from
    them; the headline log-score may not read them."""
    _, points, _, _ = A.convert_game(_game(), 0)
    votes = [p for p in points if p["kind"] == "party_vote"]
    assert votes, "no vote points produced"
    assert all(p["scored"] is False for p in votes)
    assert all(p["scored"] is True for p in points if p["kind"] == "include")


def test_avalon_self_pairs_are_excluded():
    """Leaders include themselves in 0.899 of proposals, so those points are
    near-trivial."""
    _, points, _, _ = A.convert_game(_game(), 0)
    for p in points:
        if p["kind"] == "include":
            assert p["subject"] != p["person"], "a leader was asked about themselves"


def test_avalon_proposal_is_one_observation():
    """The leader must pick exactly k, so the pairs are tied. They share a
    batch_id and carry the m for the tempered (prod p)^(1/m)."""
    _, points, _, _ = A.convert_game(_game(), 0)
    inc = [p for p in points if p["kind"] == "include"]
    batches = collections.defaultdict(list)
    for p in inc:
        batches[p["batch_id"]].append(p)
    for bid, group in batches.items():
        assert len({p["batch_size"] for p in group}) == 1
        assert group[0]["batch_size"] == len(group), (bid, len(group))


def test_avalon_answer_key_fields_are_not_in_the_corpus_file():
    """Roles, beliefs and the deception self-labels give the game away."""
    _, _, key, _ = A.convert_game(_game(), 0)
    assert key["roles"]["P1"] == "merlin"
    assert key["message_labels"][0]["deception"] == "lie"
    path = os.path.join(A.MUSING, "avalon_dialogue.json")
    if os.path.exists(path):
        blob = open(path, encoding="utf-8").read().lower()
        for field in ("\"role\"", "value2issue", "\"deception\"", "about_player"):
            assert field not in blob, f"corpus file carries {field}"


def test_avalon_no_voter_sees_another_voter_on_the_same_proposal():
    """VOTES ARE SIMULTANEOUS ACROSS PLAYERS, not just within one.

    All six vote on a proposal at once, so player B's prefix for proposal p must
    not contain player A's vote on p. This is the same cross-player problem as
    Diplomacy's simultaneous orders, and the fixture below is the second-actor
    case that a nonce test over a single actor would never reach.
    """
    _, points, _, _ = A.convert_game(_game(), 0)
    votes = [p for p in points if p["kind"] == "party_vote"]
    assert len(votes) >= 2, "need at least two voters to test the cross-player case"
    by_batch = collections.defaultdict(list)
    for v in votes:
        by_batch[v["batch_id"]].append(v)
    for bid, group in by_batch.items():
        assert len(group) >= 2, bid
        # every voter on one proposal sees EXACTLY the same prefix ...
        blobs = {" ".join(t["text"] for t in v["prefix_turns"]) for v in group}
        assert len(blobs) == 1, f"{bid}: voters were given different prefixes"
        # ... and that prefix contains nobody's vote
        blob = blobs.pop().lower()
        assert "party vote outcome" not in blob, \
            "the vote outcome line -- which names every vote -- is in a voter's prefix"
        for v in group:
            assert f"{v['person'].lower()}: yes" not in blob
            assert f"{v['person'].lower()}: no" not in blob


def test_entropy_guard_flags_a_varying_label_space():
    """A normalised entropy above 1 is impossible; it means the check was
    applied to a label space that varies across points. Avalon's joint subset
    choice offers 6 or 11 subsets per proposal but 41 distinct labels pooled."""
    import label_gate as lg
    varying = {"mean_alternatives": 10.22, "label_counts": {i: 1 for i in range(41)}}
    fixed = {"mean_alternatives": 8, "label_counts": {i: 1 for i in range(7)}}
    assert not lg.entropy_is_interpretable(varying)
    assert lg.entropy_is_interpretable(fixed)
    res = lg.evaluate({"majority_share": 0.10, "entropy_frac": 1.52,
                       "loo_person_constant": 0.07, "mean_alternatives": 10.22,
                       "label_counts": {i: 1 for i in range(41)}, "points": 148,
                       "people": 101})
    ent = [c for c in res["checks"] if c["gate"] == "entropy_frac"][0]
    assert "N/A" in (ent.get("note") or ""), "the guard must SAY it is not interpretable"
    assert res["pass"], "majority_share still carries the balance question"


import collections  # noqa: E402  (used by the batch test above)

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
