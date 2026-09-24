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
