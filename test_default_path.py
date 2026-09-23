"""THE DEFAULT PATH IS SACRED. Golden hashes of every prompt v2 could disturb.

PREREG standing rule 4: with every new flag off, every prompt must be
byte-identical to what it was before. This is not stylistic. All 100+ logged
findings, `baseline_metrics.json` and the whole of PRECHECKS were measured
against these exact strings, so a one-character drift on the default path
invalidates more evidence than any v2 feature can generate.

test_methods.py already asserts CONTENT on the seeding prompt ("no METHOD text",
"Generate a numbered list of 4 hypotheses"). Content assertions catch a feature
leaking in; they do not catch a stray space, a reordered clause or a changed
f-string. These hashes catch those.

WHAT THESE GUARD AGAINST SPECIFICALLY. v2 adds a portfolio field to
HypothesisV3, makes `anchor` a derived view of the priority-1 motive, and (in
Task 4) factors `_mint_commitments` out of `perturb_anchored`. Each of those
touches a code path that builds a prompt. The refactor is only safe if the
string it produces is identical, and "I read it carefully" is not a check.

THE HASHES WERE TAKEN WITH tracer.py UNMODIFIED. `git diff` showed hypothesis.py
as the only changed file at the moment they were generated, so they record the
pre-v2 wording rather than whatever v2 happened to produce.

REGENERATING THEM IS A DELIBERATE ACT. If a hash fails, the default path moved.
Either revert, or -- if the change is intended and the evidence it invalidates
is understood -- update the constant AND say so in PREREG, because every number
measured against the old prompt is now measured against something else.
"""
import hashlib
import types

import numpy as np

import tracer
from hypothesis import HypothesesSetV3


GOLDEN = {
    'rank_scorer_system':  '927f671a2643c3b5',
    'rank_scorer_user':    '5e91aab1433207ff',
    'standard_block':      'aa387b6c0d4702f0',
    'seeding_user':        '0b433e2a1802170e',
    'seeding_system':      'e3b0c44298fc1c14',   # sha256("") -- seeding sends none
    'split_system':        '4901b50b1499e9a9',
    'split_user':          '41ce49f8a4820689',
    'perturb_system':      '860ddeae0142773d',
    'perturb_user':        'a318fad044778df0',
}


def _h(s: str) -> str:
    return hashlib.sha256((s or '').encode('utf-8')).hexdigest()[:16]


def _assert(name: str, text: str):
    got = _h(text)
    assert got == GOLDEN[name], (
        f"DEFAULT PATH MOVED: {name} hashes {got}, expected {GOLDEN[name]}.\n"
        "Every logged finding was measured against the previous string. Revert, "
        "or update GOLDEN and record the change in PREREG.")


class _Stub:
    """Records prompts and returns a well-formed COMMITMENT/BELIEF/STANDARD list.

    The reply has to parse, or the operator bails before reaching the code whose
    prompt is being pinned and the test passes by never running.
    """
    _REPLY = ("1. COMMITMENT: aim one | BELIEF: b | STANDARD: s\n"
              "2. COMMITMENT: aim two | BELIEF: b | STANDARD: s")

    def __init__(self):
        self.prompts = []
        self.systems = []

    def interact(self, prompt, **kw):
        self.prompts.append(prompt)
        self.systems.append(kw.get('system_prompt'))
        return self._REPLY

    def batch_interact(self, prompts, **kw):
        sp = kw.get('system_prompts')
        out = []
        for i, p in enumerate(prompts):
            self.prompts.append(p)
            self.systems.append(sp[i] if isinstance(sp, list) else sp)
            out.append(self._REPLY)
        return out


class _SeedStub(_Stub):
    def interact(self, prompt, **kw):
        self.prompts.append(prompt)
        self.systems.append(kw.get('system_prompt'))
        return "\n".join(f"{i+1}. Katara believes thing {i}" for i in range(8))


def _slate(n=4):
    s = HypothesesSetV3('Katara', ['c'], [{}],
                        [f'account {i}' for i in range(n)],
                        np.array([1.0 / n] * n),
                        anchors=[f'aim {i}' for i in range(n)])
    for i, h in enumerate(s.hypotheses):
        h._likelihood_rank = n - i          # worst-ranked heavy particle is split-eligible
    return s


# -- module-level builders: pure, and public API to the offline audits --------

def test_rank_scorer_prompts_unchanged():
    """audit_vacuity.py parses <observed next action> out of a LOGGED prompt and
    rebuilds the rest verbatim, so this wording is a contract with the audit as
    well as with the model."""
    sysp, usr = tracer.rank_scorer_prompts('Katara', 3, 'CTX', 'HYPS', 'ACTION')
    _assert('rank_scorer_system', sysp)
    _assert('rank_scorer_user', usr)


def test_standard_block_unchanged():
    _assert('standard_block', tracer.standard_block('Katara', None, None, settles=''))


# -- operator prompts: captured through a stub model -------------------------

def test_seeding_prompt_unchanged():
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(
        n_hypotheses=4, methods=None, goal_seeding=True, infer_motive=False,
        extract_anchors_flag=False, use_anchor=False, dataset='musing',
        use_cot=False, input_is_chat=True, target_perceptions='sight',
        use_helper_llm=False)
    t.target_agent = 'Katara'
    t.assumption = ''
    t.tracer_model = _SeedStub()
    t._last_standards = []
    tracer.Tracer.initialize(t, {'state': 'ctx', 'action': 'Katara speaks.'},
                             {'state': '', 'action': ''})
    assert len(t.tracer_model.prompts) == 1, "default seeding is ONE pooled call"
    _assert('seeding_user', t.tracer_model.prompts[0])
    _assert('seeding_system', t.tracer_model.systems[0] or '')


def test_split_prompt_unchanged():
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(
        n_hypotheses=4, methods=None, split_weight_quantile=0.20, split_children=2,
        protect_leader=1, merge_percentile=95.0, merge_floor=0.40,
        standard_prompt=None, merge_anchors=False, population_cap=6,
        legacy_form=False)
    t.target_agent = 'Katara'
    t.tracer_model = _Stub()
    t._profile_settles = ''
    t._accum = {}
    t._retired = {}
    t._revival_counts = {}
    t._step_idx = 1
    tracer.Tracer.split_and_merge(t, _slate(), 'CTXSTR')
    assert t.tracer_model.prompts, "split never reached the model"
    _assert('split_system', t.tracer_model.systems[0] or '')
    _assert('split_user', t.tracer_model.prompts[0])


def test_perturb_mint_prompt_unchanged():
    """Task 4 factors _mint_commitments out of perturb_anchored. That refactor is
    only legal if this hash is unchanged afterwards."""
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(
        n_hypotheses=4, methods=None, standard_prompt=None, revive_retired=False,
        legacy_form=False, rebirth_at_fair_share=False, infer_motive=False)
    t.target_agent = 'Katara'
    t.tracer_model = _Stub()
    t._profile_settles = ''
    t._accum = {}
    t._retired = {}
    t._revival_counts = {}
    t._step_idx = 1
    tracer.Tracer.perturb_anchored(t, _slate(), [3], 'CTXSTR', None, 'stagnation')
    assert t.tracer_model.prompts, "perturb never reached the model"
    _assert('perturb_system', t.tracer_model.systems[0] or '')
    _assert('perturb_user', t.tracer_model.prompts[0])


# -- the v2 code must be inert when unused -----------------------------------

def test_portfolio_is_absent_on_the_default_path():
    s = _slate()
    assert all(h.portfolio is None for h in s.hypotheses)
    assert all(h.motives is None for h in s.hypotheses)
    assert not any(h.pinned for h in s.hypotheses)
    assert not s.any_portfolio


def test_default_dump_carries_no_portfolio_key():
    """read_steps filters unknown keys against __annotations__, so a stray key
    would rehydrate silently -- but a default-path dump must still be the shape
    every logged finding was written against."""
    s = _slate()
    assert 'portfolios' not in s.dump()


def test_update_anchor_still_works_without_a_portfolio():
    """The raise-guard must fire ONLY on portfolio particles. If it fired
    generally it would break split_and_merge and revive_retired on the default
    path, which the prompt hashes above would not catch."""
    s = _slate()
    h = s.hypotheses[0]
    before = h.root_id
    h.update_anchor('a different aim', revision=True)
    assert h.anchor == 'a different aim'
    assert h.root_id != before, "anchor == root still holds on the default path"


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
