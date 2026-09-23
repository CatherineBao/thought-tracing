"""Offline tests for the METHODS registry and the per-particle method label.

No LLM calls: the tracer's model is stubbed, so these run in CI and in a
sandbox. What they protect, in order of importance:

  1. THE DEFAULT PATH. With no method selected every generation prompt must be
     byte-identical to what it was before methods.py existed. FINDINGS.md's
     measurements are all against those prompts, so a one-character drift on
     the default path invalidates more evidence than any method can generate.
  2. The inheritance rule: method travels with the ANCHOR, not the lineage.
     Propagate, resample and merge inherit; seed, split and mint assign; revive
     restores. Getting this wrong does not crash anything -- it silently
     misattributes every finding the audit reports.
"""
import types
import numpy as np

from methods import METHODS, method_rule, parse_methods
from hypothesis import (HypothesisV3, HypothesesSetV3,
                        resample_hypotheses_with_other_info)
import tracer


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_every_method_has_a_family():
    from methods import FAMILIES, family_of
    for k, m in METHODS.items():
        assert m.family in FAMILIES, (k, m.family)
        assert family_of(k) == m.family
    assert family_of(None) == '(none)'
    assert family_of('nonsense') == '(none)'
    # the cut the grouping exists for: premortem and silence generate from a
    # state the record does not show, which is why both escaped the baseline's
    # scene-bound restatement
    assert METHODS['premortem'].family == METHODS['silence'].family == 'counterfactual'
    assert METHODS['anomaly'].family == METHODS['ach'].family == 'record'


def test_family_pooling_cannot_lower_novelty():
    # pooling gives a group MORE stems, so a clause one member found and
    # another did not is unique at family level; the pooled figure can only
    # rise or hold, never fall
    from audit_methods import pass1
    from methods import family_of
    ev = {
        (1, 'a'): {'clause': 'Avoid being powerless again', 'method': 'premortem',
                   'root_id': 'r1'},
        (2, 'b'): {'clause': 'Maintain her autonomy', 'method': 'silence',
                   'root_id': 'r2'},
    }
    by_m = pass1(dict(ev), set())
    by_f = pass1(dict(ev), set(), lambda e: family_of(e['method']))
    assert set(by_m) == {'premortem', 'silence'}
    assert set(by_f) == {'counterfactual'}
    assert by_f['counterfactual']['novelty'] >= max(
        by_m[m]['novelty'] for m in by_m)


def test_registry_is_the_twelve():
    assert set(METHODS) == {
        'anomaly', 'role', 'silence', 'devil', 'ach', 'assume',
        'premortem', 'presentation', 'crystal', 'backcast', 'signpost', 'analogy'}


def test_default_path_is_empty_at_every_site():
    for site in ('seed', 'perturb', 'split', 'standard'):
        assert method_rule(None, site, 'Katara') == ""
        assert method_rule("", site, 'Katara') == ""


def test_unknown_key_is_tolerated_at_runtime_but_rejected_at_parse():
    # tolerant where a run would otherwise die mid-trace...
    assert method_rule('nonsense', 'perturb', 'Katara') == ""
    # ...and loud where the typo is actually made
    try:
        parse_methods('anomly')
    except ValueError as e:
        assert 'anomly' in str(e)
    else:
        raise AssertionError("parse_methods accepted an unknown key")


def test_parse_methods_dedupes_order_preserving():
    # a repeat would silently weight that method double in the round-robin
    assert parse_methods('devil,anomaly,devil') == ['devil', 'anomaly']
    assert parse_methods('') is None
    assert parse_methods(None) is None


def test_target_name_is_interpolated_not_left_as_a_placeholder():
    for k in METHODS:
        for site in ('seed', 'perturb', 'split', 'standard'):
            frag = method_rule(k, site, 'Katara')
            assert '{t}' not in frag, f"{k}/{site} left an uninterpolated placeholder"


def test_axis_routing():
    base = tracer.standard_rule('Wolf', 'v1')
    for k, m in METHODS.items():
        got = tracer.standard_block('Wolf', 'v1', k)
        if m.axis == 'anchor':
            # an anchor-axis method must not shift the STANDARD register, or
            # the mode lexicon that bins it is measuring itself
            assert got == base, f"{k} leaked into the standard block"
        else:
            assert got.startswith(base) and len(got) > len(base), \
                f"{k} replaced the standard prompt instead of appending to it"


def test_standard_block_default_parity():
    for v in ('v1', 'v2', 'v3', 'v4', None):
        assert tracer.standard_block('Katara', v, None) == tracer.standard_rule('Katara', v)


def test_sole_standard_method():
    # the extractor makes ONE call over ALL texts, so it may only carry a
    # method's frame when that frame is unambiguous
    assert tracer.sole_standard_method(None) is None
    assert tracer.sole_standard_method(['anomaly', 'devil']) is None
    assert tracer.sole_standard_method(['anomaly', 'signpost']) == 'signpost'
    assert tracer.sole_standard_method(['signpost', 'crystal']) is None


def test_seed_fragments_carry_the_output_guard():
    # the seed site's contract is only "a numbered list", so a method that asks
    # for private reasoning will otherwise write that reasoning into the answer
    from methods import SEED_OUTPUT_GUARD
    guard = SEED_OUTPUT_GUARD.format(t='Katara')
    for k, m in METHODS.items():
        frag = method_rule(k, 'seed', 'Katara')
        if m.axis == 'standard':
            assert frag == "", f"{k} is standard-axis and must not frame seeding"
        else:
            assert frag.endswith(guard), f"{k} seed fragment lacks the output guard"


def test_a_label_is_only_applied_where_the_frame_was_applied():
    from methods import contributing_methods
    # A standard-axis method contributes no commitment-site fragment, so a
    # particle it "generated" was in fact generated by the default prompt.
    # Observed live before this filter covered the mint: signpost was credited
    # with the revive of "Forgive Yon Rha", on the strength of an empty string.
    for site in ('seed', 'perturb', 'split'):
        assert contributing_methods(['anomaly', 'silence', 'signpost'], site) == \
            ['anomaly', 'silence'], site
        assert contributing_methods(['signpost', 'crystal'], site) == [None], site
        assert contributing_methods(None, site) == [None], site
    # ...and at the standard site the routing is exactly inverted
    assert contributing_methods(['anomaly', 'signpost'], 'standard') == ['signpost']


def test_mint_chooser_never_picks_a_non_contributing_method():
    c = _Chooser(['anomaly', 'silence', 'signpost', 'crystal'])
    got = [c._mint_method('stagnation') for _ in range(6)]
    assert set(got) == {'silence'}, got
    # anomaly is anchor-axis but needs_action, so only the surprise path
    c2 = _Chooser(['anomaly', 'silence', 'signpost', 'crystal'])
    assert set(c2._mint_method('surprise') for _ in range(4)) == {'anomaly'}
    # a list with nothing eligible anywhere mints unframed and unlabelled,
    # rather than mislabelling
    assert _Chooser(['signpost', 'crystal'])._mint_method('surprise') is None


def test_emphasis_is_stripped_before_a_commitment_is_stored():
    # the anchor is the root's identity and is compared BY STRING, so the same
    # commitment with and without emphasis founded two roots and split its mass
    n = tracer.normalize_commitment
    assert n("**Forgive Yon Rha**") == "Forgive Yon Rha"
    assert n("  Forgive   Yon  Rha ") == "Forgive Yon Rha"
    assert n("Forgive Yon Rha") == "Forgive Yon Rha"
    assert n("") == "" and n(None) is None
    # and what the validator accepts is what gets stored
    ok, _ = tracer.valid_commitment("**Avenge her mother**")
    assert ok and n("**Avenge her mother**") == "Avenge her mother"


# --------------------------------------------------------------------------
# the inheritance rule
# --------------------------------------------------------------------------

def _set(n=4, methods=None, anchors=None):
    texts = [f"hypothesis {i}" for i in range(n)]
    anchors = anchors or [f"Aim {i}" for i in range(n)]
    return HypothesesSetV3('Katara', [], [], texts, np.ones(n) / n,
                           anchors=anchors, methods=methods)


def test_seeding_assigns_per_particle():
    s = _set(4, methods=['anomaly', 'silence', 'anomaly', 'silence'])
    assert s.methods == ['anomaly', 'silence', 'anomaly', 'silence']


def test_methods_padded_when_short():
    s = _set(4, methods=['anomaly'])
    assert s.methods == ['anomaly', None, None, None]


def test_propagation_inherits():
    parent = _set(2, methods=['devil', 'role'])
    child = HypothesesSetV3('Katara', [], [], ['a', 'b'], np.ones(2) / 2,
                            parent_hypotheses=parent.hypotheses)
    assert child.methods == ['devil', 'role']


def test_resample_copy_inherits():
    by_anchor = {'Aim 0': 'anomaly', 'Aim 1': 'silence',
                 'Aim 2': 'devil', 'Aim 3': 'role'}
    s = _set(4, methods=['anomaly', 'silence', 'devil', 'role'])
    s.weights = np.array([0.97, 0.01, 0.01, 0.01])
    s.weight_details = {'prompts': ['p'] * 4, 'reasonings': ['r'] * 4}
    out = resample_hypotheses_with_other_info(s, ess=1.0)
    # A duplicate is a COPY of a hypothesis, not a new one, so it keeps the
    # label of the commitment it copied -- resampling is not a generation
    # event. Checked against each survivor's own anchor rather than against a
    # predicted multiplicity, which is the resampler's business, not ours.
    for h in out.hypotheses:
        assert h.method == by_anchor[h.anchor], (h.anchor, h.method)


def test_new_anchor_takes_the_new_method():
    h = HypothesisV3('Katara', [], [], 't', 0.5, anchor='Old aim', method='anomaly')
    root_before = h.root_id
    h.update_anchor('New aim', revision=True, standard='a count', method='devil')
    assert h.method == 'devil'
    assert h.standard == 'a count'
    assert h.root_id != root_before, "a new anchor must found a new root"


def test_unchanged_anchor_keeps_the_method():
    h = HypothesisV3('Katara', [], [], 't', 0.5, anchor='Same aim', method='anomaly')
    h.update_anchor('Same aim', method='devil')
    assert h.method == 'anomaly', "an unchanged anchor is not a generation event"


def test_revive_without_a_method_does_not_erase_one():
    # revive_retired restores a commitment whose method it may not know.
    # Deliberately asymmetric with `standard`, which DOES clobber on None
    # because a stale settling condition is actively wrong.
    h = HypothesisV3('Katara', [], [], 't', 0.5, anchor='Old aim', method='silence')
    h.update_anchor('Revived aim', revision=True, standard=None, method=None)
    assert h.method == 'silence'
    assert h.standard is None


def test_method_is_dumped():
    s = _set(2, methods=['crystal', 'backcast'])
    d = s.dump()
    assert d['methods'] == ['crystal', 'backcast']
    # standard was missing from dump() entirely, which left the tracer dump and
    # the steps JSONL disagreeing about what a particle is
    assert 'standards' in d


def test_merge_survivor_keeps_its_own_method():
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(merge_percentile=95.0, n_hypotheses=4)
    # two near-identical texts so merge has something to absorb
    s = HypothesesSetV3('Katara', [], [],
                        ['she wants to protect her brother from the fire nation',
                         'she wants to protect her brother from the fire nation too',
                         'she wants to avenge her mother at any cost'],
                        np.array([0.4, 0.3, 0.3]),
                        anchors=['Protect her brother', 'Protect her brother',
                                 'Avenge her mother'],
                        methods=['anomaly', 'devil', 'silence'])
    out, merged, _ = tracer.Tracer.merge_similar(t, s)
    # a merge survivor is not a generation event: it keeps its own label, and
    # the absorbed particle's label goes with the particle
    for h in out.hypotheses:
        assert h.method is not None
    assert {h.anchor: h.method for h in out.hypotheses}['Avenge her mother'] == 'silence'


def test_population_cap_preserves_methods():
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(n_hypotheses=2)
    s = _set(6, methods=['anomaly', 'silence', 'devil', 'role', 'ach', 'assume'])
    s.weights = np.array([0.3, 0.25, 0.2, 0.15, 0.06, 0.04])
    out, cap, bound = tracer.Tracer.enforce_population_cap(t, s)
    by_anchor = {h.anchor: h.method for h in out.hypotheses}
    expected = dict(zip([f"Aim {i}" for i in range(6)],
                        ['anomaly', 'silence', 'devil', 'role', 'ach', 'assume']))
    for anchor, m in by_anchor.items():
        assert m == expected[anchor], (anchor, m)


def test_split_children_carry_the_parent_method():
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(
        split_children=2, split_weight_quantile=0.20, methods=['devil'],
        standard_prompt='v1', legacy_form=False, merge_percentile=95.0,
        n_hypotheses=8)

    class _M:
        def __init__(self): self.systems = []
        def batch_interact(self, prompts, system_prompts=None, **kw):
            self.systems = system_prompts
            return ["1. COMMITMENT: Shield Sokka | BELIEF: b1 | STANDARD: s1\n"
                    "2. COMMITMENT: Shield Aang | BELIEF: b2 | STANDARD: s2"] * len(prompts)
    t.tracer_model = _M()
    t.merge_similar = lambda h: (h, 0, [])

    s = _set(4, methods=['anomaly', 'silence', 'devil', 'role'])
    # make particle 0 the split candidate: top-quartile weight, worst rank
    s.weights = np.array([0.7, 0.1, 0.1, 0.1])
    for i, h in enumerate(s.hypotheses):
        h._likelihood_rank = 4 if i == 0 else 1
    out, info = tracer.Tracer.split_and_merge(t, s)
    if not info['split_fired']:
        return  # trigger did not select anything; the inheritance rule is
                # covered by the constructor tests either way
    kids = [h for h in out.hypotheses if h.anchor in ('Shield Sokka', 'Shield Aang')]
    assert kids, [h.anchor for h in out.hypotheses]
    for h in kids:
        assert h.method == 'anomaly', (h.anchor, h.method)
    # The split prompt carried the PARENT's frame, not the run's method list:
    # a child refines its parent's commitment, so it is generated under the
    # frame that produced that commitment.
    sysp = t.tracer_model.systems[0] or ""
    assert 'anomaly detection' in sysp, "split prompt lost the parent's frame"
    assert "devil's advocate" not in sysp, "split prompt used the run list, not the parent"


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------

def _merge_tracer(**over):
    t = tracer.Tracer.__new__(tracer.Tracer)
    cfg = dict(merge_percentile=95.0, merge_anchors=False,
               n_hypotheses=8, retired_cap=40)
    cfg.update(over)
    t.args = types.SimpleNamespace(**cfg)
    t._retired = {}
    return t


def test_merge_conserves_mass_when_one_particle_matches_two_survivors():
    # the weight loop used to re-test every survivor against every absorbed
    # particle, so a particle similar to TWO survivors was counted twice --
    # mass invented, then normalised away, tilting the posterior
    t = _merge_tracer()
    txt = "Katara wants to protect her brother from the fire nation soldiers"
    s = HypothesesSetV3('Katara', [], [], [txt, txt, txt, 'something wholly other'],
                        np.array([0.4, 0.3, 0.2, 0.1]),
                        anchors=['A', 'B', 'C', 'D'])
    out, merged, _ = tracer.Tracer.merge_similar(t, s)
    assert merged >= 1
    assert abs(float(np.sum(out.weights)) - 1.0) < 1e-9
    # the survivor of the duplicate family must hold the family's whole mass,
    # and the unrelated particle must be untouched
    by_anchor = {h.anchor: float(w) for h, w in zip(out.hypotheses, out.weights)}
    assert abs(by_anchor.get('D', 0) - 0.1) < 1e-9, by_anchor


def test_anchor_merge_is_off_by_default():
    t = _merge_tracer()
    called = []
    t.merge_equivalent_anchors = lambda *a, **k: called.append(1) or {}
    s = _set(4, methods=[None] * 4)
    tracer.Tracer.merge_similar(t, s)
    assert not called, "anchor merge must not run unless --merge-anchors is set"


def _anchor_set(anchors, weights):
    return HypothesesSetV3('Katara', [], [],
                           [f'belief prose number {k}' for k in range(len(anchors))],
                           np.array(weights), anchors=anchors)


def test_adjudicator_merges_the_same_aim_and_spares_the_opposite():
    """The load-bearing test. Lexical similarity rates

        Avenge her mother / Punish her mother's killer   (same)      0.17
        Avenge her mother / Forgive her mother's killer  (opposite)  0.17

    identically, so the candidate generator must offer BOTH and the model must
    separate them. Merging the opposite pair would delete the one commitment
    this corpus is documented as never reaching.
    """
    t = _merge_tracer(merge_anchors=True)
    asked = {}

    class _M:
        def interact(self, prompt, **kw):
            out = []
            for ln in prompt.splitlines():
                if '|' not in ln or not ln.strip()[0].isdigit():
                    continue
                n = ln.split('.')[0].strip()
                asked[n] = ln
                out.append(f"{n}: " + ("DIFFERENT" if 'Forgive' in ln else "SAME"))
            return "\n".join(out)
    t.tracer_model = _M()

    s = _anchor_set(['Avenge her mother', "Punish her mother's killer",
                     "Forgive her mother's killer", 'Preserve her peace of mind'],
                    [0.4, 0.3, 0.2, 0.1])
    got = tracer.Tracer.merge_equivalent_anchors(t, s, already={})

    pairs = " ".join(asked.values())
    assert 'Punish' in pairs and 'Forgive' in pairs, \
        "both lookalikes must be offered for adjudication"
    assert 'Preserve her peace of mind' not in pairs, \
        "an unrelated aim must not even be a candidate"
    # index 1 (Punish) absorbed into 0 (Avenge, heavier); Forgive untouched
    assert got.get(1) == 0, got
    assert 2 not in got, "the opposite aim was merged away"
    assert 3 not in got


def test_adjudicator_defaults_to_different_when_the_verdict_is_unparsed():
    # wrongly merging destroys a hypothesis; wrongly keeping costs one particle
    t = _merge_tracer(merge_anchors=True)
    t.tracer_model = types.SimpleNamespace(
        interact=lambda *a, **k: "the model rambled and gave no verdicts")
    s = _anchor_set(['Avenge her mother', "Punish her mother's killer"], [0.6, 0.4])
    assert tracer.Tracer.merge_equivalent_anchors(t, s, already={}) == {}


def test_adjudicator_skips_particles_text_merge_already_took():
    t = _merge_tracer(merge_anchors=True)
    t.tracer_model = types.SimpleNamespace(
        interact=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("nothing left to ask about")))
    s = _anchor_set(['Avenge her mother', "Punish her mother's killer"], [0.6, 0.4])
    assert tracer.Tracer.merge_equivalent_anchors(t, s, already={1: 0}) == {}


def test_identical_anchors_in_DIFFERENT_roots_merge_without_asking():
    # the anchor==root invariant says one clause is one root, so two roots
    # carrying the same string should never have existed; the strings settle
    # it and spending a call to confirm would be waste
    t = _merge_tracer(merge_anchors=True)
    t.tracer_model = types.SimpleNamespace(
        interact=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not call the model for identical clauses")))
    s = _anchor_set(['Avenge her mother', 'Avenge her mother',
                     'Preserve her peace of mind'], [0.5, 0.3, 0.2])
    # canonicalisation gives the two identical clauses one root at founding;
    # force them apart to represent two roots that drifted onto one clause
    s.hypotheses[1].root_id = 'a-separate-root'
    got = tracer.Tracer.merge_equivalent_anchors(t, s, already={})
    assert got == {1: 0}, got


def test_dedup_never_collapses_a_root_s_own_multiplicity():
    """Resampling encodes the posterior in multiplicity.

    A root holding three copies at 0.1 carries 0.3. Merging its copies does
    not remove redundancy -- it deletes the representation of that
    hypothesis's mass, and the heaviest hypothesis loses the most. Measured
    before this guard: 5 of 8 particles absorbed at step 0, every one of them
    a copy of a single seeded commitment.
    """
    t = _merge_tracer(merge_anchors=True)
    t.tracer_model = types.SimpleNamespace(
        interact=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a root's own copies must never be adjudicated")))
    s = _anchor_set(['Understand Zuko\'s actions and intentions'] * 3
                    + ['Preserve her peace of mind'], [0.3, 0.3, 0.3, 0.1])
    # founding canonicalisation already puts the three identical clauses in
    # one root; assert that, then assert dedup leaves them alone
    assert len({h.root_id for h in s.hypotheses[:3]}) == 1
    assert tracer.Tracer.merge_equivalent_anchors(t, s, already={}) == {}
    out, merged, _ = tracer.Tracer.dedupe_anchors(t, s)
    assert merged == 0 and len(out.hypotheses) == 4


# --------------------------------------------------------------------------
# the mint chooser
# --------------------------------------------------------------------------

class _Chooser:
    _mint_method = tracer.Tracer._mint_method
    def __init__(self, methods):
        self.args = types.SimpleNamespace(methods=methods)


def test_mint_chooser_is_none_on_the_default_path():
    assert _Chooser(None)._mint_method('surprise') is None


def test_mint_chooser_round_robins():
    c = _Chooser(['silence', 'assume', 'premortem'])
    got = [c._mint_method('stagnation') for _ in range(6)]
    assert got == ['silence', 'assume', 'premortem'] * 2


def test_mint_chooser_filters_on_needs_action():
    # anomaly and ach reason from a datum that does not fit; only the surprise
    # path puts one in the prompt. Firing them elsewhere would stamp a label on
    # a prompt that never used the method.
    c = _Chooser(['anomaly', 'silence', 'ach'])
    assert set(c._mint_method('surprise') for _ in range(6)) == {'anomaly', 'ach'}
    c2 = _Chooser(['anomaly', 'silence', 'ach'])
    assert set(c2._mint_method('stagnation') for _ in range(6)) == {'silence'}


def test_mint_chooser_falls_back_rather_than_blocking_repair():
    # diversity repair is the operator's job and must not be blocked by
    # bookkeeping: an empty filtered pool falls back to the full list
    c = _Chooser(['anomaly', 'ach'])
    assert c._mint_method('stagnation') in ('anomaly', 'ach')


# --------------------------------------------------------------------------
# seeding: the default prompt must not move
# --------------------------------------------------------------------------

class _StubModel:
    """Records every prompt and returns a well-formed numbered list."""
    def __init__(self):
        self.prompts = []
    def interact(self, prompt, **kw):
        self.prompts.append(prompt)
        n = kw.get('_n', 8)
        return "\n".join(f"{i+1}. Katara believes thing {i}" for i in range(n))
    def batch_interact(self, prompts, **kw):
        return [self.interact(p) for p in prompts]


def _seed(methods, n_hypotheses=4):
    t = tracer.Tracer.__new__(tracer.Tracer)
    t.args = types.SimpleNamespace(
        n_hypotheses=n_hypotheses, methods=methods, goal_seeding=True,
        infer_motive=False, extract_anchors_flag=False, use_anchor=False,
        dataset='musing', use_cot=False, input_is_chat=True,
        target_perceptions='sight', use_helper_llm=False)
    t.target_agent = 'Katara'
    t.assumption = ''
    t.tracer_model = _StubModel()
    t._last_standards = []
    return t


def test_seeding_default_prompt_is_unchanged():
    t = _seed(None)
    out = tracer.Tracer.initialize(t, {'state': 'ctx', 'action': 'Katara speaks.'},
                                   {'state': '', 'action': ''})
    prompt = t.tracer_model.prompts[0]
    # exactly one seeding call, no method text, and the count phrasing the
    # pre-change code used
    assert len(t.tracer_model.prompts) == 1
    assert 'METHOD' not in prompt
    assert 'Generate a numbered list of 4 hypotheses' in prompt
    assert out.methods == [None] * 4


def test_seeding_one_call_per_method_and_labels_attach():
    t = _seed(['anomaly', 'silence'])
    out = tracer.Tracer.initialize(t, {'state': 'ctx', 'action': 'Katara speaks.'},
                                   {'state': '', 'action': ''})
    assert len(t.tracer_model.prompts) == 2, "one seeding call per method, pooled"
    # each call carries its own frame and only its own
    assert 'the detail that does not fit' in t.tracer_model.prompts[0]
    assert 'what is missing' not in t.tracer_model.prompts[0]
    assert 'what is missing' in t.tracer_model.prompts[1]
    assert 'the detail that does not fit' not in t.tracer_model.prompts[1]
    # N is respected, and the split is interleaved so neither method is starved
    assert len(out.hypotheses) == 4
    assert sorted(out.methods) == ['anomaly', 'anomaly', 'silence', 'silence']


def test_seeding_respects_n_when_methods_do_not_divide_it():
    t = _seed(['anomaly', 'silence', 'devil'], n_hypotheses=4)
    out = tracer.Tracer.initialize(t, {'state': 'ctx', 'action': 'Katara speaks.'},
                                   {'state': '', 'action': ''})
    assert len(out.hypotheses) == 4
    # interleaving means the truncation costs the LAST method one particle,
    # not all of them
    assert set(out.methods) == {'anomaly', 'silence', 'devil'}


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
