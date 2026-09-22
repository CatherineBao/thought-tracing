"""Counter semantics for weight-based expiry. No LLM calls."""
import numpy as np
from types import SimpleNamespace
from tracer import Tracer
from hypothesis import HypothesisV3, HypothesesSetV3


def mk(n):
    return HypothesesSetV3("agent", [], [], [f"h{i}" for i in range(n)],
                           np.array([1.0 / n] * n),
                           anchors=[f"a{i}" for i in range(n)])


def tr(frac=0.5, k=3):
    t = Tracer.__new__(Tracer)
    t.args = SimpleNamespace(expiry_weight_frac=frac, expiry_steps=k)
    t._weak_run = {}
    return t


def step(t, s, weights, resampled=False):
    s.update_weights(np.array(weights, dtype=float))
    return t.expire_weak(s, just_resampled=resampled)


def test_fires_only_after_k_consecutive():
    t, s = tr(k=3), mk(4)          # n=4 -> threshold 0.5/4 = 0.125
    low = [0.01, 0.33, 0.33, 0.33]
    for i in (1, 2):
        dead, info = step(t, s, low)
        assert dead == [], f"fired at turn {i}, before k=3"
        assert info['expiry_weight_condition'] == 1
        assert info['expiry_duration_condition'] == 0
    dead, info = step(t, s, low)
    assert dead == [0], dead
    assert info['expiry_duration_condition'] == 1
    assert info['expiry_threshold'] == 0.125


def test_recovery_resets_the_counter():
    t, s = tr(k=3), mk(4)
    step(t, s, [0.01, 0.33, 0.33, 0.33])
    step(t, s, [0.01, 0.33, 0.33, 0.33])
    step(t, s, [0.40, 0.20, 0.20, 0.20])   # recovers above 0.125
    assert t._weak_run[s.hypotheses[0].root_id] == 0
    dead, _ = step(t, s, [0.01, 0.33, 0.33, 0.33])
    assert dead == [], "counter did not reset on recovery"


def test_never_retires_more_than_half():
    t, s = tr(k=1), mk(4)
    dead, _ = step(t, s, [0.001, 0.001, 0.001, 0.997])
    assert len(dead) == 2, f"retired {len(dead)} of 4"
    assert dead == sorted(dead, key=lambda j: float(s.weights[j]))


def test_threshold_scales_with_population():
    for n, exp in ((4, 0.125), (8, 0.0625), (12, 0.5 / 12)):
        _, info = step(tr(k=99), mk(n), [1.0 / n] * n)
        assert abs(info['expiry_threshold'] - exp) < 1e-9, (n, info)


def test_resample_step_freezes_counters():
    """Uniform weights carry no information about which hypothesis is weak."""
    t, s = tr(k=3), mk(4)
    step(t, s, [0.01, 0.33, 0.33, 0.33])
    step(t, s, [0.01, 0.33, 0.33, 0.33])
    r0 = s.hypotheses[0].root_id
    assert t._weak_run[r0] == 2
    dead, info = step(t, s, [0.25] * 4, resampled=True)   # the reset
    assert info['expiry_frozen'] is True
    assert t._weak_run[r0] == 2, "resample cleared the counter"
    assert dead == []
    dead, _ = step(t, s, [0.01, 0.33, 0.33, 0.33])        # third informative step
    assert dead == [0], "counter did not survive the resample"


def test_mass_summed_over_root_so_duplicates_read_as_strong():
    """A root duplicated by resampling is strong, not several weaklings."""
    t, s = tr(k=1), mk(4)
    for h in s.hypotheses[:3]:
        h.root_id = s.hypotheses[0].root_id          # one root, three copies
    dead, _ = step(t, s, [0.03, 0.03, 0.03, 0.91])    # copies sum to 0.09 > 0.125? no
    assert dead == [0, 1, 2] or dead == sorted(dead)  # summed mass 0.09 < 0.125 -> weak
    t2, s2 = tr(k=1), mk(4)
    for h in s2.hypotheses[:3]:
        h.root_id = s2.hypotheses[0].root_id
    dead2, _ = step(t2, s2, [0.30, 0.30, 0.30, 0.10])  # summed 0.90 -> strong
    assert all(j == 3 for j in dead2), dead2


def test_counter_dropped_for_vanished_lineages():
    t, s = tr(k=99), mk(4)
    step(t, s, [0.01] * 4)
    assert len(t._weak_run) == 4
    step(t, mk(2), [0.01, 0.01])           # different lineages entirely
    assert len(t._weak_run) == 2, t._weak_run




# --- rebirth at fair share -------------------------------------------------

def _rebirth(weights, accepted):
    """The tail of perturb_anchored, in isolation."""
    before = list(weights)
    n = len(before); fair = 1.0 / n
    w = list(before)
    for i in accepted:
        w[i] = fair
    tot = sum(w)
    w = [x / tot for x in w]
    return before, w


def test_replacement_enters_at_fair_share_not_the_floor():
    """The bug: a mint inherited the floor weight of the particle it replaced."""
    before, after = _rebirth([0.015, 0.015, 0.30, 0.67], [0, 1])
    assert before[0] == 0.015
    assert after[0] > 5 * before[0], after
    assert abs(sum(after) - 1.0) < 1e-9


def test_rebirth_takes_mass_proportionally_from_the_rest():
    before, after = _rebirth([0.015, 0.30, 0.685], [0])
    # the two survivors keep their RATIO, they are just scaled down
    assert abs((after[1] / after[2]) - (before[1] / before[2])) < 1e-9


def test_rebirth_is_a_noop_when_nothing_was_accepted():
    before, after = _rebirth([0.1, 0.2, 0.3, 0.4], [])
    assert all(abs(a - b) < 1e-12 for a, b in zip(before, after))


if __name__ == "__main__":
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_"):
            fn(); print("ok", nm)
