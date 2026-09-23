"""Phase 4 prerequisite: mass conservation for split and merge.

TWO invariants, and only one catches the failure that matters:

  weight conservation  sum(child weights) == parent weight.
                       Necessary, but a split that gave its children the WRONG
                       ROOTS still passes it.

  root-mass conservation
                       Split children receive new, mutually exclusive anchors
                       and therefore new roots (anchor==root invariant). The
                       mass held by the parent's root BEFORE the split must
                       equal the sum held by the children's new roots AFTER.
                       This is what verifies the parent's ancestral mass
                       TRANSFERRED to its children rather than leaking to
                       unrelated roots or being double-counted into a surviving
                       parent root.

Plus the source-term check: distinct_roots must increase by exactly
(children - 1). If it does not, the children inherited rather than founded and
split has silently become mass division that creates no diversity -- the exact
failure the anchor lifecycle was rewritten to prevent.
"""
import numpy as np

from hypothesis import HypothesesSetV3

TOL = 1e-9


def _mass_by_root(hyps):
    out = {}
    for h, w in zip(hyps.hypotheses, hyps.weights):
        out[h.root_id] = out.get(h.root_id, 0.0) + float(w)
    return out


def split_particle(hyps, idx, child_anchors):
    """Reference split: parent mass distributed evenly among mutually exclusive children."""
    parent = hyps.hypotheses[idx]
    share = float(hyps.weights[idx]) / len(child_anchors)
    texts = list(hyps.texts)
    weights = [float(w) for w in hyps.weights]
    anchors = [h.anchor for h in hyps.hypotheses]
    accs = list(hyps.accumulators)
    parents = list(hyps.hypotheses)

    texts[idx] = f"{parent.text} [refined: {child_anchors[0]}]"
    weights[idx] = share
    anchors[idx] = child_anchors[0]
    for a in child_anchors[1:]:
        texts.append(f"{parent.text} [refined: {a}]")
        weights.append(share)
        anchors.append(a)
        accs.append(parent.raw_accumulator)
        parents.append(parent)

    out = HypothesesSetV3(hyps.target_agent, hyps.contexts, hyps.perceptions,
                          texts, np.array(weights), parent_hypotheses=parents,
                          accumulators=accs)
    # a NEW anchor founds a NEW root -- this is what makes split a source term
    out.update_anchors(anchors)
    for h in out.hypotheses:
        h.split_child = h.anchor in child_anchors
    return out


def merge_particles(hyps, i, j):
    """Reference merge: survivor absorbs the other's weight; absorbed root dies."""
    keep, drop = (i, j) if float(hyps.weights[i]) >= float(hyps.weights[j]) else (j, i)
    texts, weights, anchors, accs, parents = [], [], [], [], []
    for k, h in enumerate(hyps.hypotheses):
        if k == drop:
            continue
        w = float(hyps.weights[k])
        if k == keep:
            w += float(hyps.weights[drop])
        texts.append(h.text); weights.append(w); anchors.append(h.anchor)
        accs.append(h.raw_accumulator); parents.append(h)
    out = HypothesesSetV3(hyps.target_agent, hyps.contexts, hyps.perceptions,
                          texts, np.array(weights), parent_hypotheses=parents,
                          anchors=anchors, accumulators=accs)
    return out, hyps.hypotheses[drop].root_id, hyps.hypotheses[keep].root_id


def build(n=4):
    ctx = [{'state': 's', 'action': 'a'}]
    perc = [{'state': None, 'action': None}]
    anchors = [f'commitment {k}' for k in range(n)]
    w = np.array([0.4, 0.3, 0.2, 0.1][:n])
    return HypothesesSetV3('X', ctx, perc, [f't{k}' for k in range(n)], w, anchors=anchors)


def test_split_conserves_weight_and_root_mass():
    hyps = build()
    before_roots = _mass_by_root(hyps)
    parent_root = hyps.hypotheses[0].root_id
    parent_w = float(hyps.weights[0])
    parent_root_mass = before_roots[parent_root]
    n_roots_before = len(before_roots)

    out = split_particle(hyps, 0, ['refine A', 'refine B', 'refine C'])
    after_roots = _mass_by_root(out)

    children = [h for h in out.hypotheses if h.split_child]
    assert len(children) == 3, f"expected 3 children, got {len(children)}"

    # 1. weight conservation
    child_w = sum(float(w) for h, w in zip(out.hypotheses, out.weights) if h.split_child)
    assert abs(child_w - parent_w) < TOL, f"weight not conserved: {child_w} vs {parent_w}"

    # 2. root-mass conservation -- the invariant weight conservation cannot catch
    child_roots = {h.root_id for h in children}
    child_root_mass = sum(after_roots[r] for r in child_roots)
    assert abs(child_root_mass - parent_root_mass) < TOL, \
        f"root-mass not conserved: {child_root_mass} vs {parent_root_mass}"
    assert parent_root not in child_roots, "children must FOUND roots, not inherit the parent's"

    # 3. untouched roots unchanged -- nothing leaked
    for r, m in before_roots.items():
        if r == parent_root:
            continue
        assert abs(after_roots[r] - m) < TOL, f"unrelated root {r} changed: {m} -> {after_roots[r]}"

    # 4. source term: distinct_roots += (children - 1)
    assert len(after_roots) == n_roots_before + 2, \
        f"split is not a source term: {n_roots_before} -> {len(after_roots)}, expected +2"

    # 5. total mass still normalized
    assert abs(sum(after_roots.values()) - 1.0) < TOL
    print("  split: weight OK | root-mass OK | roots founded OK | +2 roots OK")


def test_merge_conserves_weight_and_root_mass():
    hyps = build()
    before = _mass_by_root(hyps)
    n_before = len(before)
    out, dead_root, keep_root = merge_particles(hyps, 0, 1)
    after = _mass_by_root(out)

    assert abs(sum(after.values()) - 1.0) < TOL, "merge must preserve total mass"
    assert dead_root not in after, "absorbed root must die"
    expected = before[dead_root] + before[keep_root]
    assert abs(after[keep_root] - expected) < TOL, \
        f"survivor mass wrong: {after[keep_root]} vs {expected}"
    assert len(after) == n_before - 1, "merge must remove exactly one root"
    for r, m in before.items():
        if r in (dead_root, keep_root):
            continue
        assert abs(after[r] - m) < TOL, f"unrelated root {r} changed"
    print("  merge: total mass OK | absorbed root died | survivor absorbed OK | -1 root OK")


def test_split_then_merge_roundtrip():
    """Split then merge the children back: mass must return to the parent's root total."""
    hyps = build()
    parent_root_mass = _mass_by_root(hyps)[hyps.hypotheses[0].root_id]
    out = split_particle(hyps, 0, ['refine A', 'refine B'])
    kids = [k for k, h in enumerate(out.hypotheses) if h.split_child]
    merged, dead, keep = merge_particles(out, kids[0], kids[1])
    after = _mass_by_root(merged)
    assert abs(after[keep] - parent_root_mass) < TOL, \
        f"roundtrip lost mass: {after[keep]} vs {parent_root_mass}"
    assert abs(sum(after.values()) - 1.0) < TOL
    print("  roundtrip: split->merge returns the parent's root mass exactly")


if __name__ == "__main__":
    print("Phase 4 mass-conservation prerequisites\n")
    test_split_conserves_weight_and_root_mass()
    test_merge_conserves_weight_and_root_mass()
    test_split_then_merge_roundtrip()
    print("\nALL PASS")
