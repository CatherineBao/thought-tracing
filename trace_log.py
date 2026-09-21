"""Per-step instrumentation for the particle filter.

One StepRecord per trajectory step, written as JSONL. viz.py reads only this
stream. Nothing here imports tracer.py, so the log schema stays independent of
the filter internals.

The effective-temperature recorder is the load-bearing piece: agents call
record_llm_call() at the point they resolve the temperature actually sent to the
API, so the log reports reality rather than the intent at the call site.
"""
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_particle_id() -> str:
    return uuid.uuid4().hex[:8]


# --------------------------------------------------------------------------
# effective-temperature recorder
# --------------------------------------------------------------------------

class _CallRecorder:
    """Collects LLM calls for the step currently being traced.

    Agents call record() from inside generate(), where the effective temperature
    has just been resolved. Thread-safe because batch_generate fans out over a
    ThreadPoolExecutor.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: List[Dict[str, Any]] = []
        self._stage = "unknown"

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def record(self, *, temperature, max_tokens=None, model=None, attempt=0, stage=None,
               finish_reason=None, truncated=None, thoughts_tokens=None, visible_tokens=None) -> None:
        with self._lock:
            self._calls.append({
                "stage": stage or self._stage,
                "effective_temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
                "attempt": attempt,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "thoughts_tokens": thoughts_tokens,
                "visible_tokens": visible_tokens,
            })

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            calls, self._calls = self._calls, []
            return calls

    def peek(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._calls)


RECORDER = _CallRecorder()


def record_llm_call(**kwargs) -> None:
    """Module-level entry point so agents can instrument with one import."""
    RECORDER.record(**kwargs)


def set_stage(stage: str) -> None:
    RECORDER.set_stage(stage)


# --------------------------------------------------------------------------
# step record
# --------------------------------------------------------------------------

@dataclass
class ParticleRecord:
    particle_id: str
    # stable across steps -- this is what every per-particle series keys on
    lineage_id: Optional[str] = None
    # Ancestral root; invariant across resampling. Weight trajectories are
    # reset by resampling, ancestral MASS is not.
    root_id: Optional[str] = None
    parent_id: Optional[str] = None
    split_child: bool = False
    resample_duplicate: bool = False
    text: str = ""
    anchor: Optional[str] = None
    # normalized posterior weight, used for argmax churn
    weight: float = 0.0
    # unnormalized accumulated log-weight: alpha*log w_{t-1} + beta*log L_t.
    # This is the particle's own evidence trajectory and the ONLY series that
    # direction reversals may be counted on.
    raw_accumulator: Optional[float] = None
    likelihood: Optional[float] = None
    likelihood_rank: Optional[int] = None
    # split trigger, logged separately whether or not split fires, so a zero
    # split count can be attributed to "nothing to split" vs "can never fire"
    mass_condition_met: Optional[bool] = None
    rank_condition_met: Optional[bool] = None
    parse_failed: bool = False
    perturbed: bool = False
    perturb_accepted: Optional[bool] = None
    anchor_revisions: int = 0


@dataclass
class StepRecord:
    step_idx: int
    particles: List[ParticleRecord] = field(default_factory=list)

    weights_pre: List[float] = field(default_factory=list)
    weights_post: List[float] = field(default_factory=list)

    # two ESS values, never conflated: likelihood_ess gates Phase 1,
    # posterior_ess drives the resample trigger
    likelihood_ess: Optional[float] = None
    # posterior_ess is the ESS of the particles ACTUALLY LOGGED below, i.e.
    # after any operator fired. posterior_ess_pre is the value that drove the
    # resample decision. Logging only the pre value while listing post-operator
    # particles made the record internally inconsistent at exactly the steps
    # that matter most.
    posterior_ess_pre: Optional[float] = None
    # ESS over ANCESTRAL ROOT mass, normalized by the PARTICLE count -- reads as
    # "effective independent hypotheses out of N". Normalizing by distinct-root
    # count instead would put it on a different scale from posterior ESS and
    # make the two incomparable.
    root_mass_ess: Optional[float] = None
    # Distinct ancestral roots alive this step. First-class, not derived: this
    # is the quantity Phase 3 exists to protect, and resampling collapses it
    # (measured 8 -> 3-4, recovering over ~5 steps).
    distinct_roots: Optional[int] = None
    # Mean text dissimilarity BETWEEN copies that share a root. The gap the
    # anchor==root equivalence opens: two copies of one commitment, propagated
    # independently, can drift to genuinely different beliefs while still
    # counting as one root -- so root count would UNDERSTATE diversity, the
    # mirror of particle ESS overstating it. 3b's anchor is precisely what is
    # supposed to prevent that drift, so the pre-3b baseline must carry this
    # field or the anchor cannot be shown to work.
    within_root_divergence: Optional[float] = None
    # Merge threshold resolved from THIS run's similarity distribution, and the
    # percentile it came from. Logged so a later reader knows the cut was
    # derived, not inherited.
    merge_threshold_resolved: Optional[float] = None
    merge_percentile: Optional[float] = None
    merge_metric: Optional[str] = None
    # Roots founded THIS step (by 3e) and the weight they hold. The discriminator
    # between "the source term works" and "divergence does not survive the
    # likelihood evaluator": a minted root that is immediately downweighted means
    # the scorer penalizes genuine alternatives, which is a finding about the
    # SCORER, not about 3e, and nothing downstream fixes it.
    # 3e trigger, logged as TWO separate conditions -- the same lesson as the
    # split diagnostic. root_mass_ess conflates "few distinct roots"
    # (the collapse 3e repairs) with "many roots, concentrated weight" (the
    # filter correctly converging, which must NOT be perturbed). Logging one
    # boolean makes "never fired" ambiguous between opposite causes.
    perturb_mass_condition: Optional[bool] = None   # root-mass below threshold
    perturb_collapse_condition: Optional[bool] = None  # duplicate roots exist
    perturb_candidates: Optional[int] = None
    # which path fired: 'collapse' (duplicates to break apart) or 'stagnation'
    # (no duplicates, but root-mass has been low for k consecutive steps)
    perturb_path: Optional[str] = None
    perturb_sustained_steps: Optional[int] = None
    # Phase 4 split trigger, logged as two conditions like 3e's.
    # The ORIGINAL spec (weight > 3x mean AND mid-pack likelihood rank) fires
    # once in 476 particle-steps: corr(weight, likelihood rank) = -0.866 and all
    # 57 particles above 3x mean were rank 1. Weight IS accumulated likelihood,
    # so "carries mass" entails "currently top-ranked". Redesigned as
    # prior/likelihood DISAGREEMENT: top-quartile weight, below-median current
    # likelihood -- accumulated support not being renewed. Fires 0.7-2.7/run.
    split_weight_condition: Optional[int] = None
    split_disagree_condition: Optional[int] = None
    split_candidates: Optional[int] = None
    population_cap: Optional[int] = None
    cap_bound: bool = False        # particles eligible to perturb
    minted_roots: List[str] = field(default_factory=list)
    minted_root_weight: Optional[float] = None
    # roots holding >1 particle, i.e. where drift is even possible
    multi_particle_roots: Optional[int] = None
    # ESS/n over the PARSED subset. Masking caps raw ESS at n, so a step with
    # several parse failures looks sharp while being perfectly flat within its
    # scored set (observed: 4.70 on 5 particles read as "sharp" against a raw
    # gate of 6.0). The gate is on this, not on raw ESS.
    likelihood_ess_norm: Optional[float] = None
    posterior_ess: Optional[float] = None
    top_to_median_ratio: Optional[float] = None

    mass_moved_by_floor: Optional[float] = None
    epsilon: Optional[float] = None
    ess_threshold: Optional[float] = None
    ess_threshold_name: Optional[str] = None

    population_pre: Optional[int] = None
    population_post: Optional[int] = None
    population_cap: Optional[int] = None
    cap_bound: bool = False

    # Snapshot of the population BEFORE any operator fired. Post-resample
    # parent_ids point into this set, so without it root reconstruction breaks
    # at every resample boundary -- every particle looks like a founder.
    pre_operator_particles: List[Dict[str, Any]] = field(default_factory=list)

    operators_fired: List[str] = field(default_factory=list)
    anchor_collapses: List[Dict[str, Any]] = field(default_factory=list)

    jaccard_matrix: Optional[List[List[float]]] = None
    cosine_matrix: Optional[List[List[float]]] = None
    mean_pairwise_cosine: Optional[float] = None
    min_pairwise_cosine: Optional[float] = None
    mean_pairwise_jaccard: Optional[float] = None

    raw_verdicts: List[Any] = field(default_factory=list)
    parsed_scores: List[Any] = field(default_factory=list)
    parse_failures: int = 0
    truncated_calls: int = 0

    # Scoring inputs, so a step's verdict can be replayed from this file alone.
    # Reconstructing them from the tracer dump worked but needed a second file
    # and exact context matching; scorer A/Bs are routine now and Phase 4 will
    # want the same replay against a branching population.
    likelihood_prompt: Optional[str] = None
    likelihood_system_prompt: Optional[str] = None
    scored_texts: List[str] = field(default_factory=list)
    scored_action: Optional[str] = None

    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def ess_norm(weights, mask=None) -> Optional[float]:
    """Normalized ESS in (0, 1]: 1.0 is perfectly uniform, lower is sharper.

    Scale-free, so it is comparable across steps with different numbers of
    parsed verdicts and across runs with different N.
    """
    ws = [float(w) for w in weights]
    if mask is not None:
        ws = [w for w, bad in zip(ws, mask) if not bad]
    if len(ws) < 2:
        return None
    e = ess(ws)
    return None if e is None else e / len(ws)


def ess(weights, mask=None) -> Optional[float]:
    """1/sum(w^2) over a normalized vector, renormalized after masking.

    `mask` is a list of booleans marking entries to EXCLUDE (parse failures).
    An unparsed verdict carries no information, so counting its imputed weight
    toward separation would report discrimination that never happened. Returns
    None when fewer than two entries survive.
    """
    ws = [float(w) for w in weights]
    if mask is not None:
        ws = [w for w, bad in zip(ws, mask) if not bad]
        if len(ws) < 2:
            return None
    total = sum(ws)
    if not ws or total <= 0:
        return None
    ws = [w / total for w in ws]
    denom = sum(w * w for w in ws)
    return (1.0 / denom) if denom > 0 else None


def top_to_median_ratio(weights) -> Optional[float]:
    ws = sorted((float(w) for w in weights), reverse=True)
    if not ws:
        return None
    mid = ws[len(ws) // 2] if len(ws) % 2 else (ws[len(ws) // 2 - 1] + ws[len(ws) // 2]) / 2.0
    return (ws[0] / mid) if mid > 0 else float("inf")


def count_reversals(series, deadband: float = 1e-3) -> int:
    """Direction reversals in one particle's UNNORMALIZED accumulator.

    Changes with magnitude below `deadband` are treated as flat rather than as a
    direction, so floating-point jitter on a flat series cannot manufacture
    reversals. Never call this on normalized weights: those are coupled across
    particles and would fire on a neighbour's movement.
    """
    vals = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
    direction = 0
    reversals = 0
    for a, b in zip(vals, vals[1:]):
        delta = b - a
        if abs(delta) < deadband:
            continue
        d = 1 if delta > 0 else -1
        if direction and d != direction:
            reversals += 1
        direction = d
    return reversals


def argmax_churn(steps: List[StepRecord], tol: float = 1e-9) -> int:
    """Steps at which the highest-weight particle changes identity.

    Ties do not count. After a resample every particle carries 1/N, so max()
    picks arbitrarily among equals and a naive count reports churn on almost
    every step -- an artifact of tie-breaking, not a leader actually changing.
    A step with no strict leader carries the previous leader forward.

    Correct on normalized weights: it is inherently a cross-particle comparison,
    so renormalization does not distort it.
    """
    churn = 0
    prev = None
    for s in steps:
        ws = [(p.weight if p.weight is not None else -1.0, p.lineage_id or p.particle_id) for p in s.particles]
        if len(ws) < 1:
            continue
        ws.sort(key=lambda t: -t[0])
        if len(ws) > 1 and abs(ws[0][0] - ws[1][0]) <= tol:
            continue  # no strict leader this step
        top = ws[0][1]
        if prev is not None and top != prev:
            churn += 1
        prev = top
    return churn


def kendall_tau(a: List[str], b: List[str]) -> Optional[float]:
    """Kendall's tau between two orderings over the same items."""
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    ra = {k: i for i, k in enumerate([x for x in a if x in set(common)])}
    rb = {k: i for i, k in enumerate([x for x in b if x in set(common)])}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s_ = (ra[x] - ra[y]) * (rb[x] - rb[y])
            if s_ > 0:
                conc += 1
            elif s_ < 0:
                disc += 1
    tot = conc + disc
    return None if tot == 0 else (conc - disc) / tot


def rank_persistence(steps) -> Dict[str, Any]:
    """Does the same lineage keep winning?

    Accumulation amplifies weak per-step signal by ~1/(1-alpha) ONLY if rank
    order is stable. If it churns, accumulation averages it to nothing, and no
    amount of per-step prompt work helps.
    """
    orders = []
    for s in steps:
        ranked = [p for p in s.particles if p.weight is not None]
        if len(ranked) < 2:
            continue
        ranked.sort(key=lambda p: -p.weight)
        orders.append([(p.lineage_id or p.particle_id) for p in ranked])
    if len(orders) < 2:
        return {}
    taus = [t for t in (kendall_tau(a, b) for a, b in zip(orders, orders[1:])) if t is not None]
    tops = [o[0] for o in orders]
    repeats = sum(1 for a, b in zip(tops, tops[1:]) if a == b)
    n_l = len({l for o in orders for l in o})
    return {
        "mean_kendall_tau": sum(taus) / len(taus) if taus else None,
        "median_kendall_tau": _median(taus),
        "top_repeat_rate": repeats / max(1, len(tops) - 1),
        "top_repeat_chance": 1.0 / n_l if n_l else None,
        "distinct_top_lineages": len(set(tops)),
        "n_transitions": len(taus),
    }


def ancestral_mass(steps) -> Dict[str, List[Optional[float]]]:
    """Total posterior mass per ancestral root, per step.

    Resampling resets weights to 1/N because the posterior is then encoded in
    MULTIPLICITY -- duplicated winners carry the information. Correct as
    inference, but it restarts every per-particle weight trajectory, so
    reversals and argmax churn become uninterpretable across the boundary and
    "strengthens over the run" cannot be observed on any run where resampling
    fires.

    Ancestral mass is invariant to that encoding choice: a winner duplicated
    three times at 1/N each holds 3/N of the mass, which is exactly the
    strength the reset appeared to erase. Reversals are counted on this.
    """
    roots = []
    for s in steps:
        for p in s.particles:
            r = p.root_id or p.lineage_id or p.particle_id
            if r not in roots:
                roots.append(r)
    out = {r: [] for r in roots}
    for s in steps:
        tot = {}
        for p in s.particles:
            r = p.root_id or p.lineage_id or p.particle_id
            tot[r] = tot.get(r, 0.0) + (p.weight or 0.0)
        for r in roots:
            out[r].append(tot.get(r))
    return out


def resolve_deadband(series_list, frac: float = 0.25) -> float:
    """Deadband as a FRACTION of the run's median step size.

    An absolute 0.001 against a median |delta| of 0.084 counted jitter as
    trend (84x too small). An absolute value recalibrated for one run drifts
    out of calibration as soon as the step-size distribution changes -- which
    Phase 3 does directly, by widening likelihood spreads. Always log the
    resolved value alongside the metric it produced.
    """
    deltas = [abs(b - a) for ser in series_list
              for a, b in zip(ser, ser[1:]) if a is not None and b is not None]
    if not deltas:
        return 1e-3
    return max(1e-9, frac * _median(deltas))


def root_mass_ess_over_n(particles) -> Optional[float]:
    """Effective independent hypotheses out of N, from ancestral root mass.

    Duplicated winners share a root, so their copies do not count as
    independent. Normalized by particle count (not root count) so it sits on
    the same scale as posterior ESS and the two can be differenced.
    """
    tot = {}
    n = 0
    for p in particles:
        n += 1
        r = p.root_id or p.lineage_id or p.particle_id
        tot[r] = tot.get(r, 0.0) + (p.weight or 0.0)
    if n < 2 or not tot:
        return None
    vals = list(tot.values())
    z = sum(vals) or 1.0
    vals = [v / z for v in vals]
    denom = sum(v * v for v in vals)
    return (1.0 / denom) / n if denom > 0 else None


def minted_root_survival(steps) -> Dict[str, Any]:
    """Track what happens to roots 3e founds, after it founds them.

    Branch A (source term works): minted roots hold or grow their mass.
    Branch B (divergence rejected downstream): minted roots are downweighted
    within a step or two and root-mass never recovers. B is a scorer finding.
    """
    minted = {}
    for s in steps:
        for r in (s.minted_roots or []):
            minted[r] = {"born": s.step_idx, "mass": []}
    if not minted:
        return {"minted_count": 0}
    for s in steps:
        tot = {}
        for p in s.particles:
            k = p.root_id or p.lineage_id or p.particle_id
            tot[k] = tot.get(k, 0.0) + (p.weight or 0.0)
        for r, rec in minted.items():
            if s.step_idx >= rec["born"]:
                rec["mass"].append(tot.get(r, 0.0))
    out, survived, faded = {}, 0, 0
    for r, rec in minted.items():
        m = rec["mass"]
        if len(m) < 2:
            continue
        first, last = m[0], m[-1]
        out[r] = {"born": rec["born"], "at_birth": round(first, 4),
                  "final": round(last, 4), "ratio": round(last / first, 3) if first > 0 else None}
        if first > 0 and last >= first * 0.5:
            survived += 1
        else:
            faded += 1
    return {"minted_count": len(minted), "tracked": len(out),
            "survived": survived, "faded": faded,
            "survival_rate": (survived / max(1, survived + faded)), "detail": out}


def split_candidates(particles, weight_quantile: float = 0.25, rank_margin: int = 0):
    """Particles whose accumulated support is not renewed by the present observation.

    Returns indices that are top-quantile by accumulated weight AND below median
    on the CURRENT likelihood rank. See split_weight_condition for why the
    original absolute-threshold spec was unbuildable.
    """
    ps = [(i, p) for i, p in enumerate(particles)
          if p.weight is not None and p.likelihood_rank is not None]
    n = len(ps)
    if n < 4:
        return [], 0, 0
    top_k = max(1, int(n * weight_quantile))
    by_w = sorted(ps, key=lambda t: -t[1].weight)
    heavy = {i for i, _ in by_w[:top_k]}
    mid = n // 2 + rank_margin
    disagree = {i for i, p in ps if p.likelihood_rank >= mid}
    return sorted(heavy & disagree), len(heavy), len(disagree)


def resolve_merge_threshold(pairs, percentile: float = 95.0) -> Optional[float]:
    """Merge threshold as a PERCENTILE of the run's own similarity distribution.

    Metric is Jaccard, chosen by a copy-history control: resample duplicates are
    known-same-hypothesis (identical text at birth, shared root), everything else
    known-different. Jaccard AUC 0.790 vs embedding cosine 0.716 -- cosine barely
    separates the classes here because every hypothesis concerns one agent in one
    scene in similar prose, compressing cosine into 0.85-1.00.

    A fixed cut repeats the N/2 mistake. The legacy cosine >= 0.90 absorbed 27 of
    40 KNOWN-DIFFERENT pairs, i.e. merge would have destroyed the root diversity
    3e exists to create.

    Default is deliberately conservative: in a system whose dominant failure is
    decay, a false merge costs far more than a missed one.
    """
    vals = sorted(v for v in pairs if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, int(percentile / 100.0 * len(vals)))
    return vals[max(0, idx)]


def nonlinearity_summary(steps: List[StepRecord], deadband=None, deadband_frac: float = 0.25) -> Dict[str, Any]:
    """The deliverable: strengthen, weaken, merge, split, or revise.

    Reversals are counted on ANCESTRAL MASS, not per-particle weights.
    Resampling resets weights to 1/N because the posterior is then encoded in
    multiplicity, so a weight trajectory cannot express "strengthened over the
    run" on any run where resampling fires. Ancestral mass is invariant to that
    encoding choice: a winner duplicated three times at 1/N holds 3/N.

    The deadband defaults to a fraction of the run's own median step and the
    resolved value is reported. A fixed absolute value silently decalibrates
    when the step-size distribution moves -- which Phase 3 does directly.
    """
    mass = ancestral_mass(steps)
    series = {r: [v for v in ser if v is not None] for r, ser in mass.items()}
    series = {r: ser for r, ser in series.items() if len(ser) > 1}
    if deadband is None:
        deadband = resolve_deadband(list(series.values()), deadband_frac)

    revs = {r: count_reversals(ser, deadband) for r, ser in series.items()}
    counts = sorted(revs.values())
    median = counts[len(counts) // 2] if counts else 0

    # Raw counts are confounded by SURVIVAL LENGTH: a root alive for 3 steps
    # cannot reverse more than once, so short-lived roots contribute structural
    # zeros and drag the median down. Observed per-root [0,0,0,0,0,4,6,9] --
    # a median of 0 meaning "half the roots died early", not "trajectories were
    # flat". Rate per alive-step is the comparable quantity, and it matters more
    # once Phase 3 starts moving root counts deliberately.
    rates = {r: (count_reversals(ser, deadband) / max(1, len(ser) - 1))
             for r, ser in series.items()}
    rate_vals = sorted(rates.values())
    long_lived = {r: v for r, v in revs.items() if len(series[r]) >= 5}
    ll = sorted(long_lived.values())

    by_particle: Dict[str, List[float]] = {}
    for s in steps:
        for p in s.particles:
            if p.raw_accumulator is not None:
                by_particle.setdefault(p.lineage_id or p.particle_id, []).append(p.raw_accumulator)
    acc_series = [v for v in by_particle.values() if len(v) > 1]
    acc_db = resolve_deadband(acc_series, deadband_frac)
    acc_revs = sorted(count_reversals(v, acc_db) for v in acc_series)

    ops: Dict[str, int] = {}
    for s in steps:
        for op in s.operators_fired:
            ops[op] = ops.get(op, 0) + 1
    n_resample = ops.get("resample", 0)

    return {
        "reversals_median": median,
        "reversal_rate_median": (rate_vals[len(rate_vals) // 2] if rate_vals else 0.0),
        "reversal_rate_max": (max(rate_vals) if rate_vals else 0.0),
        "reversals_median_long_lived": (ll[len(ll) // 2] if ll else 0),
        "long_lived_roots": len(ll),
        "root_lifetimes": {r: len(ser) for r, ser in series.items()},
        "reversals_max": max(counts) if counts else 0,
        "reversals_per_root": revs,
        "roots_with_reversal": sum(1 for v in counts if v > 0),
        "reversal_basis": "ancestral_mass",
        "deadband_resolved": deadband,
        "deadband_frac": deadband_frac,
        "accumulator_reversals_median": acc_revs[len(acc_revs) // 2] if acc_revs else 0,
        "accumulator_deadband_resolved": acc_db,
        "distinct_roots": len(series),
        "distinct_lineages": len(by_particle),
        "argmax_churn": argmax_churn(steps),
        "resample_segments": n_resample + 1,
        "merge_count": ops.get("merge", 0),
        "split_count": ops.get("split", 0),
        "anchor_revision_count": ops.get("anchor_revision", 0),
        "resample_count": n_resample,
        "perturb_count": ops.get("perturb", 0),
        "operators": ops,
    }

# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

class RunLogger:
    """Writes one JSONL of StepRecords plus a one-line summary to runs.jsonl."""

    def __init__(self, out_dir: str, run_id: str, meta: Optional[Dict[str, Any]] = None):
        self.out_dir = out_dir
        self.run_id = run_id
        self.meta = dict(meta or {})
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{run_id}.steps.jsonl")
        self.runs_path = os.path.join(out_dir, "runs.jsonl")
        self.steps: List[StepRecord] = []
        self._fh = open(self.path, "w", encoding="utf-8")
        self._t0 = time.time()
        self._step_t0 = self._t0

    def step(self, record: StepRecord) -> StepRecord:
        now = time.time()
        if record.wall_time_s is None:
            record.wall_time_s = now - self._step_t0
        self._step_t0 = now
        if not record.llm_calls:
            record.llm_calls = RECORDER.drain()
        self.steps.append(record)
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False, default=_fallback) + "\n")
        self._fh.flush()
        return record

    def close(self) -> Dict[str, Any]:
        summary = self.summary()
        self._fh.close()
        with open(self.runs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, default=_fallback) + "\n")
        return summary

    def summary(self) -> Dict[str, Any]:
        temps = [c.get("effective_temperature") for s in self.steps for c in s.llm_calls]
        temps = [t for t in temps if t is not None]
        like = [s.likelihood_ess for s in self.steps if s.likelihood_ess is not None]
        post = [s.posterior_ess for s in self.steps if s.posterior_ess is not None]
        out = {
            "run_id": self.run_id,
            "steps": len(self.steps),
            "wall_time_s": time.time() - self._t0,
            "total_llm_calls": sum(len(s.llm_calls) for s in self.steps),
            "truncated_calls": sum(1 for s in self.steps for c in s.llm_calls if c.get("truncated")),
            "parse_failures": sum(s.parse_failures for s in self.steps),
            "effective_temperatures": sorted(set(temps)),
            "median_likelihood_ess": _median(like),
            "median_likelihood_ess_norm": _median(
                [s.likelihood_ess_norm for s in self.steps if s.likelihood_ess_norm is not None]),
            "median_posterior_ess": _median(post),
        }
        out.update(nonlinearity_summary(self.steps))
        out.update(self.meta)
        return out


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _fallback(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def read_steps(path: str) -> List[StepRecord]:
    """Rehydrate StepRecords from JSONL, tolerating fields not yet wired up."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            parts = [ParticleRecord(**{k: v for k, v in p.items() if k in ParticleRecord.__annotations__})
                     for p in d.pop("particles", [])]
            rec = StepRecord(**{k: v for k, v in d.items() if k in StepRecord.__annotations__})
            rec.particles = parts
            out.append(rec)
    return out
