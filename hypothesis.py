import re
import random
import numpy as np
from typing import List
from rich import print
from rich.panel import Panel

from trace_log import new_particle_id


class HypothesisV3():
    def __init__(self, target_agent: str, contexts: List[str], perceptions: List[dict], text: str, weight: float, parent_hypothesis: 'HypothesisV3' = None, anchor: str = None, particle_id: str = None, raw_accumulator: float = None, lineage_id: str = None, standard: str = None, method: str = None):
        self.target_agent = target_agent
        self.contexts = contexts
        # self.context_history = context_history
        self.perceptions = perceptions
        self.text = text
        self.weight = weight
        # self.observation = observation
        self.parent = parent_hypothesis
        # Stable identity. Positional index is not identity: merge, split and
        # resample all reorder the population, and lineage has to survive that.
        # Two identities, deliberately distinct:
        #   particle_id : this instance at this step; fresh every step, so it
        #                 labels the edges of the lineage graph.
        #   lineage_id  : the trajectory this particle continues. Inherited on
        #                 propagation, so it is stable across steps.
        # Keying a weight series on particle_id yields one point per particle
        # and makes reversals and argmax churn unmeasurable -- every step looks
        # like a brand-new population.
        self.particle_id = particle_id or new_particle_id()
        self.parent_id = parent_hypothesis.particle_id if parent_hypothesis is not None else None
        # At initialization the anchor IS the root: each founding hypothesis
        # carries its own distinguishing commitment.
        if lineage_id is not None:
            self.lineage_id = lineage_id
        elif parent_hypothesis is not None:
            self.lineage_id = parent_hypothesis.lineage_id
        else:
            self.lineage_id = new_particle_id()
        self.split_child = False
        # Root of the ancestral tree. Survives resampling: a duplicated winner
        # gets a FRESH lineage_id (separate trajectory from here) but keeps its
        # root, so the mass of all its descendants stays attributable to one
        # ancestor. This is what makes strengthen/weaken observable across a
        # resample boundary -- see ancestral_mass() in trace_log.
        if parent_hypothesis is not None:
            self.root_id = parent_hypothesis.root_id
        else:
            self.root_id = self.lineage_id
        # The distinguishing goal-level commitment. Inherited on propagation,
        # replaced on split and on anchor revision, differentiated on resample.
        self.anchor = anchor if anchor is not None else (parent_hypothesis.anchor if parent_hypothesis is not None else None)
        # What would SETTLE the question for them -- the evidence they would
        # accept as showing they were wrong. Distinct from the anchor on
        # purpose: two people can hold the same aim and still disagree about
        # what counts as having met it, and that disagreement is invisible to
        # a representation that only records wants. Measured on the Bloomfield
        # purple dispute, that is exactly the axis the parties separate on --
        # Wolf judges a count by what is delivered to the grower, the
        # engineers by what the pipeline can reproduce -- and the filter had
        # to encode it as a status motive because it had nowhere else to put
        # it. Travels WITH the anchor: propagation inherits both, and anything
        # that founds a new root supplies a new one.
        self.standard = standard if standard is not None else (parent_hypothesis.standard if parent_hypothesis is not None else None)
        # METHOD IS A PROPERTY OF THE COMMITMENT, NOT OF THE TRAJECTORY.
        # A method is a way of GENERATING a commitment, so a particle that
        # generated nothing this step used no method:
        #   generated (seed, split child, mint) -> the method of the call
        #   inherited (propagate, resample copy, merge survivor) -> unchanged
        #   revived                             -> the ORIGINAL method, restored
        #                                          with the original root
        # Travelling with the anchor rather than the lineage is what makes the
        # label attributable: the audit asks which method FOUND a commitment,
        # and a lineage-borne label would credit whichever method seeded the
        # trajectory a mint later replaced.
        self.method = method if method is not None else (parent_hypothesis.method if parent_hypothesis is not None else None)
        # Unnormalized accumulated log-weight. The particle's own evidence
        # trajectory; the only series reversals may be counted on.
        self.raw_accumulator = raw_accumulator if raw_accumulator is not None else 0.0
        self.anchor_revisions = 0
        self.operators = []

    def note_operator(self, name: str):
        self.operators.append(name)

    def update_standard(self, new_standard):
        if new_standard:
            self.standard = new_standard

    def update_anchor(self, new_anchor: str, revision: bool = False, founds_root: bool = True,
                      standard: str = None, method: str = None):
        """Set the anchor. A NEW anchor founds a NEW root.

        ANCHOR IDENTITY == ROOT IDENTITY. The anchor is the particle's
        distinguishing commitment, so a new root carrying the parent's
        commitment would not be a new hypothesis, and a new commitment under
        the parent's root would be invisible to every diversity metric.
        Keeping them as parallel fields allows both contradictions.

        This collapses the four anchor lifecycle behaviours into one rule:
          new anchor  -> founds a root   (split, revision, anchored perturbation)
          otherwise   -> inherits        (propagation, resample duplicates, merge survivor)

        A resample duplicate is a COPY of a hypothesis, not a new one, so it
        keeps both. Regenerating diversity is the perturbation's job, which is
        why resampling and repair have cleanly opposed roles.
        """
        changed = new_anchor is not None and new_anchor != self.anchor
        self.anchor = new_anchor
        # A new commitment brings its own standard; without this the particle
        # keeps the settling condition of the commitment it just replaced.
        if changed:
            self.standard = standard
            # Deliberately asymmetric with `standard`, which clobbers on None.
            # A stale settling condition is actively WRONG -- it settles a
            # commitment the particle no longer holds. A stale method label is
            # merely uninformative, and revive_retired calls this without
            # knowing the method, so clobbering would erase the very
            # provenance revive exists to restore.
            if method is not None:
                self.method = method
        if changed and founds_root:
            self.root_id = new_particle_id()
        if revision:
            self.anchor_revisions += 1
            self.note_operator('anchor_revision')

    def update_accumulator(self, value: float):
        self.raw_accumulator = value

    def update_text(self, new_text):
        self.text = new_text

    def update_weight(self, new_weight):
        self.weight = new_weight

    def __repr__(self) -> str:
        return f"Text: {self.text} Weight: {self.weight}"

class HypothesesSetV3():
    def __init__(self, target_agent: str, contexts: List[dict], perceptions: List[dict], texts: List[str], weights, parent_hypotheses: List[HypothesisV3] = None, previous_ess: float = None, weight_details: dict = None, anchors: List[str] = None, accumulators: List[float] = None, lineage_ids: List[str] = None, methods: List[str] = None, **kwargs):
        self.target_agent = target_agent
        self.contexts = contexts
        self.perceptions = perceptions
        self.texts = texts
        self.weights = weights
        self.kwargs = kwargs
        # self.observation = observation
        self.previous_ess = previous_ess
        self.weight_details = weight_details
        self.lookahead_scores = kwargs.get('lookahead_scores', None)
        n = len(texts)
        anchors = list(anchors) if anchors is not None else [None] * n
        methods = list(methods) if methods is not None else [None] * n
        accumulators = list(accumulators) if accumulators is not None else [None] * n
        lineage_ids = list(lineage_ids) if lineage_ids is not None else [None] * n
        if len(lineage_ids) < n:
            lineage_ids = lineage_ids + [None] * (n - len(lineage_ids))
        if len(anchors) < n:
            anchors = anchors + [None] * (n - len(anchors))
        if len(methods) < n:
            methods = methods + [None] * (n - len(methods))
        if len(accumulators) < n:
            accumulators = accumulators + [None] * (n - len(accumulators))
        if parent_hypotheses is not None:
            parents = list(parent_hypotheses)
            if len(parents) < n:
                parents = parents + [None] * (n - len(parents))
            self.hypotheses = [
                HypothesisV3(target_agent, contexts, perceptions, text, weight, parent_hypothesis=parent,
                             anchor=anchor, raw_accumulator=acc, lineage_id=lid, method=meth)
                for text, weight, parent, anchor, acc, lid, meth in zip(texts, weights, parents, anchors, accumulators, lineage_ids, methods)
            ]
        else:
            self.hypotheses = [
                HypothesisV3(target_agent, contexts, perceptions, text, weight, parent_hypothesis=None,
                             anchor=anchor, raw_accumulator=acc, lineage_id=lid, method=meth)
                for text, weight, anchor, acc, lid, meth in zip(texts, weights, anchors, accumulators, lineage_ids, methods)
            ]
        # ANCHOR == ROOT, enforced at founding too. Two founding particles that
        # carry the SAME commitment are one hypothesis explored twice, not two,
        # and must share a root -- otherwise the initial root count is inflated
        # by however many anchors the extractor duplicated. Observed with
        # belief-level seeding: 5 distinct anchors across 6 particles founded 6
        # roots, overstating diversity by one before the run even started.
        if parent_hypotheses is None:
            canon = {}
            for h in self.hypotheses:
                key = (h.anchor or '').strip().lower()
                if not key:
                    continue
                if key in canon:
                    h.root_id = canon[key]
                else:
                    canon[key] = h.root_id
        self.anchors = [h.anchor for h in self.hypotheses]

    @property
    def particle_ids(self):
        return [h.particle_id for h in self.hypotheses]

    @property
    def lineage_ids(self):
        return [h.lineage_id for h in self.hypotheses]

    @property
    def root_ids(self):
        return [h.root_id for h in self.hypotheses]

    @property
    def methods(self):
        return [getattr(h, 'method', None) for h in self.hypotheses]

    def update_anchors(self, new_anchors, revision: bool = False, founds_root: bool = True):
        self.anchors = list(new_anchors)
        for hypothesis, anchor in zip(self.hypotheses, new_anchors):
            hypothesis.update_anchor(anchor, revision=revision, founds_root=founds_root)

    def update_accumulators(self, values):
        for hypothesis, value in zip(self.hypotheses, values):
            hypothesis.update_accumulator(value)

    @property
    def accumulators(self):
        return [h.raw_accumulator for h in self.hypotheses]

    def update_weights(self, new_weights):
        self.weights = new_weights
        for hypothesis, weight in zip(self.hypotheses, new_weights):
            hypothesis.update_weight(weight)

    def dump(self):
        """
        Dump the hypotheses set to a dictionary.
        """
        hypotheses_details = []
        for hypothesis in self.hypotheses:
            if 'details' in hypothesis.__dict__:
                details = hypothesis.details
            else:
                details = None
            hypotheses_details.append({
                'details': details
            })
        weights = self.weights
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        return {
            'target_agent': self.target_agent,
            'contexts': self.contexts,
            'perceptions': self.perceptions,
            'texts': self.texts,
            'weights': list(weights),
            'kwargs': self.kwargs,
            'previous_ess': self.previous_ess,
            'details': hypotheses_details,
            # Previously dropped on the floor: the verdicts and reasonings were
            # computed every step and never reached disk, so there was no way to
            # audit what the evaluator actually said.
            'weight_details': self.weight_details,
            'particle_ids': self.particle_ids,
            'lineage_ids': self.lineage_ids,
            'root_ids': self.root_ids,
            'parent_ids': [h.parent_id for h in self.hypotheses],
            'anchors': [h.anchor for h in self.hypotheses],
            'standards': [getattr(h, 'standard', None) for h in self.hypotheses],
            'methods': self.methods,
            'accumulators': self.accumulators,
        }

    def __iter__(self):
        return iter(self.hypotheses)

    def __getitem__(self, idx: int):
        return self.hypotheses[idx]

    


def systematic_resample(weights) -> List[int]:
    """Systematic resampling. Replaces multinomial `random.choices`.

    Multinomial draws N independent samples, so a particle can be missed
    entirely by chance: P(never drawn) = (1-w)^N. At N=8 that is 34% for a
    particle holding exactly 1/N -- which is the state of EVERY particle
    immediately after a resample -- and still 17% for one holding 20% of the
    belief mass. Measured in real runs: lineages holding 0.212, 0.208 and 0.182
    disappeared at a resample for no reason connected to evidence.

    Systematic resampling walks the weight CDF in 1/N steps from a single
    random offset. Any particle with weight >= 1/N is GUARANTEED at least one
    copy, and the variance of the copy count is far lower. Simulated over 2000
    resamples with the top particle at w=0.21: multinomial lost it 14.7% of the
    time, systematic 0%.

    This is the standard choice in the SMC literature for exactly this reason.
    """
    w = [float(x) for x in weights]
    n = len(w)
    total = sum(w)
    if n == 0 or total <= 0:
        return list(range(n))
    w = [x / total for x in w]
    # one draw per 1/n interval, offset by a single uniform sample
    u0 = random.random() / n
    out, cum, j = [], w[0], 0
    for k in range(n):
        u = u0 + k / n
        while u > cum and j < n - 1:
            j += 1
            cum += w[j]
        out.append(j)
    return out


def resample_hypotheses_with_other_info(hypotheses: HypothesesSetV3, ess: float) -> HypothesesSetV3:
    """
    Resample the hypotheses based on the weights.
    """
    print("Resampling hypotheses @x@x@x@x@x@x")
    for h, w in zip(hypotheses.texts, hypotheses.weights):
        print(Panel(h, title=f"Weight: {w:.2f}", style="purple"))

    resampled_idxs = systematic_resample(hypotheses.weights)
    target_agent = hypotheses.target_agent
    texts = [hypotheses.texts[idx] for idx in resampled_idxs]
    weights = np.ones(len(texts)) / len(texts)
    # The resampled particle becomes the parent -- not its parent. Taking
    # `.parent` here froze the chain at its original depth, so lineage never
    # grew and split siblings were indistinguishable from independent arrivals.
    parents = [hypotheses.hypotheses[idx] for idx in resampled_idxs]
    anchors = [hypotheses.hypotheses[idx].anchor for idx in resampled_idxs]
    accumulators = [hypotheses.hypotheses[idx].raw_accumulator for idx in resampled_idxs]
    weight_detail_prompts = [hypotheses.weight_details['prompts'][idx] for idx in resampled_idxs]
    weight_detail_predictions = [hypotheses.weight_details['reasonings'][idx] for idx in resampled_idxs]
    weight_details = {'prompts': weight_detail_prompts, 'reasonings': weight_detail_predictions} #, 'evaluations': weight_detail_evaluations}

    resampled_hypotheses = HypothesesSetV3(target_agent, hypotheses.contexts, hypotheses.perceptions, texts, weights, parent_hypotheses=parents, previous_ess=ess, weight_details=weight_details, anchors=anchors, accumulators=accumulators)

    # Flag every copy after the first so viz can hatch duplicates and the
    # frozen-duplicate failure stays visible if it returns.
    seen = set()
    for hypothesis, idx in zip(resampled_hypotheses.hypotheses, resampled_idxs):
        dup = idx in seen
        hypothesis.resample_duplicate = dup
        if dup:
            # A duplicate is a separate trajectory from here on, so it must not
            # share the source's lineage_id -- otherwise two particles write to
            # one weight series and the series is meaningless.
            hypothesis.lineage_id = new_particle_id()
        hypothesis.note_operator('resample')
        seen.add(idx)

    return resampled_hypotheses

def compute_ess(hypotheses: HypothesesSetV3) -> float:
    """
    Compute the effective sample size of the hypotheses.
    """
    ess = 1 / np.sum(np.square(hypotheses.weights))
    return ess

def extract_question(text: str) -> str:
    """
    Extract the question from the text.
    """
    # Extract the question from the text. Extract the line that starts with "Question:" using regex.
    match = re.search(r'Question:(.*)', text)
    if match:
        question = match.group(1).strip()
        rest = text.split(question)[1].strip().split("Answer:")[0].strip().replace("\n", " ")
        question = question.replace("Answer yes or no.", "").strip()
        if rest != "":
            question = f"{question} {rest}"
        return question
    else:
        return "none"

