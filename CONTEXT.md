# Context briefing: long-horizon theory-of-mind tracing on real work transcripts

A self-contained orientation for someone joining a brainstorm about this project.
No repo access assumed. Everything numeric here is measured and reproducible from
logs on disk; the two primary evidence documents are `PRECHECKS.md` (numbered
findings P-1 … P-14) and `POSTMORTEM.md` (the map over them).

---

## 1. What this is, in one paragraph

A fork of the COLM 2025 paper repo **"Thought Tracing: Hypothesis-Draiven
Theory-of-Mind Reasoning for Large Language Models"** (Kim, Sclar, Zhi-Xuan,
Ying, Levine, Liu, Tenenbaum, Choi — arXiv 2502.11881). The upstream system is
an LLM-driven **particle filter over hypotheses about what a person believes and
wants**, evaluated on synthetic ToM benchmarks (ToMi, FANToM, BigToM, MMToM-QA).
We took that machinery and pointed it at something it was not built for: **the
long-term mental state of real named employees, on real Slack transcripts, with
no gold labels.** The fork is now roughly 15,000 lines ahead of upstream, almost
all of it instrumentation, operators, audits and evaluation scaffolding.

The honest summary of the last stretch of work is: **the mechanics all work and
are verified; the phenomenon we were trying to detect has not been detected by
any of five independent designs; and we have localised why.** That localisation
is the most valuable thing in the repo and is the natural starting point for a
brainstorm.

---

## 2. The goal, and the motivating case

Model a person's **standing** mental state — motives, commitments, what they'd
accept as settling a question — accumulated across an entire corpus of their
work traffic, not just within one conversation.

The concrete case that drove the design is a real thread, `blueberry_size`, at an
agricultural-data company (internally: the "Bloomfield" corpus):

> Two groups argue about "size" for weeks. One side means **commercial caliber
> grading for export**. The other means **green-berry sizing for harvest
> timing**. Both say "size", both surround it with millimetres and berry
> categories, and **neither side ever notices they are discussing different
> things.**

That is the target phenomenon: a **referent split** — a shared term used for two
different purposes — surfacing as a detectable misalignment between two people's
traced mental states. It is a good target because it is (a) real, (b) costly, and
(c) invisible to anything that reads topic labels or keywords.

The planned architecture was a **two-timescale filter**: a slow *standing* board
per person that survives across the whole corpus, plus fast *episodic* boards
that open when two people diverge — so a misalignment could surface without
evicting the long-term picture.

---

## 3. How the system actually works

### 3.1 The core loop

Per traced person, per conversational turn (a "step"):

1. **Perception tracking** — what did this person plausibly see/hear by now?
2. **Propagation** — each of N hypotheses (default N=4–8) is advanced given the
   new turn.
3. **Scoring** — an LLM rates how likely each hypothesis makes the *observed next
   action*. Three modes: `rank` (force a strict total order), `independent`,
   `allocation` (distribute 100 points). A **null/baseline hypothesis** can be
   included, so the filter can discover that *no* live hypothesis explains the
   turn ("surprise").
4. **Reweighting** — `w_t ∝ w_{t-1}^α · L_t^β`, α=0.85 on every run on disk
   (prior half-life ≈ 4.3 steps), plus an ε floor so a null-level hypothesis
   survives, plus `--baseline-blend 0.15` uniform mixed back in.
5. **Population operators** fire conditionally: `resample` (systematic, below
   ESS threshold), `perturb` (mint a new hypothesis, optionally triggered by
   surprise), `split` (fork a heavy-but-poorly-ranked hypothesis), `merge`
   (collapse hypotheses that are one aim), `expand`, `expire` (retire a
   persistently weak hypothesis, optionally into a revivable cache).

### 3.2 The representation — the part worth understanding

Each particle carries more than a prose belief. The fields that matter:

| Field | What it is |
|---|---|
| `anchor` | The **commitment**: what the person is trying to achieve, in their own terms, ≤8 words. This is the particle's identity — roots are keyed on it. |
| `standard` | **What would settle the question for them** — the evidence they'd accept as showing they were wrong. |
| `method` | Which generation technique proposed it (see below). |
| `lineage_id` / `root_id` / `particle_id` | Three distinct identities so that merge/split/resample reordering doesn't destroy the ability to measure churn, ancestral mass and survival. |

**`standard` is a deliberate design bet.** Two people can share an aim and still
disagree about what counts as having met it, and that disagreement is invisible
to a representation that only records wants. On the purple-threshold dispute,
that is exactly the axis the parties separate on — one judges a count by what is
delivered to the grower, the other by whether an instrument reproduces it. Prompt
variant **v4** forces `standard` into four reserved categories:
`COLLEAGUE / CUSTOMER / MEASUREMENT / DOCUMENT`.

**Commitment validity is enforced**, because unconstrained minting produced
conversational moves rather than aims: once split started firing, 23 of 42
anchors in one run came back as "Ask Zuko to leverage Fire Nation intelligence
networks…" — the next thing to say, not a goal.

### 3.3 Twelve generation methods

`methods.py` implements twelve structured analytic techniques as swappable prompt
fragments, appended (never replacing) at four generation sites
(`seed`, `perturb`, `split`, `standard`):

`anomaly`, `role`, `silence`, `devil`, `ach` (competing hypotheses), `assume`,
`premortem`, `presentation`, `crystal`, `backcast`, `signpost`, `analogy`.

The reason they exist is a diagnosed structural gap: **every hypothesis source in
the filter is a *reader*, not a *generator*.** It sees one transcript and reports
what the transcript says. That works when the motive is spoken and fails when it
is not — and in a Slack work thread, nobody states why they want a parameter. So
the honest summary of what someone's messages *do* is "define numerical
thresholds", which describes the message and not the reason for sending it. Each
method is a device for **proposing a cause that was never said**. Method is
tagged on every hypothesis because *method predicts failure mode*.

### 3.4 Two external-prior mechanisms

- **`profiles.py` / `--character-profile`** — an external baseline character
  profile (seat, stake, pressure, influence structure) rendered into the seeding
  prompt. Represents "the understanding you walk into the room already holding."
  Seed-only, on purpose, so its entire measurable effect is the founding set and
  what survives from it. Carries a `read_confidence` that is **never** routed
  into weights (filter weights are normalised posteriors over a live population;
  an external confidence is on nobody's scale).
- **`memory.py` / `--memory`** — a durable per-person commitment store that
  survives between runs and between processes. Holds both *live* particles at
  run end and *retired* ones with peak weight, root, method and exit path.
  Config-isolated, atomic writes, exact-string keys.

### 3.5 Hard rules the codebase enforces (each from a specific measured failure)

- **The default path is sacred.** With every flag off, every prompt is
  byte-identical to before these modules existed. All 100+ logged findings are
  against those prompts; one character of drift destroys more evidence than any
  feature generates.
- **A stored or external weight is never a prior.** Restored particles enter at
  uniform weight.
- **Never compare two runs by a per-run median.** Mint birth mass spans 0.12–2.18
  of fair share *within a single run*. Reading a two-seed ordering of medians as
  an effect is how three interventions in a row were each briefly believed to
  work. Pool the events, permute the labels, report a p.
- **A variant is an improvement only if it raises the real score WITHOUT raising
  the control.** Three "culture detectors" died for lack of this.
- **Thresholds cannot answer "same aim".** Over 22k logged pairs,
  identical-commitment pairs sit at median Jaccard 0.397 against 0.208 for
  different-commitment pairs. "Avenge her mother" vs "Forgive her mother's
  killer" scores the same as a synonym pair. So `merge_equivalent_anchors` uses a
  **model call**, not a threshold.

---

## 4. Data and evaluation setup

Nine narrative/work corpora in `data/musing/`, exported in FANToM chat format
(`Name: utterance` lines): `bloomfield` (the real Slack corpus — 402 sets / 2,190
turns), `oppenheimer`, `odyssey`, `atla`, `lbj`, `boeing`, `agzen`, `exyn`,
`synthetic`.

Crucially, **these corpora have no manufactured belief asymmetry.** FANToM
engineers one (a speaker leaves the room, information is exchanged, they return).
Inside one real set, every speaker hears every line, so a first-order belief
question over a single set has a trivial answer. The asymmetry that *does* exist
is **between** conversations, which is what `<corpus>_gaps.json` reports:
per channel, each speaker present → absent → present again, plus the sets they
missed. Stitching `[last_present] + [missed…] + [rejoins_at]` yields a "gold"
context with genuinely inaccessible information.

**There are no gold labels for the thing we actually care about** (which turns
carry the referent split). That absence is the single largest problem in the
project — see §7.

On disk: ~115 run logs, 472 JSONL step streams, 6,160 step transitions available
for offline replay. Nine offline test files (no LLM calls) guard the invariants;
`memory.py` alone has 36 tests. A large family of audit scripts replay logs
offline so questions can be answered without new API spend: `audit_ties`,
`audit_vacuity`, `audit_margins`, `audit_revival`, `audit_profile`,
`audit_methods`, `sweep_alpha`, `referent_split`, `verify_memory`,
`validate_profiles`, `replay_likelihood`.

---

## 5. What was tried to detect the misalignment — and how each failed

Five designs, two independent layers, one conclusion.

| # | Design | Result | Finding |
|---|---|---|---|
| 1 | Episode trigger from the **mass** distribution over v4 standard categories | Measures a dyad *trait*, not an episode. Separation is constant on a stable board, so the persistence condition is free. | P-8 |
| 2 | Episode trigger from **likelihood margins** instead of mass | Margins move only 1.1–1.5× as much as mass, are heavily quantised, and the argmax sits in one bucket 44 times in 52. | P-8 |
| 3 | Episode trigger from the **undefined-lean rate** (no commitment beats the null) | Lift signs **disagree by party**: +0.16 / −0.18 / −0.07 against a 0.09 floor. The control dyad beat the test dyad. | P-10 |
| 4 | **Transcript-side** detector: compare the words each party puts around a shared term, with a permutation null. No board, no LLM. | Raw version detected writing *style* (test 47% vs control 48%). Normalised, the target term `size` ranked **39 of 68** — *below* the median term. | P-11 |
| 5 | Raise **α** (the prior's exponent) to stabilise the board | Churn went **up** at every α, all within noise. Seed spread at α=0.98 was 43.3 per 100 — larger than the effect being chased. | P-14 |

Design 4 is the important negative result, because it doesn't touch the filter at
all. The conclusion it licenses is a statement about the **phenomenon**, not about
the detectors:

> The misalignment is a difference in **what a measurement is FOR**. Purpose
> leaves no co-occurrence footprint. Both parties say "size" surrounded by
> millimetres, categories and blueberries. It is not in the board's fit, not in
> the standard distribution, and not in vocabulary.

Two further things did not survive contact with unselected data:

**The v4 standard axis is not a personality trait.** On a hand-picked 6-set span,
one party looked strongly `CUSTOMER` (0.65) and separated from another at total
variation 0.70 — *stable across three seeds*. On 42 unselected sets from the same
topic, his `CUSTOMER` mass is 0.00–0.19 and the separation drops to 0.22, while
the **control** pair rises to 0.31. The same person moved more across spans
(TV 0.69) than two different people differed within one span (0.36), and the
ordering of pairs inverts. **The measurement is stable across seeds and unstable
across spans** — and seed spread was the floor every existing control checked
against. Span variation is ~10× larger and was never checked.

**Commitments do not recur across runs.** Two runs, same person, same config,
adjacent spans, run as separate processes: 44 commitments, **zero** exact
overlap, zero stem overlap, zero pairs clearing a 0.6 Jaccard floor. The one
near-match is "Ensure labeling quality" / "Ensure clear labeling" — the same
commitment to a human reader, invisible to any lexical rule. So the memory store
is a correct **log** and consolidates nothing.

---

## 6. The one finding that explains the rest

The filter's leader is unstable, and the instability is **upstream of everything
that was built.**

```
argmax churn: 55.6 per 100 steps (median over 52 runs; ZERO runs at zero)

  operators account for            <= 30%   (P-13)
  reweighting accounts for            70%   (P-13)
  ...and inside that, the SCORER picks a different
     best hypothesis on 72% of steps           (P-14)

likelihood rank movement: 1.79 places/step   (random shuffle = 2.7)
weight     rank movement: 1.03 places/step   (population of 8)
```

Attribution detail (6,160 transitions, no new LLM calls — `operators_fired` plus
the `pre_operator_particles` snapshot already in the log):

```
operators_fired    steps  P(churn)   lift
(none)              3864     30.1%  -11.0%
merge               1141     86.9%  +45.8%
split                903     87.7%  +46.6%
expand               302     86.8%  +45.7%
resample             141     53.9%  +12.8%
perturb              800     37.2%   -3.8%
expire               534     37.1%   -4.0%
```

Note `perturb` and `expire` — the two operators usually blamed for destroying
hypotheses — are *below* baseline. `merge`/`split`/`expand` nearly guarantee churn
when they fire but only fire on ~a third of steps.

**The weight update is doing its job**: it damps 1.79 places of likelihood
movement down to 1.03. But you cannot make a stable leader out of an input that
reshuffles at 72%, and pushing the damping harder only blunts surprise detection
(responsiveness fell from 5.3 to 2.7 at α=0.90).

So **stratification, cross-run memory and α all sit downstream of the problem.**
That is why five reasonable designs each came back null.

This is consistent with an earlier finding (`FINDINGS.md` §4, recoverable at
`git show a84d09a^:FINDINGS.md`): on these corpora "the work of belief tracking is
done upstream of the particle population". Rank-first scoring raised Kendall τ
from 0.687 to 0.926 — but **τ is rank *correlation*, and an argmax can flip
freely while τ stays high.** It does.

### A related structural finding

A large block of the population is **weight-identical** — invisible to every rule
that ranks by weight (expiry, resampling, perturbation triggers, slot choice for
a mint). The weakest half is exactly tied on **55% of surprise steps**; the
largest weight-identical block is a median 17% of the population, rising to 43%
on tied steps, and it **grows** as a run proceeds (13% → 20%, in 115 of 187
runs). Two obvious causes were ruled out: the ε floor (`mass_moved_by_floor` is 0
on 90% of tied steps) and resample uniformity (no resample fired on any of them).
This bounds a whole class of intervention — three separate rules for choosing
which weak particle a mint replaces were built, measured and removed after this.

---

## 7. The label problem

Every signal design above was scored against a positive region that is
**contaminated**. The six hand-picked "color case" sets turn out to be mostly
Agrovision hand-counts and a scrum blog link, not the 2-Purple dispute. And the
corpus's topic labels are **thread-level, not content-level**: the whole corpus
carries exactly two topic tuples, and a set that is pure ops noise ("turn it off
and back on and connect that blue cable") is still labelled `blueberry_size`.

With a bad label **you cannot distinguish "this design is bad" from "this label
cannot see it."** This was routed around twice and both times the result was
uninterpretable.

Hence the standing prerequisite: **Stage 0 labelling** — hand-label the
split-bearing turns in `blueberry_size`, two labellers, κ reported, frozen before
any run. Roughly a day of work. It is the precondition for every other question
here.

---

## 8. What is live and verified

- **`memory.py` + wiring** — durable per-person store: cross-process
  accumulation, config isolation, atomic writes, no stored weight reachable as a
  prior, restored commitments marked so they aren't mistaken for something the
  filter found on its own. A correct log. (36 tests)
- **`trace_log.py`** — per-step JSONL instrumentation. `margins` and
  `lean_undefined` are now logged; they were computed on every baseline-scored
  step and thrown away, which is why no run on disk could be replayed to sweep a
  threshold.
- **`profiles.py` + `validate_profiles.py`** — external priors, plus the
  non-circular way to check them: a profile cannot be validated by output it
  produced (a target handed *someone else's* record lands 0.08 from that record —
  inside the same-seed noise floor). So each profile's falsifiable claim is
  scored against **record-only runs that never saw the profile**. First run:
  2 of 3 profiles SUPPORTED, 1 CONTRADICTED — and the contradicted one had the
  *highest* stated confidence (0.88). **Stated confidence does not predict
  accuracy**, which is why validation, not a confidence gate, should drive
  suppression.
- **The audit/replay suite** — reproduction path for P-11, P-12, P-14, offline.
- **`motive-auditor`** — a blinded LLM-judge agent that classifies each traced
  commitment as RESTATEMENT / MOTIVE / UNSUPPORTED against only the turns the
  filter actually saw.

**Currently uncommitted work in progress**: memory wiring into the tracer
(`_restore_memory`, `_checkpoint_memory`, `_persist_memory`), the profile
confidence gate, and the `settles_mode` injection into the STANDARD prompt.

---

## 9. Known-dead. Do not rebuild.

- Episode detection from board fit, from likelihood margins, or from standard-category distributions.
- Lexical matching for cross-run commitment identity.
- α (or any weight-update parameter) as a stability lever.
- `memory.consolidate()` and its thread-adjacency machinery — proven unable to fire.
- The horizon / two-timescale stratification machinery — unreachable once episode detection closed, and it targets at most 30% of the churn anyway.
- Any retry of a signal design **against the current labels**.

One non-obvious result worth keeping in mind: **hedging prompt wording does not
make a model hold something lightly.** A `test` mode that put the transcript
record first, ranked an external note as weaker, and admitted it only on silence
*increased* adoption of the note (0.22 vs 0.08). Presence or absence of the text
is the only lever measured to work.

---

## 10. Open questions — the actual brainstorm surface

**The blocking question, upstream of everything else:**

> Is **per-step comparative ranking of eight similar hypotheses** a well-posed
> task for an LLM at all? Or does the filter need a fundamentally different
> evidence signal?

The scorer reshuffles its top pick on 72% of steps and moves 1.79 of a possible
2.7 rank places per step. Everything downstream — stability, memory,
stratification, episode detection — is blocked on this. Candidate directions,
none tested:

- Score over **fewer, more distinct** hypotheses (is the difficulty that eight
  near-paraphrases are genuinely indistinguishable?).
- Score **pairwise** with aggregation instead of forcing a strict total order.
  Related: on a surprise step *every* commitment lost to the null, so any
  ordering among the failures may be confabulated — `audit_margins.py` tests
  exactly this and it is the cleanest available probe of whether the scorer's
  order is real.
- Change the evidence unit: score against a **window** of turns rather than a
  single next action, or against something other than next-action likelihood.
- Accept churn and measure something churn-invariant instead (ancestral mass,
  root survival) — i.e. stop treating "a stable leader" as the deliverable.

**Second, and cheaper:**

> **Semantic matching for commitment identity** — embeddings, or an LLM judge
> asked whether two commitments are the same aim.

This is the only untried item with a clear expected payoff. `merge_equivalent_anchors`
already uses a model rather than a threshold *inside* a run, for precisely this
reason; the cross-run store simply never inherited it. Note it fixes only the
"nothing recurs" finding, not the scorer problem underneath.

**Third, the representational question the negative results point at:**

> If the misalignment is a difference in **what a measurement is FOR**, what
> representation could carry *purpose*?

`standard` ("what would settle this for them") was the bet on this, and it
behaves *correctly* while simply not being a dispositional property of a person —
it faithfully tracks what the conversation is about. The honest reading is that
purpose is a property of a **person × task**, not of a person, and the
representation has no slot for that. This is wide open and is probably the most
interesting thing to think about from scratch.

**Fourth, a measurement-design question:**

> The vacuity problem: nothing in the filter ever challenges a persistent winner.

`split` fires on heavy AND below-median rank. A commitment that is too *general*
fails that test in the wrong direction — being compatible with whatever happens,
it ranks near the top every step and is never eligible to be split. Screened over
logged runs, roots like "Protect someone from harm" and "To appear generous" hold
30–47% of mass for 34–60 steps at mean rank ≈1 and are split candidates on under
3% of their steps. But a root that ranks first every step is **either vacuous or
simply right**, and the logs cannot tell those apart. `audit_vacuity.py` proposes
the counterfactual — hold the slate and context fixed, substitute a **decoy
action** from a different step, re-score, and measure whether rank falls — but
this is a design question worth pressure-testing before spending on it.

---

## 11. Repo map

```
tracer.py          4,155 lines. The filter: scoring, operators, propagation, prompts.
hypothesis.py      Particle + population classes, resampling, ESS, identity fields.
trace_log.py       Per-step JSONL instrumentation. Nothing here imports tracer.
run_musing.py      Driver for the data/musing corpora. ~70 CLI flags = the arm space.
methods.py         The twelve generation methods as prompt fragments.
profiles.py        External baseline character profiles as a seed-time prior.
memory.py          Durable cross-run per-person commitment store.
restatement.py     Are commitments motives, or restatements? (echo / bound / redundancy)
agents/            Model backends: gemini, gpt, together, vllm.
data/musing/       Nine corpora + gaps + speakers + manifest.
musing_out/        ~115 run logs, 472 JSONL step streams.
audit_*.py         Offline replay audits (ties, vacuity, margins, revival, profile, methods).
eval_*.py          Evaluation harnesses (motive separation, profile arms, quality, method).
test_*.py          Nine offline test files, no LLM calls.
PRECHECKS.md       The evidence. Findings P-1 … P-14, with numbers.
POSTMORTEM.md      The map over PRECHECKS: what was tried, why it stopped.
README.md          Upstream paper README (benchmark eval commands, citation).
```

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **particle / hypothesis** | One candidate account of what the target believes and wants. |
| **anchor / commitment** | The ≤8-word aim that gives a particle its identity. |
| **standard** | What the target would accept as settling the question. |
| **root** | The founding ancestor of a lineage; mass is attributed to roots. |
| **argmax churn** | How often the highest-weighted root changes step to step. The headline instability metric. |
| **surprise step** | A step where the null hypothesis outranks every live commitment. |
| **lean_undefined** | All margins ≤ 0 — nothing beat the null. |
| **mint** | Creating a new hypothesis mid-run (via perturb, split or revival). |
| **fair share** | 1/n of the mass. Birth weights are always reported as a fraction of this, because n decays from 8 to 4–6 over a run. |
| **arm** | One configuration in a comparison (e.g. `none` / `infer` / `profile`). |
| **span** | The set of turns one run covered. Span variation ≫ seed variation — the central methodological lesson. |
| **referent split** | Two people using one term for two different purposes. The target phenomenon. |
| **Stage 0** | The frozen hand-labelling task that everything else is blocked on. |
