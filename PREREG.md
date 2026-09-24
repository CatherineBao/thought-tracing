# Pre-registration

Every phase writes its entry here **before it runs**, and the entry is
committed before the first call is made. An entry names four things and nothing
else is negotiable afterwards:

| | |
|---|---|
| **metric** | the single number the decision turns on, and the script that computes it |
| **pass threshold** | the value it must clear, fixed in advance, with the noise floor it is being compared against |
| **control arm** | what the same measurement reads on data the effect should not be in |
| **decision** | what happens on pass and on fail — both branches, written down |

The outcome is appended under the entry afterwards, with the predictions left
**unedited**, following `musing_out/meta/settles_prereg.md`.

This file exists because of one measured pattern: PRECHECKS records three
interventions in a row that were each briefly believed to work, and each was
read off a per-run median against a noise floor nobody had measured. The cost
of this file is ten minutes per phase. The cost of not having it is on the
record.

---

## Standing rules

These are not advice. Each one is a specific measured failure, and a phase that
breaks one is not reporting a result.

1. **Never compare two runs by a per-run median.** Mint birth mass spans
   0.12–2.18 of fair share *within a single run*. Pool the events, permute the
   labels, report a p.
2. **A variant is an improvement only if it raises the real score WITHOUT
   raising the control.** Three culture detectors died for lack of this; the
   transcript-side detector scored 47% test against 48% control.
3. **Span variation ≫ seed variation.** The v4 standard axis was stable across
   three seeds and inverted across spans (TV 0.69 for one person across spans,
   0.36 between two people within one). Seed spread was the floor every
   existing control checked against, and it is roughly a tenth of the right
   one. **Every threshold below is stated against a span floor.**
4. **The default path is sacred.** With every flag off, every prompt is
   byte-identical to before. All 100+ logged findings are against those
   prompts.
5. **A stored or external weight is never a prior.** Restored particles enter at
   uniform weight.
6. **Nothing tuned on dev is re-tuned after looking at test.** One look. The
   split is in `splits.json` and `make_splits.py --check` fails if it drifts.

---

## Frozen artifacts

| File | What it fixes | Verify with |
|---|---|---|
| `splits.json` | dev/test, corpora and eval vintage | `python make_splits.py --check` |
| `baseline_metrics.json` | the numbers every later phase is measured against | `python baseline_snapshot.py --check` |
| `logprob_backend.json` | the pinned scoring model and the checks it passed | `python confirm_logprobs.py --self-test` |

All three carry `git_commit` and regenerate deterministically. `--check` exits
non-zero on drift rather than rewriting, so a changed number is a failing
command and not something somebody notices later.

---

## 0.3 The pinned logprob backend — **NOT CONFIRMED**

| | |
|---|---|
| pin | `Qwen/Qwen2.5-7B-Instruct`, tokenizer `Qwen/Qwen2.5-7B-Instruct`, dtype `bfloat16` |
| why this one | ungated on the Hub (no licence click between a reader and a reproduction), 128k context via rope scaling — a noisy window is ~1,000 turns — served by vLLM with no custom path, and one 80GB card holds it in bf16 with room for the KV cache |
| status | **the capability is implemented and unconfirmed.** This machine has no `vllm`, no `torch`, no `transformers` and no GPU, and no endpoint is reachable. `confirm_logprobs.py --self-test` passes; the backend itself has never returned a number. |
| to confirm | `vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 131072` then `python confirm_logprobs.py --server http://HOST:8000/v1` |

**No Phase 2 entry may be filled in until `logprob_backend.json` exists with
`confirmed: true`.** Log-probs are not comparable across weights, tokenizer or
dtype, so a threshold tuned under one pin is meaningless under another; that is
why the pin lives in a file the confirmation writes rather than in this
sentence.

The confirmation is four checks, because a list of floats proves nothing:

- **aligned** — the tail carries a log-prob per continuation token and the join
  did not retokenize
- **responsive** — a plausible continuation outscores a scrambled one on the
  same prompt (a backend returning a constant, or the prompt's log-probs
  shifted by one, passes `aligned` and fails this)
- **ordered** — the same continuation scores higher after the context that
  licenses it than after one that does not
- **deterministic** — two identical requests return identical floats

---

## Power, stated before anything is scored

Pair accuracy against a 50% chance rate, exact binomial:

| pool | n | threshold for p ≤ 0.05 | best possible p |
|---|---|---|---|
| altered dev | 79 | 48 / 79 (60.8%) | — |
| **altered test** | **38** | **25 / 38 (65.8%)** | — |
| real dev | 6 | 6 / 6 | 0.0156 |
| **real test** | **4** | **unreachable** | **0.0625** |

**The four organic test pairs cannot reach significance even if every one is
correct.** They are pre-registered as a confirmatory read and may never be
reported as a measurement. This is a property of the vintage, not of any
detector: ten organic pairs exist in total.

**The two variants are not independent.** `minimal` and `noisy` versions of one
case share the bearing messages, so 38 × 2 is not 76 independent trials and
must never be pooled as such. Pre-registered in advance:

- **primary: `minimal`, 38 test cases, threshold ≥ 25.** The capability
  question — is it detectable when the bearing evidence is handed over.
- **secondary: `noisy`, the same 38 cases, threshold ≥ 25, reported always.**
  The retrieval question — is it findable inside ~1,000 turns of real traffic.

Both are reported in every result. A `minimal` win reported without its `noisy`
number is not a result.

---

## 0.1 The split — frozen

Two sections, two rules, two scopes, because they carry different risk.

**`corpora/`** — `data/musing` sets, cut into contiguous date spans. set_ids are
date-ordered within a corpus, so a contiguous date span is a contiguous id
range and the cut is checkable by eye.

| block | dev | test | cut |
|---|---|---|---|
| bloomfield / purple_threshold | 166 | 162 | 2024-04-25 |
| bloomfield / blueberry_size | 38 | 36 | 2024-11-28 |
| boeing | 28 | 24 | 2016-04-01 (4 undated sets unassigned) |
| oppenheimer | 32 | 23 | cumulative-turn midpoint of narrative order |

*Scope: unlabelled mechanics only* — churn, rank movement, tie blocks, scorer
probes, swapped-person controls. These sets carry no misalignment label, so
text shared with an eval case cannot inflate a detection score. **Do not score
a detector here.**

**`eval/`** — `altered_real_v1`, assigned **whole org at a time**.

| side | orgs | altered cases | real pairs |
|---|---|---|---|
| dev | bloomfield, agzen, aigen | 79 | 6 |
| test | exyn, flovision, flovision_solutions, musing_ai | 38 | 4 |

A finer cut was **measured and rejected**, not assumed: 224 within-org case
pairs share messages, the worst at 98.8%, and the noisy variant of one
bloomfield case reaches 204 of that corpus's 402 sets. Org is the smallest unit
that can be disjoint. Verified: **0 of 96,025 messages appear on both sides.**

Which org goes where is not a coin flip. bloomfield is the only eval org this
repo has worked on — 140 of 248 logged runs, five detector designs, every
finding in PRECHECKS — so it is **dev**, because prior exposure on the test side
buys optimism. agzen and exyn both have corpora in `data/musing` and neither has
a single logged run; exyn is test because it carries two real cases to agzen's
one, and the test side is the one short of real cases.

**Known limitation, stated now:** the test side is four orgs but only one of
them (exyn) carries an organic case. Org-level generalisation of the organic
signal is untested and cannot be tested with this vintage.

---

## 0.4 The baseline — frozen

`baseline_metrics.json`, recomputed from 212 runs on disk. The
production-config pool (α=0.85, n=8, rank scorer, no baseline scorer; 75 runs,
1,754 transitions) is the like-for-like comparison against PRECHECKS:

| metric | PRECHECKS | recomputed here |
|---|---|---|
| argmax churn / 100 steps | 55.6 | **56.41** (mean 56.37, range 23.1–81.5, **0 runs at zero**) |
| likelihood rank movement | 1.79 | **1.70** places/step |
| weight rank movement | 1.03 | **1.00** places/step |
| random-shuffle ceiling | ~2.7 | **1.94** (this pool's median population is 6, not 8) |
| likelihood top-1 changes root | 72% | **70.4%** |
| biggest weight-identical block | 17% | **16.7%** of the population |
| weak half tied on a surprise step | 55% | **50.8%** (over the 54 runs that log `surprise` — **no production run does**) |

Operator attribution over 1,754 transitions reproduces P-13 independently:

```
              steps   P(churn)   lift
(none)          966      45.3%  -11.9%
merge           533      84.2%  +27.0%
split           382      84.8%  +27.6%
expand          183      84.2%  +26.9%
resample          9      66.7%   +9.4%
expire          238      51.3%   -6.0%
perturb          78      41.0%  -16.2%
```

`perturb` and `expire` — the two operators usually blamed — are **below**
baseline. Vacuity screen: **176 suspects of 819 commitments** over 168 of 233
tracer dumps.

Differences from PRECHECKS are **reported, not reconciled**. PRECHECKS computed
these over a run set it did not record; tuning the script until the numbers
match would be fitting the measurement to the claim. What matters downstream is
that a later phase compares against *this* file, recomputed the same way.

**The shuffle ceiling is the number to watch.** At a median population of 6 the
ceiling is 1.94 and the scorer moves 1.70 — **87% of the way to a random
reshuffle**, worse than the 1.79/2.7 = 66% that PRECHECKS quotes at n=8. Any
claim that a change stabilised the scorer must be stated as a fraction of the
ceiling for its own population, never as a bare number of places.

---

## 0.5 The swapped-person control — built

`swapped_control.py`. Given a run and a step, returns a slate traced for a
**different person on the same span, same configuration, same step**. Coverage
on the logs as they stand: **121 of 226 runs** have a control, 100 of them at
tier 1 (same arm).

It **refuses rather than approximates**, and every refusal names the constraint
that refused it. A control that is a bit off on the span or the configuration is
worse than no control: it turns a null into an ambiguous result, and PRECHECKS
has two of those already. Specifically:

- a run with a different `n_hypotheses` is never returned — rank is bounded by
  population size
- a control from the other side of `splits.json` is refused
- two people on one span have different step counts (41 against 37; 15 against
  6), so an out-of-range step refuses instead of clamping to the last one, and
  `align="fraction"` reports the step it actually landed on

14 offline tests in `test_swapped_control.py`, no LLM calls.

---

## Template for a phase entry

```markdown
## Phase N — <one line>

Written BEFORE the arm was launched. Baseline on disk: <number, from
baseline_metrics.json or splits.json>.

**What is being changed.** <the one thing>

**Metric.** <number> computed by <script>.

**Pass threshold.** <value>, against a span floor of <value> measured by
<what>. Not a seed floor.

**Control arm.** <what> — and what the metric reads there now.

**Decision.** On pass: <what happens>. On fail: <what happens>.

**Predicted failure mode.** <the way this could produce a number that looks
like a result and is not>, and the arm that would catch it.

---

# OUTCOME (written after scoring; predictions above unedited)
```

---

## Phase 2 — logprob scoring — **STUB, DO NOT RUN**

Blocked on `logprob_backend.json` showing `confirmed: true`. Recorded here so
the blocking dependency is visible, not as a filled entry.

The question it is aimed at is the blocking one in CONTEXT §10: the scorer
picks a different best hypothesis on 70.4% of steps and moves 87% of the way to
a random reshuffle, before any weight update touches it. A continuation
log-prob is a different evidence signal for the same quantity — it asks the
model for P(observed action | hypothesis) directly rather than asking it to
impose a strict total order on eight near-paraphrases.

Before this runs, fill in: the metric (likelihood rank movement as a fraction
of the shuffle ceiling, against **1.6972 / 1.9444 = 0.873**), the pass threshold
with its span floor, the control arm (`swapped_control.py` — the same step
scored against a slate traced for a different person, which should *not*
separate), and both decision branches.

---

## Phase CP — choice-point forecasting

Written BEFORE any extraction ran. No number below has been seen.

**What is being changed.** Nothing in the filter. This is a different evidence
unit: instead of scoring hypotheses against *the observed next action* at every
turn, hypotheses are scored against *what a person did at a moment where they
could have done otherwise*. Choice points are extracted by an LLM over
`bloomfield`, `boeing` and `oppenheimer`, split in time by `splits.json`, and
thirteen generators propose motives from the earlier choices only.

**Metric.** Pooled **lift** — hypothesis-arm accuracy minus `obvious`-arm
accuracy on the same held-out choice points — computed by
`choice_forecast.py`. Paired, one-sided McNemar exact, plus a label-permutation
p at the family level, plus Benjamini-Hochberg q per hypothesis.

**Pass threshold.** McNemar p ≤ 0.05 **and** real lift > swapped-person control
lift. Three bands, fixed now:

- `swap ≥ real` → **CONTROL MOVED WITH IT**. Not a finding.
- `0.5 × real ≤ swap < real` → **WEAK**. Reported as weak, never as an effect.
- `swap < 0.5 × real` → **EFFECT**.

The floor this is stated against is the **swapped-person arm measured on the
same run**, not a seed floor. Standing rule 3.

**Control arms.** Four, each against a different objection:

| arm | objection it answers |
|---|---|
| `habit` | "you discovered that people repeat themselves" |
| `role` | "you discovered what people in this seat do" — from the record, never from the FABRICATED `*_profiles.json` |
| `obvious` | "a careful reader would have said that anyway" — **the bar** |
| `swap` | "the motive text is a *think harder* prompt" — the same motive forecasting a *different* person's choices |

**Tripwire, checked before anything else is read.** A `blind` arm sees only the
option labels — no transcript, no person, no situation. The extractor saw the
outcome when it wrote the alternatives, so **if blind accuracy exceeds chance +
0.10 the run is VOID** and no lift in it means anything.

**Hand check.** `--spot-check` samples choice points and asks a human exactly
one question: *was this a real choice*. Not whether any motive is insightful —
a human judging insight is how the last five designs became uninterpretable.
**Threshold: ≥ 80% judged real.** Below that the extraction is the problem and
the forecast numbers are not interpretable.

**Decision.**
- On **EFFECT**: `SURPRISES.md` is the deliverable, and the next step is the
  `altered_real_v1` dev side, where there is a label.
- On **WEAK** or **CONTROL MOVED WITH IT**: report that the motive is working
  as a prompt, not as a person model, and stop. Do not tune the motive prompt
  against the same test split.
- On **NO EFFECT**: report the `obvious` accuracy as the thing to beat and the
  per-method table as the map of what did not work. The negative is the result.
- On **VOID**: rebuild extraction so alternatives are generated without sight
  of the outcome, then rerun. No lift from this run may be quoted.

**Predicted failure modes.**
1. *The motive is a "think harder" prompt.* Any extra text in the prompt raises
   the arm that has it. Caught by `swap`: a generic prompt effect raises the
   control identically.
2. *The alternatives leak.* Caught by `blind`.
3. *Habit eats everything.* If `habit` ≈ `obvious` ≈ hypothesis, the corpora
   have too little behavioural variety for a motive to add anything, and the
   honest report is about the corpora, not the method.
4. *Surprises are luck.* With thirteen generators somebody is right on every
   item by chance. The null is the **swapped-motive hit count on the same
   misses**, printed beside the list; a surprise list no longer than its own
   control is not a finding.

---

# OUTCOME (written after scoring; predictions above unedited)

Run: 787 choice points extracted over bloomfield / boeing / oppenheimer, 156
motives over a cast of 12, **90 held-out choices**, 2,481 forecasts, chance
0.311. `choice_forecast.json`, `SURPRISES.md`.

## The tripwire fired

```
blind    0.622      chance 0.311 + 0.10 = 0.411      -> VOID
```

**VERDICT: VOID by the pre-registered rule.** No lift in this run may be
quoted as a clean result, and none is below.

**Post-hoc, written after the rule fired and changing nothing about the
verdict:** the pre-registered rule compares `blind` to CHANCE, which conflates
two different things — an option set whose phrasing gives the answer away, and
an option set on which the modal action is usually right. The second is the
class prior and a blind guesser recovers it for free. Separating them:

```
blind 0.622   <   majority 0.656
```

Blind does **not** beat the constant. What it recovered is the class prior, not
the phrasing. So the diagnosis is **a degenerate label distribution, not outcome
leakage** — and re-running the same extraction would not fix it. The tripwire as
written is the wrong test; it should compare `blind` to `majority`. That is a
correction to this file, not a rescue of this run.

## The result that matters, and it is not the lift

```
majority (constant "HOLD")   0.656
blind                        0.622
habit                        0.589
role                         0.589
obvious   <-- the bar        0.533
--------------------------------------------
pooled hypothesis            0.653      lift +0.120 over `obvious`
                                        clustered permutation p = 0.005
swapped-person control                  lift +0.091   (76% of the real lift)
```

**The pre-registered bar was badly chosen.** `obvious` — the model with the
whole transcript and no motive — is *worse than a constant*, by 12 points. A
lift measured against it is a lift over something no careful reader would
actually do.

**The full apparatus lands at the constant.** 156 motives, 13 generators, three
corpora: pooled hypothesis accuracy 0.653 against always-say-`HOLD` at 0.656.
Everything the motives buy is the class prior that `obvious` was losing.

**No hypothesis survives correction. Zero of 156 clear q <= 0.10.** The largest
individual lifts (+0.333) all sit on the person with n=3.

## Predictions scored

1. **"The motive is a think-harder prompt." CONFIRMED.** The swapped-person
   control reached +0.091 against a real +0.120 — 76% of the effect available
   without knowing whose motive it is. Even with a clean tripwire this would
   have landed in the pre-registered **WEAK** band, not EFFECT.
2. **"The alternatives leak." FIRED, BUT MISDIAGNOSED BY MY OWN RULE.** See
   above: blind beats chance and loses to the constant.
3. **"Habit eats everything." CONFIRMED, and worse than predicted.** `HOLD` is
   ~70% of every corpus. A constant beats habit, role, *and* the obvious read,
   and ties the entire motive apparatus.
4. **"Surprises are luck." CONFIRMED.** 27 choice points where the obvious read
   failed and some motive succeeded; the swapped-person motives — which cannot
   apply to the person whose choice they forecast — caught **25**. The list is
   its own null. `SURPRISES.md` is written, and reports 0 credible items.

One generator behaved differently and is not a finding: **revealed preference**
is the only one whose swapped-control lift is *negative* (-0.045 against a real
+0.094) — its motives help their own person and actively hurt on somebody else,
which is the signature person-specificity would have. It is 1 of 13, q = 0.74,
and stated here as a lead, not a result.

## Decision taken

Per the pre-registered branch for VOID, plus what the post-hoc diagnosis
changes:

- **Do not** re-run this extraction. The problem is not outcome leakage, so a
  blind-safe rewrite of the alternatives fixes nothing.
- **The tripwire is corrected in the code** to compare `blind` against
  `majority`, and `majority` is promoted to a permanent baseline.
- **The bar is corrected**: `obvious` alone is not a sufficient control on a
  task with a dominant class. Future runs score against `max(obvious,
  majority)`.
- **The task needs choice points where the answer is not `HOLD`.** A forecasting
  benchmark that is 70% one label cannot distinguish a person model from a
  constant, whatever the method. Either extraction is restricted to contested
  moments, or the label is made finer than the seven-way taxonomy.
- Nothing here is tuned against this split. The corpora keep their dev/test
  assignment.


---

# VERSION 2 — the choice-point filter with portfolio particles

v1 closed on P-14: the per-step rank scorer picks a different best hypothesis on
72% of steps, so stability, memory and alpha are all downstream of an evidence
signal that reshuffles. Phase CP then tried a different evidence unit and went
VOID, diagnosed as a degenerate label distribution (~70% `HOLD`) rather than
leakage.

v2 changes the evidence unit AND the particle: the filter runs at **choice
points**, each particle is a **portfolio** of 1–3 co-existing motives in priority
order, and a particle's likelihood is **the probability it gave the action the
person actually took**, recorded before the reveal.

Every entry below follows the template above. Two rules are promoted from the
Phase CP outcome into standing rules for all of v2:

- **The tripwire compares `blind` to `majority`, never to chance.**
- **The bar is never `obvious` alone** on a task with a dominant class.

---

## Gate 0 — is motive marginal mass churn-invariant by construction?

Written BEFORE the measurement; the three verdict bands below were fixed in
`prequential.py:gate0()` and committed to the function body before it was
executed. This is a **diagnostic on already-frozen data with no tunable
parameter**, not a blocker: its job is to tell every later phase how to read a
churn number.

**What is being asked.** The M2 gate was drafted as "motive-marginal churn far
below the old 55.6 argmax churn". That is not a legal comparison on its face — a
new metric is always free to look better than an old one, and standing rules 1
and 3 both bite. But the question is answerable for free, because a motive's
marginal mass on a **one-motive portfolio is its root's mass**, every v1 particle
**is** a one-motive portfolio, and `read_steps` rehydrates old logs. So the v2
metric can be computed retrospectively on the frozen v1 pool, through the same
`portfolio.py` code path v2 will use.

**Metric.** Median marginal churn per 100 steps over the `production_config`
pool, computed by `prequential.py --gate0`.

**Pass threshold.** Three bands, fixed in advance:

| band | reading |
|---|---|
| median < 5.0 | **VACUOUS** — churn-invariance is a property of the metric, not of portfolios. M2 is restated before any spend |
| within ±2.0 of the frozen figure | **REPRODUCES** — same quantity, so the M2 comparison is like-for-like |
| otherwise | **DIVERGES** — not the same quantity; resolve why before either is used as a gate |

**Control arm.** Lineage-level churn on the same runs, reported beside it. If the
two were interchangeable the plan's premise would be wrong in the other
direction.

**Decision.** On REPRODUCES: M2 may be stated as a churn comparison, with churn
reported as a **diagnostic** and prequential lift as the gate. On VACUOUS: M2 is
restated before anything is built. On DIVERGES: neither number is used until the
difference is explained.

**Predicted failure mode.** Reading a number that is low because the metric
cannot move, and calling it stability. Caught by the VACUOUS band and by
`runs_at_zero`.

---

# OUTCOME (written after scoring; predictions above unedited)

```
pool production_config    75 runs, 1,829 steps

marginal churn / 100      median 55.5556   mean 55.9375   range 23.0769 - 81.4815
runs at zero churn        0
lineage churn / 100       median 42.3077          <-- a DIFFERENT metric
frozen argmax churn / 100 56.4103                 delta -0.8547
```

**VERDICT: REPRODUCES.** Marginal churn and the frozen root-level figure are the
same quantity on one-motive portfolios. **Zero runs sit at zero churn**, so the
metric is not free — it is measuring the same instability v1 measured, and a
fall in it under v2 would mean something.

**A correction to the v2 plan, found by writing this check.** The plan asserted
that `baseline_metrics.json`'s 56.41 is computed on `lineage_id` by
`trace_log.argmax_churn`. **It is not.** `baseline_snapshot.read_run` computes its
own churn on `top_root` taken from `roots_of()`, which sums weight per **root**,
so the frozen figure was already root-level and the like-for-like number existed
all along. `trace_log.argmax_churn` is a separate lineage-level metric and reads
**42.31** on the same pool — 13 points apart, which is exactly how much damage
swapping them would have done. Both are reported by `--gate0` from here on so the
substitution cannot be made silently again.

The residual −0.85 against `baseline_snapshot` is **reported, not reconciled**,
following the rule already applied to the PRECHECKS deltas: the two differ in
tie-breaking and in how particles with a null `root_id` are skipped. Tuning
`prequential.py` until the numbers matched would be fitting the measurement to
the claim.

---

## v2 standing rules, added to the six above

7. **The tripwire compares `blind` to `majority`, never to chance.** Recovering
   the class prior is free; only beating it indicates the option phrasing
   carries the answer. Implemented in `label_gate.check_options_only`.
8. **The bar is never `obvious` alone** on a task with a dominant class, and it
   is **frozen on dev** before test is read. A bar taken as a pointwise max at
   the realised outcome is not a proper score and is biased against every arm.
9. **Every LLM arm carries its own dev-fitted temperature**, and the epsilon cap
   on per-item log-score contributions is applied identically to every arm
   including the bar. Calibrating one arm and not the others converts a
   calibration gain into apparent lift.
10. **Subsample whole UNITS, never choice points.** The filter is sequential;
    dropping points from inside a person's timeline scores a filter that skipped
    its own history.
11. **The v2 path is `choice_tracer.py` + `run_choice.py`.** `tracer.py`'s
    `_trace` is not modified. Enforced by `test_default_path.py`, which pins
    golden hashes of the rank-scorer, standard-block, seeding, split and perturb
    prompts.

### Frozen artifacts, v2 additions

| File | What it fixes | Verify with |
|---|---|---|
| `label_gate.json` | the label distribution a corpus was cleared on | `python label_gate.py --corpus X --side dev --by-choice-type` |
| `forecast_backend.json` | the pinned forecaster and the checks it passed | `python confirm_forecast.py --gemini` |
| `splits_v2.json` | dev/test, M1/M2 partitions and units for the new corpora | `python make_splits.py --check-v2` |

`splits.json` and `baseline_metrics.json` are **not regenerated**. Adding a v2
corpus to either would make every existing `--check` fail, which is the drift
alarm working correctly.

---

## The decision table — written before any v2 number is read

It is entirely plausible that CaSiNo stays calibration-only, atla is demoted to
diagnostics, and Diplomacy is late. Without this table the generalisation claim
can end with no valid test and nobody notices until the end.

| outcome | the claim becomes |
|---|---|
| **M1 fails on CaSiNo** | **STOP.** Do not build the filter on an untested signal -- that was v1's mistake. Either revise the forecaster (calibration, prompt, letter scheme) and re-run M1, or report the negative and halt. No M2 work proceeds |
| all corpora pass their gates | the generalisation claim, reported per corpus |
| atla contaminated | claim rests on Diplomacy plus the CaSiNo silent subgroup; atla is diagnostics only |
| **Diplomacy fails the label gate** | promote `bloomfield` / `boeing` -- private, so they cannot be memorised, and Phase CP failed there on label degeneracy, not contamination. Their taxonomy is fixed below, not chosen after seeing Diplomacy fail |
| CaSiNo deal responses fail the gate | concessions carry M1, or CaSiNo drops to the silent subgroup only |
| **CaSiNo fails the `context_neutral` headroom gate** (plausible -- negotiators state their priorities aloud) | CaSiNo drops to the silent subgroup only; if that subgroup misses its MDE, CaSiNo is calibration-only and contributes no measurement |
| **smeeple transcripts turn out to be generated from the personas** | smeeple is **dropped**, not caveated. Recovering scripted concealment is circular |
| **smeeple consent** | SETTLED, see "Recorded decisions": the deliverable is internal, so smeeple is used identified and carries the worked example. The scope condition is recorded there |
| both long-timeline corpora fail | M2's sequential half is **not reported**. M1 stands alone as a likelihood-validation result, and that is stated as the result |

### The bloomfield / boeing taxonomy, fixed now

The choice point is defined by **the request or disagreement directed at the
person** -- part of the *input* -- and never by their having responded to it.
Defining it by the response would exclude the cases where somebody **ignored**
the request, which is the behaviour most likely to hide a motive; that is the
same outcome-selection error as discarding a choice point whose actual action
was not among the generated alternatives. Outcomes:

```
comply | counter-propose | defer | escalate | decline | no response in window | OTHER
```

Deliberately **not** Phase CP's seven-way taxonomy, whose `HOLD` absorbed 70% of
the mass. `no response in window` is recoverable rule-based from `addressed_to`.
`OTHER` is scored and its rate reported. The label gate applies per choice type.

**Note on the existing extractor.** `choice_points.validate` currently rejects a
record when `actual not in seen` ("actual not among the alternatives"). That is
selection on the outcome: it keeps only the predictable choices, inflates every
arm, and deletes exactly the behaviour being looked for. Every v2 converter
emits an explicit `OTHER` alternative instead, and reports the `OTHER` rate.

---

## Phase L — the label gate (per corpus, per choice type)

Written BEFORE any v2 corpus was converted.

**What is being changed.** Nothing in the filter. This is a hard precondition on
the task, run on **dev only**, before any forecasting spend.

**Metric.** Five checks computed by `label_gate.py`, three of which need no
model call.

| check | threshold | direction |
|---|---|---|
| majority-class share | <= 0.45 | lower passes |
| normalised entropy, over alternatives OFFERED | >= 0.75 x log(k) | higher passes |
| pooled leave-one-out accuracy of a per-person constant | <= 0.60 | lower passes |
| `options_only` | <= majority + 0.05 | **lower passes** -- above it the phrasing leaks |
| `context_neutral` sanity / headroom | >= majority - 0.05 **and** <= ceiling - 0.05 | **higher / lower** |

The two model arms run on a 100-point dev sample before extraction is scaled.

**Pass threshold.** All five, per choice type. A corpus that passes in aggregate
while one kind inside it is a constant has not passed.

**Control arm.** The Phase CP distribution itself, as a regression fixture:
70% `HOLD`, blind 0.622, obvious 0.533, majority 0.656. The gate must reject it,
must NOT flag it as leakage, and must fail it on the bar.

**Decision.** On pass: forecasting may be budgeted. On fail: the corpus takes its
row in the decision table. **No prompt is tuned against a failed gate** -- the
extraction or the corpus is the problem, not the wording.

**Predicted failure mode.** Inverting one of the two lookalike arms and killing a
usable corpus, or clearing a degenerate one. Caught by `test_gate_directions.py`,
which drives each gate with a value that must fail it.

---

# OUTCOME (written after scoring; predictions above unedited)

No v2 corpus has been converted yet, so no corpus has been gated. What HAS run
is the regression fixture, in `test_gate_directions.py`:

```
Phase CP replayed through label_gate:

  majority_share       0.700   FAIL   (the finding: 70% HOLD)
  entropy_frac         0.314   FAIL
  loo_person_constant  ~1.000  FAIL
  options_only         0.622   PASS   <-- blind LOST to majority 0.656
  context_neutral      0.533   FAIL   <-- the bar was below a constant
  overall                      FAIL
```

The gate reproduces Phase CP's own post-hoc diagnosis from its own numbers: the
option phrasing did **not** leak, and the pre-registered bar was the problem.
Had this gate existed, that run would have stopped before the first of its 2,481
forecasts.

---

## Phase B — forecaster confirmation — **NOT RUN**

Blocked until `forecast_backend.json` shows `confirmed: true`. Recorded now so
the dependency is visible.

The forecast answers with a single option letter `A`-`H`: one token in any
tokenizer, so one call returning top-k log-probs at that position gives an exact
normalised distribution over the alternatives (k=20 covers eight, which is also
why Diplomacy's choice unit is at most seven powers plus `HOLD`). Action labels
are **not** scored -- they are multi-token with different lengths, which is the
bias `ContinuationScore.mean` already exists to correct.

Six checks, four inherited from §0.3 and two new:

| check | meaning for a distribution |
|---|---|
| aligned | top-k sits at the answer position; returned letters are a subset of those offered; the first output token is forced to be the letter; thinking mode off and its interaction with log-probs verified |
| responsive | a portfolio naming the taken option's aim puts more mass on it than a scrambled portfolio does on the same prompt |
| ordered | more mass on an alternative after the context that licenses it than after one that does not |
| deterministic | two identical requests agree **within a declared tolerance**. Exact float identity will not hold on a hosted API, so the check is restated rather than silently relaxed |
| **letter-permutation invariance** | the same choice point under two letter assignments induces the same distribution over ACTIONS. Without this a positional prior on "A" reads as a motive effect |
| **calibration** | a reliability curve on dev plus a dev-fitted temperature **per arm**. Letter probabilities are typically overconfident, often near 1.0; that turns reweighting into hard argmax updates -- bringing back the churn v2 exists to reduce -- and makes log-scores dominated by the epsilon cap |

**Sampled frequency is not a viable production fallback**, and this is arithmetic
rather than preference: at `eps_frac = 0.12` and n = 6 the epsilon floor is
~0.02, so a sampling resolution of 1/K needs K > n/eps_frac ~ 50 or sampling
noise, not the floor, binds the weight update. It is used for the offline
pipeline and these checks only. If `response_logprobs` does not work, the honest
options are a coarse declared ranking over alternatives, or pause.

---

## Phase M1 / M2 — **NOT RUN**

Recorded as stubs so the ordering is on the record. Neither may be filled in
until `label_gate.json` and `forecast_backend.json` both pass for the corpus in
question, and the bar, the merge floor, the collapse tripwire and the surprise
margin are frozen on dev.

**M1** -- one-shot, no filter, CaSiNo. Metric: prequential log-score lift over
the frozen bar, pooled, clustered permutation on the person. Controls: `placebo`
(the binding one -- Phase CP's failure was a think-harder effect, which only a
generic same-shape portfolio isolates) and `swap`, with the swap bands corrected
for expected overlap (one in six random swaps shares the exact priority order,
half share the top item). Bands reused verbatim from Phase CP: `swap >= real` ->
CONTROL MOVED WITH IT; `0.5 x real <= swap < real` -> WEAK; `swap < 0.5 x real`
-> EFFECT, and the same bands applied to `placebo`.

**M2** -- the filter with portfolios. Gate: prequential lift per corpus, never
pooled. **Churn is a diagnostic, not the gate** -- "marginal churn far below v1"
rewards a frozen population, which is the opposite of what shift detection needs.
Stability is reported as *changes only when the evidence does*: the correlation
between per-step churn and likelihood-ratio magnitude. Gate 0 established that
marginal churn on the v1 pool reads 55.56 with zero runs at zero, so a fall in it
would mean something.

Pre-registered as the one named subgroup, so it cannot be a post-hoc rescue a
second time: **revealed preference**, the only Phase CP generator whose
swapped-control lift was negative (-0.045 against a real +0.094).

---

## What counts as a CHOICE — one definition, checked per corpus

Phase CP extracted 787 "choice points" and the run landed on a constant. Two
readings survive that: the motives were empty, or the MOMENTS were, and nothing
in the run could separate them. The extraction rule was written per corpus and
never stated as a general property, so there was nothing to check it against.

**A choice is a moment, identifiable from what came before, where the person had
at least two genuinely feasible and distinguishable courses of action, and where
plausible motives would favour different ones.**

Four clauses, each with a place it is enforced and, where possible, a number.

| # | clause | what it rules out | enforced by |
|---|---|---|---|
| **C1** | **identifiable from what came before** | a point defined by what the person did. Defining it by the response silently excludes everyone who ignored the request -- the behaviour most likely to hide a motive | the trigger rules below; `test_hindsight.py`; `ChoiceStream.prefix` |
| **C2** | **genuinely feasible** | options the person could not actually have taken -- an illegal Diplomacy order, accepting a deal nobody offered | the per-corpus feasibility rule; Diplomacy's map adjudicator |
| **C3** | **distinguishable** | option sets that are one action written twice, which inflate the count without adding a decision | `prequential.option_separability`, reported per corpus |
| **C4** | **plausible motives would favour different ones** | dramatic-looking moments where every account predicts the same thing | `prequential.forecast_disagreement`, reported and stratified |

### The trigger rule for every corpus

Written now, so no corpus gets a rule invented after its numbers are seen.

| corpus | trigger (C1 -- all input-side) | feasibility (C2) | outcomes |
|---|---|---|---|
| **CaSiNo** | partner submits a deal (deal response), or the person submits one (concession) | a deal response needs a deal on the table; `Walk-Away` is always available | `Accept` / `Reject` / `Walk-Away` / `Counter`; concessions: which issue gives ground, or none |
| **Diplomacy** | the player exchanged messages with power P this phase, one point per (player, P) pair | **map adjudicator** -- only legal supports and attacks are listed | `support` / `attack` / `neither` |
| **atla** | **a character is asked, ordered, offered or challenged by another character** | the other character is present in the scene and the action is available in it | `comply` / `refuse` / `deflect` / `counter-propose` / `escalate` / `no response in scene` / `OTHER` |
| **bloomfield / boeing** | a request or disagreement is directed at the person | the person is present in the channel and addressed | `comply` / `counter-propose` / `defer` / `escalate` / `decline` / `no response in window` / `OTHER` |
| **smeeple** | as bloomfield | as bloomfield | as bloomfield |

The atla and bloomfield outcome sets are deliberately parallel: the
generalisation claim is comparative across corpora, and two differently-shaped
taxonomies would make the comparison a comparison of taxonomies. `deflect`
covers changing the subject or joking it away; `escalate` covers raising the
stakes or striking first. `OTHER` is always listed, always scored, and its rate
always reported -- a point whose actual action is not among the alternatives is
**never discarded**, because that is selection on the outcome.

### C4, measured

At each choice point, the population's forecasts disagree by

```
D = H( sum_i w_i p_i )  -  sum_i w_i H( p_i )        bits
```

the Jensen-Shannon divergence of the population, which is exactly the mutual
information between *which account is right* and *which action is taken*. It is
zero if and only if every account forecasts identically, and bounded above by
`H(w) <= log2(n)`; the normalised form divides by `H(w)` so corpora with
different population sizes are comparable.

**It is reported and stratified, never used to select.** Dropping
low-disagreement points would discard exactly the moments the population found
uninformative and report the remainder as if it were the task -- the same error
as conditioning on "the majority action was not taken". Every point stays in
exactly one stratum and both are reported (`test_choice_definition.py` asserts
that nothing is dropped).

**The cut is frozen on dev**: the median of the dev disagreement distribution,
stated as a rule and its resolved value written into `label_gate.json`.
Re-cutting on test would let the boundary move to wherever the lift happened to
be.

**Pre-registered prediction, so the stratification cannot be read after the
fact.** If the motives are doing discriminating work, **lift concentrates in the
high-disagreement stratum**. If lift is flat across the two strata, the motives
are not discriminating between moments even if the pooled number is significant,
and that is reported as the result. If the corpus is mostly low-disagreement, the
finding is about the **corpus** -- as Phase CP's was -- and the honest report
says so rather than quoting the pooled lift.

---

## New situations — the test the time split does not perform

Forecasting well on familiar situations does not show a model would forecast
well once something changes, which is the whole point of asking what somebody
would do. The dev/test split in `splits_v2.json` cuts on **time**, so the test
side still contains the same KINDS of moment as the dev side; a model that had
simply memorised "in situations like this, this person does that" would pass it.

Two transfer axes, both pre-registered, reported beside the pooled number.

**T1 -- leave-one-situation-type-out (LOTO).** Situation type is a categorical
label on the choice point, taken from the **input** and fixed before any
forecast. Motives are seeded on every type except T and forecast only on T.

| corpus | situation type held out |
|---|---|
| CaSiNo | negotiation phase: opening / bargaining / closing. Seeded on the first two, tested on closing, where `Walk-Away` and `Accept` live |
| Diplomacy | relationship state with that power: allied / neutral / hostile, from recent orders. The interesting hold-out is motives learned while allied, tested after the relationship turned |
| atla | trigger type: asked / ordered / offered / challenged |
| bloomfield / boeing / smeeple | request type, and channel |

**T2 -- cross-span transfer.** Motives learned on one thread or span of a person,
forecast on another span of the SAME person. This is the standing-motive claim
and the closest available check on "would this still be true elsewhere".
Available on Diplomacy, atla, bloomfield and boeing. **Not available on CaSiNo**,
which has no persistent participant id and one dialogue per person -- stated here
so its absence is not later read as a null result.

**Decision.** M2's headline is reported **pooled AND per held-out type**. A
method whose lift vanishes under T1 has not shown it generalises to new
situations, **however significant the pooled lift is**, and the report says so
in those words. Neither axis may be dropped after its number is seen.

**Power.** LOTO shrinks n by construction, so the MDE is computed per held-out
type on the partitioned size, not on the whole. A type below its MDE is reported
as **exploratory**, never as a measurement. The `placebo` and `swap` controls are
recomputed **within each held-out type** -- a control measured on the pooled
distribution is not a control for the stratum.

**Predicted failure mode.** Reporting a pooled lift that is carried entirely by
one situation type, and calling it a person model. Caught by T1 being mandatory
rather than exploratory, and by the per-stratum controls.

---

## Recorded decisions

Both taken by the project owner, recorded here because each changes what may be
reported and neither should have to be reconstructed later.

- **CaSiNo is acquired into `data/`** (public, CC-BY-4.0) and converted by
  `casino_ingest.py`.
- **smeeple is used identified, including as the deliverable's worked example.**
  The scope this rests on is that **the deliverable is internal**: this repo and
  its reports are not published outside the organisation. The corpus contains a
  real child and inferred private concerns about him, so if the deliverable is
  ever taken outside that boundary, consent and de-identification become
  preconditions again and this entry is the record of what the decision assumed.
  The **provenance** precondition is unaffected and still blocking: if the
  transcripts turn out to have been generated from the personas rather than the
  personas written from a real recording, `hidden_concerns` is a script and
  recovering it is circular, so smeeple is dropped rather than caveated.

---

# OUTCOME — Phase L on CaSiNo (no-model half; written after measuring, predictions unedited)

`data/casino/casino.json`, 1,030 dialogues, 2,060 participants, mean 13.9 turns.
Two candidate choice types were counted **before any forecasting was budgeted**,
per the standing instruction to count structured-only points first rather than
assume a volume.

### deal_response — **FAILS**, and the pre-registered branch applies

```
n = 1,181 over 1,084 people, k = 4

  Accept-Deal 1005   Reject-Deal 167   Walk-Away 9   Counter 0   Keep-Talking 0

  [FAIL] majority_share      0.8510   must be <= 0.45
  [FAIL] entropy_frac        0.3254   must be >= 0.75
  [PASS] loo_person_constant 0.0567
```

**0.851 is worse than Phase CP's 0.656.** `Counter` and `Keep-Talking` never
occur: the other party's next turn after a `Submit-Deal` is always one of
accept / reject / walk away, so the option set is a three-way choice that is
85% one label. This is the `HOLD` problem again, and it was found for the price
of reading the file.

Per the decision table row "CaSiNo deal responses fail the gate": deal responses
are **dropped**, and the allocation choice carries M1.

### allocation — **PASSES** all three no-model checks

The choice is *which issue the proposer claims the largest share of*, read from
`task_data.issue2youget` at each `Submit-Deal`. Ties are their own labels rather
than being broken, because a deal that claims two issues equally is a different
claim from one that picks a favourite.

```
n = 1,181 over 1,084 people, k = 7

  Water 276  Firewood 274  Food 235
  Firewood+Water 130  Food+Water 130  Firewood+Food 119  all-three 17

  [PASS] majority_share      0.2337   must be <= 0.45
  [PASS] entropy_frac        0.9137   must be >= 0.75
  [PASS] loo_person_constant 0.0542   must be <= 0.60
```

This is also the choice that bears directly on the hidden label, which is itself
close to uniform -- all six priority orders fall between 0.163 and 0.171 across
2,060 participants. A corpus whose ground truth is balanced and whose choice
labels are balanced cannot be won by a constant, which is the property Phase CP
lacked.

`options_only` and `context_neutral` are still **pending**; they need the model
and run on a 100-point dev sample before extraction is scaled.

### CaSiNo cannot carry any sequential claim — sharper than the plan assumed

```
structured points per person   mean 1.15   median 1   max 13
Submit-Deals per person        1: 1,025 people   2: 40   3+: 19
concession points (a 2nd+ offer by the same person, corpus-wide)   97
```

The plan said "the filter barely runs here -- 2-5 choice points per person". The
measured figure is a **median of one**. There is no sequence to filter on the
structured path, and "concession relative to the previous offer" is not a viable
choice type either: 97 points corpus-wide, over roughly 90 people.

**CaSiNo is therefore Milestone 1 only, and that is now a measurement rather
than an expectation.** Its job is to validate the choice likelihood and the
recovery of a known priority order one-shot. Every sequential claim -- the
filter, the operators, marginal churn, transfer -- rests on atla, Diplomacy and
the bloomfield contingency, and no M2 number may be quoted from CaSiNo.

---

# OUTCOME — CaSiNo's five open decisions, settled by measurement before the converter

Every figure below is a count over `data/casino/casino.json`. No model was called.

### 1. The allocation is already in the chat, so the prefix is cut earlier

```
an explicit PROSE allocation precedes the Submit-Deal in 1,158 / 1,181 = 0.981
prefix to the submission          mean 12.4 turns   median 11
prefix to the first allocation    mean  3.8 turns   median  3
```

**98.1%.** Forecasting the allocation from the conversation up to the submission
is reading comprehension: the two people have just spelled the deal out in prose
and the `Submit-Deal` formalises it. `context_neutral` would sit near ceiling,
the headroom gate would fail, and the lift would be a measurement of the model's
reading.

**Registered before the converter runs: the prefix is cut at the first explicit
allocation proposal by either party**, and the task is to forecast the eventual
allocation from what came before it. That is selection on the **input**, it
creates genuine headroom, and it matches the actual question -- inferring what
somebody wants before they have negotiated it out loud. It is a change to the
task definition and is recorded as one. The detector is a frozen regex (a
quantity adjacent to an issue word, or a `Submit-Deal`), reported with its hit
rate, not tuned afterwards.

### 2. The silent subgroup, counted

```
person stated no issue priority anywhere in their own prefix

  under the full cut    215 / 1,181 = 0.182
  under the early cut   743 / 1,181 = 0.629
```

Under the full cut the subgroup is 215 points, which makes it a footnote. Under
the early cut it is 743. **So #1 is necessary, not optional**: the early cut is
what turns CaSiNo's only genuine test from a footnote into a dataset. (Detector:
an issue word within a need/priority phrase, frozen and reported.)

### 3. Ties get their own outcomes

```
proposals claiming one issue outright   785  0.665
ties (2/2/1, equal splits, etc.)        396  0.335
```

**A third of all allocations are ties.** They are their own labels
(`Firewood+Water`, `Food+Water`, `Firewood+Food`, all-three), never excluded and
never broken. Excluding them would be selection on the outcome and would delete a
third of the corpus; breaking them would invent a preference the proposal does
not express. A deal claiming two issues equally is a different claim from one
that picks a favourite, and the label says so.

### 4. The M2 partition is folded into M1 test

CaSiNo has no M2 (see below), so an untouched M2 partition would buy a
replication of a calibration result. Power on the **silent subgroup** is worth
more: at 743 points corpus-wide, splitting dev/test and then test again into
M1/M2 would leave roughly 186 silent points to carry CaSiNo's only real test.
**Registered before either partition is read**, per rule 6.

### 5. CaSiNo cannot test new situations either

```
where the Submit-Deal falls in the dialogue (0 = start, 1 = end)
  p25 0.909   p50 0.909   p75 0.923   p90 0.933   mean 0.887
  share in the last 25% of the dialogue: 0.919
```

Every allocation choice happens at the end. **"Negotiation phase" does not vary**,
so it cannot be a held-out situation type, and no other input-side type varies
either at one point per person. **CaSiNo can test neither transfer axis** -- not
cross-span (no persistent participant id) and not new-situation (no phase
variation). Recorded now so neither absence is later read as a null result.

**CaSiNo's role, final:** Milestone 1 calibration, on the early cut, with the
silent subgroup as its one genuine test. No sequential claim, no transfer claim.

---

# OUTCOME — Diplomacy, gated before any CaSiNo converter work

Run first, deliberately: if Diplomacy failed the way CaSiNo's deal responses did,
the bloomfield contingency would become the main path and the CaSiNo work would
have been done against the wrong plan.

**The season field exists.** Messages carry `seasons` ('Spring' / 'Fall' /
'Winter') alongside `years` and `game_id`, and the moves files are keyed
`DiplomacyGame{N}_{year}_{season}`. Alignment is therefore
`(game_id, year, season)` and the Fall-messages-before-Spring-orders leak the
plan warned about **cannot occur**. 342 phase files parse, 114 per season, game
ids 1-12 match on both sides, and **all 312 messaged phases have orders**.

Order types: MOVE 4,041, SUPPORT 2,029, HOLD 1,091, BUILD 441, DISBAND 155,
CONVOY 138. `SUPPORT` carries `from`, so the supported power resolves; occupancy
comes free from the order keys, since `orders[Power]` is keyed by that power's
own unit provinces.

### The (player, power) pair unit FAILS

```
movement phases, ordered pairs that exchanged messages

  messaged only              n=3,020   neither 0.816   attack 0.104  support 0.074
  messaged AND in contact    n=2,311   neither 0.759   attack 0.136  support 0.097

  majority_share   0.8159 / 0.7594   both FAIL (limit 0.45)
```

Restricting to messaged powers was supposed to keep `neither` down and does not:
talking to a power is not the same as being able to reach it. Adding the
secondary *in contact* restriction helps by six points and still fails. This is
structural, not fixable by a better restriction -- a power has a handful of units
and six neighbours, and acts on at most two or three of them in a phase, so the
pair framing manufactures a `neither` sink the way `HOLD` and `Accept` did.

### The primary-target unit PASSES, and is adopted

One choice per (player, movement phase): **which power does this player act
against or for this phase**, by the frozen rule *the power receiving the most of
this player's orders, ties broken alphabetically*, with `NONE` when the player
acts on nobody.

```
n = 1,058 over 83 (game, power) units, k = 8

  NONE 368 0.348 | Austria 151 0.143 | France 121 0.114 | Russia 94 0.089
  Italy 88 0.083 | England 86 0.081 | Germany 85 0.080 | Turkey 65 0.061

  [PASS] majority_share      0.3478   limit 0.45
  [PASS] entropy_frac        0.9728   limit 0.75
  [PASS] loo_person_constant 0.3478   limit 0.60
```

Seven powers plus `NONE` is **eight labels**, which is exactly the `A`-`H` letter
scheme already registered in Phase B.

**All three candidate rules were measured and all three are reported**, so the
adopted one is not a survivor of an unreported search:

```
primary attack target only      majority 0.4395  entropy 0.9077   PASS (marginal)
primary support target only     majority 0.7628  entropy 0.5166   FAIL
primary target, attack or support majority 0.3478 entropy 0.9728  PASS  <- adopted
```

**The selection was made on the LABEL DISTRIBUTION, with no forecast in
existence.** No lift, no arm and no model output influenced it. That is the same
basis on which CaSiNo's deal responses were dropped, and it is recorded here so
the choice cannot later be mistaken for a result.

### Two things this leaves open, both flagged now

- **A real adjacency map is still needed, for the ALTERNATIVES rather than the
  label.** The label needs no map, but C2 (genuinely feasible) does: offering
  "attack Turkey" to a player whose units are nowhere near Turkey is not a
  feasible option. The adjacency used above was reconstructed from observed
  transitions and is too permissive to serve -- mean degree 12.6 against a real
  board's four to five -- so the converter takes a proper standard-map
  adjacency, and that is a build item, not a measurement.
- **The relationship-state situation type must be defined from PRIOR phases
  only.** A relationship "turning" is normally identified by an attack, which is
  itself a choice outcome; if the type at phase *t* reads anything from phase *t*
  or later, the transfer split leaks the outcome it is meant to hold out.
  Registered definition: **a PRIOR phase contained an attack between these two
  powers**. Added to `test_hindsight.py`'s obligations.

---

## Refinements to the clause-4 measurement, registered

- **The normalisation bound is `min(H(w), log2 k)`, not `H(w)`.** Dividing by the
  weight entropy alone understates a saturated point: four accounts maximally
  opposed over two options have `H(w) = 2` bits but can only ever reach 1, so
  `H(w)` alone would report 0.5 for a population as split as the option set
  permits. Fixed in `prequential.forecast_disagreement`.
- **Stratify WITHIN choice type.** A type offering more options can score higher
  for free, so a single corpus-wide median cut would partly sort points by type
  rather than by disagreement. `stratify_by_disagreement(..., within='kind')`.
- **Record position in the run, and check the high stratum against it.** Weights
  start uniform and concentrate, so disagreement falls over a run by
  construction and early points land in the high stratum on position alone. Any
  concentration of lift in the high stratum is reported beside the position
  distribution, and lift that is really "lift early in the run" is named as that.
- **Flat lift across strata has two readings, and the placebo separates them.**
  Registered in advance: lift in the LOW-disagreement stratum that sits **above**
  the within-stratum placebo is a shared insight the bar lacks -- something every
  account knows and the reference does not. Lift in that stratum **at** placebo
  level is the think-harder effect, i.e. the Phase CP failure recurring. The
  placebo is computed within each stratum for exactly this reason.

---

# OUTCOME — Avalon added, and a defect it exposed in the gate itself

`sstepput/Avalon-NLU` (MIT, EMNLP 2023 Findings). 20 games x 6 players, with
**ground-truth player roles** (merlin, percival, morgana, assassin, two
servants), party proposals, per-player party votes, quest outcomes, occasional
player beliefs, and per-utterance persuasion/deception labels.

It is the strongest hidden-motive label in the project: a role is assigned at
setup, never stated in the transcript, and drives everything the player does.
Added because Diplomacy was otherwise carrying the sequential claim alone and
atla may yet fail its contamination probes.

### The gate had a defect, and a binary choice type found it

```
majority share of a BINARY label is >= 0.5 BY ARITHMETIC
```

`MAJORITY_MAX = 0.45` is therefore **unreachable at k=2** -- the gate would have
rejected a perfect coin flip, and did reject Avalon's include/exclude choice at
0.520, two points off balanced. Entropy cannot cover for it either: normalised
by log k, a 0.708/0.292 split scores 0.871 and sails through.

**Fixed: the limit is k-aware.** The standard is unchanged -- a constant must not
be close to unbeatable -- but the number is stated against what k allows:

```
k >= 3    majority <= 0.45     a constant loses more often than it wins
k == 2    majority <= 0.65     a constant is at most a 65/35 split
```

Every measured choice type re-run through the corrected gate, as one table, so a
future threshold change has to face all of them at once (`test_gate_directions`):

```
choice type                majority      k   limit  verdict
Phase CP (7-way)             0.6560    7     0.45   FAIL
CaSiNo deal_response         0.8510    4     0.45   FAIL
CaSiNo allocation            0.2337    7     0.45   PASS
Diplomacy pair               0.7594    3     0.45   FAIL
Diplomacy attack-only        0.4395    8     0.45   PASS
Diplomacy signed             0.3478   15     0.45   PASS
Avalon party vote            0.7083    2     0.65   FAIL
Avalon include/exclude       0.5203    2     0.65   PASS
```

### Avalon party vote — FAILS

```
n = 696 over 120 player-games   yes 493 / no 203 = 0.708
  [FAIL] majority_share 0.7083 (limit 0.65)   [PASS] entropy 0.8709
  [FAIL] loo_person_constant 0.6264

yes-rate by the voter's HIDDEN ROLE:
  servant-1 0.802 | servant-2 0.741 | assassin 0.698 | merlin 0.690
  percival  0.690 | morgana   0.629
```

Players approve most parties, and the **hidden role moves the marginal vote rate
by only ~17 points across all six roles**. That second number is the more
informative one: it is a direct corpus-level read on clause 4, and it says the
raw vote label carries little of what the role determines.

### Avalon include/exclude — PASSES, and is adopted

For each proposal, one point per (leader, other player): did the leader put them
on the party. Base rate is favourable by construction -- parties are 2 to 4 of 6.

```
n = 740 over 101 leader-games, k = 2, 148 proposals (sizes 2:23, 3:58, 4:67)
  exclude 385 / include 355
  [PASS] majority_share      0.5203   limit 0.65
  [PASS] entropy_frac        0.9988
  [PASS] loo_person_constant 0.4649
  OVERALL PASS
```

**7.3 choice points per leading player**, 101 leader-games over 20 games -- a
genuine if short sequence, far better than CaSiNo's median of one. Note the
leader's own include-rate barely varies by role (0.459-0.529), the same signal as
the vote: a motive model has to condition on WHO is being included, not on the
leader's rate.

Quest votes are **not usable**: only the aggregate outcome is recorded (53
succeeded / 29 failed), never the per-player pass/fail, which is the sharpest
motive signal in the game and simply is not in the data.

---

## Diplomacy, re-specified after the pair unit was dropped

### The label is SIGNED, and the previous rule was wrong

The adopted "primary target, attack or support" rule labelled a point `Germany`
whether the player attacked or supported Germany -- **opposite motives under one
label**. Neither the motive reading nor the lie metric survives that. Measured
both ways:

```
n = 1,058 player-phases

SIGNED (direction in the label, 15 labels)
  NONE 0.348 | attack:Austria 0.104 | attack:France 0.099 | attack:Russia 0.083
  attack:Germany 0.070 | attack:Turkey 0.067 | attack:Italy 0.064
  attack:England 0.060 | support:* 0.105 total
  majority 0.3478 PASS   entropy 0.8075 PASS   loo 0.3346 PASS

ATTACK-ONLY (8 labels)
  majority 0.4395 PASS   entropy 0.8494 PASS   loo 0.4036 PASS
```

Attacks carry 0.547 of the mass against support's 0.105, so the old rule was an
attack rule with ten points of support contamination. **SIGNED is adopted**: it
is the semantically correct label, it passes with a lower majority share, and it
is what the lie metric needs. Per-point alternatives are (powers in contact x 2
directions) + NONE, so the letter scheme widens from A-H to as many letters as
the largest offered set; top-k = 20 still covers it.

### Tie-breaking, reported

```
two powers drawing equal order counts
  signed label      173 / 1,058 = 0.164
  attack-only        74 / 1,058 = 0.070
```

**16.4% is not negligible**, and an alphabetical tie-break is label noise nobody
could predict. Registered tie-break, in order, each visible to a forecaster in
principle: (1) the target whose **targeted province is a supply centre**;
(2) the target against which more total units are committed; (3) alphabetical, as
a last resort, with the residual rate reported.

### The lie metric, re-specified

The old definition read divergence per (player, recipient, phase). That unit no
longer exists. **New definition, registered before any forecast exists:** the
forecast probability that the primary target is the recipient the player
**promised not to move against**, i.e. `P(attack:R)` for a recipient R with an
extracted non-aggression promise, scored as PR-AUC against the aggregated
`speaker_intention`.

Two ceilings, registered up front because both lower it:

- it catches only betrayals aimed at the player's **primary** target; a lie
  followed by a secondary attack is invisible;
- it reads only **"I won't attack you"** promises; a broken promise of support
  cannot be recovered from a target label.

This makes the existing rule -- **compare against `role` and `obvious`, never
against an absolute threshold** -- load-bearing rather than cautionary.

### Simultaneity is across players, not only within one

Batch reveal was specified for the pairs inside one player-phase. The pair unit
is gone, but the harder version remains: **all seven powers order simultaneously**,
so player B's context for phase *t* must exclude player A's phase-*t* orders and
the adjudicated results of phase *t*. That is a board-state leak, and the nonce
test will not catch it unless the fixture contains a second player acting in the
same phase. Added to `test_prequential.py`'s obligations.

### The map

The reconstruction (mean degree 12.6 against a real board's four to five) is
replaced by the standard board from the `diplomacy` Python package (Mila), licence
to be checked before it is vendored. It is needed for C2 -- listing feasible
alternatives -- not for the label.

---

## CaSiNo, two qualifications on the early cut

**The early cut lowers the achievable ceiling, and that is stated rather than
discovered.** From three turns in, the final allocation depends on the proposer's
priorities **and** on everything the partner does afterwards. Part of the outcome
is not explained by the proposer's motives at all, so the lift available to any
method is bounded below what the headroom figure suggests. Acceptable for
calibration; recorded so the number is not read as a failure of the method.

A cleaner target exists -- the proposer's **own first explicit proposal**, which
reflects their preference more purely -- but recovering it means parsing that
proposal, which reintroduces the hindsight concern the structured path was chosen
to avoid. Registered as a **secondary option**, not a switch.

**The cut rule reads only the input.** "The first explicit allocation proposal"
is detected by a frozen regex over the text and never by reading the submitted
deal; it was computed with no model call. Added to `test_hindsight.py` anyway,
because the 0.629 silent-subgroup figure depends on it entirely.

---

## Registrations closing the zero-spend stage

Written before any converter and before the first model call.

### The gate threshold is now one formula

```
majority <= 1/k + 0.30 * (1 - 1/k)
```

A constant may sit at most 30% of the way from **chance** (1/k) to **certainty**.
It reproduces 0.65 at k=2 and is **stricter** than the old flat 0.45 wherever
there are many options (0.40 at k=7, 0.39 at k=8) -- the right direction, since a
constant winning 40% of a seven-way choice is already close to unbeatable.
**Adopted on principle, not on a result**; every type measured so far passes it
exactly as it passed the table it replaces, and one test runs all of them
together so a change to SLACK must face every corpus at once.

`k` is **what is offered at a point, not the size of the label vocabulary**, and
getting that wrong is not academic: Diplomacy's signed labels span 15 values
corpus-wide but a player is offered `2 x (powers in contact) + NONE`, measured at
**mean 8.45, median 9, p90 11, max 13**. Against the vocabulary it fails by
0.001; against what is offered it passes.

```
choice type                majority      k    limit  verdict
Phase CP                     0.6560   7.00   0.4000  FAIL
CaSiNo deal_response         0.8510   4.00   0.4750  FAIL
CaSiNo allocation            0.2337   7.00   0.4000  PASS
Diplomacy pair               0.7594   3.00   0.5333  FAIL
Diplomacy attack-only        0.4395   8.00   0.3875  FAIL
Diplomacy signed             0.3478   8.45   0.3828  PASS
Avalon party vote            0.7083   2.00   0.6500  FAIL
Avalon include/exclude       0.5203   2.00   0.6500  PASS
```

Two things this table now says that the old one did not. The formula **fails
Diplomacy's attack-only label**, which is a stricter reading than the table gave
and reaches, from the label distribution alone, the same conclusion the semantic
argument reached: signed is the right label. And Diplomacy signed passes with
**0.035 of headroom** -- the thinnest margin of any adopted type, recorded so it
is not later described as comfortable.

### Diplomacy — the letter scheme widens to A-M

Signed labels offer up to **13** alternatives at a point (max measured), against
the A-H scheme registered in Phase B. Re-registered as **A-M**. The
letter-permutation invariance and single-token checks in `confirm_forecast.py`
**re-run at the larger set**: positional bias and calibration can behave
differently over thirteen options than over eight, so passing at k=8 is not
evidence at k=13. Top-k = 20 still covers it.

### Avalon — how the choice stream is built

**1. The filter learns from votes; only include/exclude is scored.** Every player
votes on every proposal, which is roughly 7-10 choices per player per game and is
the only thing giving Avalon a genuine per-player sequence. The party vote failed
the gate **as a scoring target** (0.708), and the gate exists to protect the
reported metric -- it does not forbid the filter from taking evidence there.
Registered: **weights update on votes and on include/exclude; the headline
prequential log-score is computed on include/exclude only.** The measured 17-point
spread in yes-rate across hidden roles says the votes carry real but weak
evidence, and they should carry most when conditioned on *who is on the team*,
which is the clause-4 point.

**2. A proposal is one observation, not five.** The 740 points are (leader,
candidate) pairs and a proposal produces about five of them, so 148 proposals is
about **1.5 proposals per leader-game** -- in independent decisions that is close
to CaSiNo's one per person, not a sequence. The leader must also pick exactly k
players, so the pairs are tied together. Same treatment as Diplomacy's pairs,
registered: **batch reveal per proposal**, and the **tempered `1/m` likelihood**
so one proposal counts once however many candidates it ranges over.

**3. Self-pairs are excluded** -- measured, leaders include themselves in
**133/148 = 0.899** of proposals, so those points are near-trivial. (Already
excluded from the 740 above; the rate is recorded as the justification.)

**4. Portfolios reset per game.** Roles are reassigned between games, so carrying
a portfolio across one -- through memory or otherwise -- would carry a previous
game's role into the next. A game is the span.

**5. The same-person / different-role transfer axis is NOT AVAILABLE.** The
released `users` records carry only `index`, `name`, `role`, and `name` is
`player-1`..`player-6` in **every one of the 20 games** -- a positional label, not
a person. Whatever the paper says about how many people played, the data does not
carry the mapping, so separating standing traits from the assigned role cannot be
done here. Recorded the way CaSiNo's missing participant id was recorded, so the
absence is never read as a null result.

**6. Role-revealing fields go in the answer key, and prefixes are cut early.**
Per-message **persuasion and deception strategy self-labels** and the **recorded
beliefs about others' roles** both reveal roles -- a message tagged `deception`
nearly announces an evil player. They go in the answer key and
`test_forecast_leakage.py` covers them by name. Every prefix is cut at **the last
proposal**: the assassination phase and post-game chat discuss roles openly.

### Standing after the zero-spend stage

Three corpora have cleared the gate on a named choice type -- **CaSiNo
allocation** (M1 calibration only), **Diplomacy signed** (sequential, thinnest
margin), **Avalon include/exclude** (sequential, with votes as unscored
evidence). M2 no longer depends on a single dataset, and atla's contamination
probes can now demote it without taking the milestone with it.

### The Diplomacy map — AGPL blocked, so the reconstruction was repaired instead

The canonical source (the `diplomacy` package, Mila) is **AGPL-3.0**. Too viral
to vendor into this repo, so it is not used.

The earlier reconstruction was polluted for a findable reason: it unioned a
SUPPORT order's `to` and `from`, and `from` is the **supported unit's** province,
which need not be adjacent to the supporter. Rebuilt from move geometry only --
a MOVE `A->B` is an edge; a SUPPORT at `S` for `X->Y` gives `S-Y` (the supporter
must reach the target) and `X-Y` (the supported move is itself an edge); convoys
give no edge:

```
OLD (to + from unioned)   75 provinces   mean degree 12.6   median 13   max 22
NEW (move geometry)       75 provinces   mean degree  7.5   median  7   max 14
real board                75 provinces   mean degree ~4.5 per unit type
```

Seven spot checks pass, positives and negatives: PAR-BUR yes, PAR-MUN no,
LON-NTH yes, LON-MOS no, VEN-TRI yes, SEV-BLA yes, SPA-MOS no. Written to
`data/diplomacy/adjacency.json`.

**Still approximate, and the residual bias is in the safe direction.** The union
over fleet and army adjacency is higher than either alone, and multi-coast
provinces collapse to one node, so the map stays somewhat permissive. That
offers a few infeasible alternatives (a C2 cost) and **inflates k** -- and
because the gate limit falls as k rises, an over-permissive map makes the
threshold **harder** to pass, never easier. Recorded so the direction is not
re-derived later.

---

# OUTCOME — Phase B, the forecaster backend. **THE PRIMARY PATH IS DEAD ON THIS KEY.**

Run before the M1 budget, which is the reason this phase exists. Cost: eleven
calls, all of which failed.

**Every Gemini model reachable on this key refuses log-probabilities.**

```
gemini-2.5-flash        400  "Logprobs is not enabled for models/gemini-2.5-flash"
gemini-2.5-flash-lite   400  "Logprobs is not enabled for models/gemini-2.5-flash-lite"
gemini-3.5-flash        400  "Logprobs is not enabled for this model"
gemini-3.5-flash-lite   400  "Logprobs is not enabled for this model"
gemini-3.6-flash        400  "Logprobs is not enabled for this model"
gemini-3.8-flash        400  "Logprobs is not enabled for this model"
gemini-3.1-flash-lite   400  "Logprobs is not enabled for this model"
gemini-flash-latest     400  "Logprobs is not enabled for this model"
gemini-2.0-flash        404  no longer available
gemini-2.5-pro          404  no longer available to new users
```

The SDK is not the problem: `google-genai` exposes `response_logprobs` and
`logprobs` on `GenerateContentConfig`, and the request is well formed. The
**service** declines. 32 `generateContent` models are visible on this key and
none of the plausible text models accepts the parameter.

This falsifies the plan's central technical bet -- "the forecast answers with a
single option letter, and one call returning top-k log-probs at that position
gives an exact normalised distribution". The call does not exist here.

**The pre-registered branch applies, and it was written for exactly this:**

> If `response_logprobs` does not work, the honest options are a coarse declared
> ranking over alternatives, or pause -- not K = 20 frequencies.

And the arithmetic that rules out sampling is unchanged: at `eps_frac = 0.12` and
n = 6 the epsilon floor is ~0.02, so a sampling resolution of 1/K needs **K > 50**
per particle per choice point, which is not affordable at ~600 points x 6
particles x 2 orders.

**Nothing downstream may be tuned**: `forecast_backend.json` is not written, and
the standing rule that no forecast threshold is set until it shows
`confirmed: true` now blocks M1.

**What is NOT established by this.** Whether a different provider would work
(OpenAI's chat completions expose `logprobs` with `top_logprobs` up to 20, which
is precisely the shape required; no key for it here, and the `openai` package is
not installed). Whether Vertex AI's Gemini would work -- the restriction appears
to be on the Developer API rather than on the models, but there are no GCP
credentials here to test it. Whether a declared-ranking fallback is good enough
to carry the filter; that is measurable and has not been measured.

**THE KEY IS NOT THE PROBLEM, and this was checked rather than assumed.** The
same key, same model, same call, with the flag as the only difference:

```
plain generation, gemini-2.5-flash          -> returns text        KEY WORKS
+ response_logprobs=True, nothing else changed -> 400 INVALID_ARGUMENT
+ response_logprobs=True alone                 -> 400 INVALID_ARGUMENT
+ logprobs=0                                   -> 400 INVALID_ARGUMENT
  logprobs=N without response_logprobs         -> 400 "logprobs can only be ..."
```

Fifteen models tested in total, now including the pro and preview tiers --
`gemini-3.1-pro-preview`, `gemini-pro-latest`, `gemini-3-flash-preview`,
`gemini-3.1-flash-lite-preview`, `gemini-3.7-flash` -- all returning
"Logprobs is not enabled for this model". The credential authenticates and
generates; the **feature** is not offered on the Gemini Developer API surface it
reaches. Log-probabilities are a Vertex AI capability.

One detail worth passing on rather than acting on: the wording "Logprobs is not
enabled **for this model**" reads like a per-model or per-tier entitlement rather
than a hard absence, and the credential in `.env` is not the classic `AIzaSy...`
39-character API-key shape. A standard API key on a billing-enabled project, or
the same models through Vertex, may behave differently. Untested here because
there are no other credentials.

**A second, smaller finding, free from the same probe.** Asked to "answer with
one letter" at `max_output_tokens=4` with thinking off, `gemini-2.5-flash`
replied `"The question asks Alex"` -- it did not comply with the format at all.
The single-letter constrained answer that the whole letter scheme rests on needs
its own check **whatever** backend supplies the probabilities, and that check is
now part of `aligned` rather than an assumption inside it.

This is a resourcing decision, not a methods decision, and it is recorded here
rather than worked around.

### Addendum — the credential hypothesis is falsified

The first probe ran on a 53-character `AQ.Ab8...` credential, and the outcome
noted that a standard `AIzaSy...` API key on a billing-enabled project might
behave differently. **It does not.** Re-run on a proper 39-character `AIzaSy`
key:

```
new key, plain generation, gemini-2.5-flash   -> 'OK'          KEY VALID
gemini-2.5-flash / -flash-lite                -> 400 Logprobs is not enabled
gemini-3.5-flash / 3.6-flash                  -> 400 Logprobs is not enabled
gemini-flash-latest / gemini-pro-latest       -> 400 Logprobs is not enabled
legacy google.generativeai SDK, same request  -> 400 Logprobs is not enabled
gemini-2.5-flash-001 / -preview-05-20         -> 404 not found for this API version
```

Both SDKs, every model family and tier, every naming variant. **Log-probabilities
are absent from the Gemini Developer API as such**, not gated behind a key format
or a project tier. One of the four standing options is therefore closed, which is
worth the two probes it cost: a Gemini credential of any shape will not unblock
this.

The remaining options are unchanged -- **Vertex AI** (same models, where the
capability is documented, needs GCP credentials), an **OpenAI key**
(`top_logprobs` up to 20, exactly the required shape), the pre-registered
**declared-ranking** fallback, or **pause** the forecaster while the converters
proceed, since none of them depend on it.
