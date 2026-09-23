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
| **smeeple consent is not documented** | smeeple stays internal and de-identified; the worked example uses a non-smeeple case |
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
