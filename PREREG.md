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

_pending_
