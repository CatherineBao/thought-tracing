# ThoughtTracing: upstream defects, a diversity half-life, and a null result

Work on a modified ThoughtTracing particle filter (branch `nonlinear-particles`), run against
narrative dialogue corpora. Four upstream bugs are reproducible on the released code and affect
anyone running it. The rest is specific to our modifications.

---

## 1. Upstream defects

These are independent of anything we changed. Each is reproducible on upstream
`skywalker023/thought-tracing` at `966eb27` and worth reporting separately.

### 1.1 `batch_interact` silently discards temperature

`agents/base.py:87` and `agents/gemini.py:92` accept a `temperature` argument and then call
`batch_generate(..., temperature=None, max_tokens=None)`. The caller's value never reaches the API.
`_set_default_args` (`agents/base.py:44`) pins the fallback to `0`.

Consequence: every batched call runs greedy. `rejuvenate_hypotheses` (`tracer.py:659`) passes
`temperature=1` specifically to diversify paraphrases; it ran at 0 and returned identical text, so
the diversity-repair mechanism could never repair anything.

`AsyncConversationalGPTBaseAgent` overrides `batch_interact` correctly (`gpt.py:216`), so the bug
is Gemini/base-only. Our runs were Gemini.

**Reproduction.** Five propagation calls at `temperature=0.7` on a real 5,206-char prompt:

| | logged temperature | unique outputs | Jaccard diversity |
|---|---|---|---|
| before fix (effective 0.0) | 0.0 | **1/5** | 0.000 |
| after fix | 0.7 | **5/5** | 0.833 |

**Fix.** Forward `temperature` and `max_tokens` in both call sites.

### 1.2 Thinking models truncate before the answer, and it reads as a verdict

`prompt_likelihood` (`tracer.py:611`) requests `max_tokens=512`. On a reasoning model, thinking
tokens count against `max_output_tokens`.

| `max_tokens` | finish_reason | thinking tokens | visible tokens | contains `Answer:` |
|---|---|---|---|---|
| 512 (upstream) | **MAX_TOKENS** | 488 | 20 | **no** |
| 2048 | STOP | 1254 | 241 | yes → `Answer: (a)` |

At 512 the response truncates at ~88 characters, before the `Answer:` line the parser needs. **Every
verdict is a parse failure.** The scores that survive are artifacts: the matcher at `tracer.py:581`
searches the whole prose for `(a)`-shaped substrings, so a truncated fragment sometimes matches.

Observed on a 28-step run before the fix: verdicts took exactly two values, `3.0` and `0.001`,
never the four middle buckets — which looks like a decisive evaluator and is actually noise.

**Fix.** Raise the budget, and/or disable thinking so `max_tokens` means what the prompts assume.
Log `finish_reason`; a silent `MAX_TOKENS` is invisible otherwise.

### 1.3 Parse failure is indistinguishable from "Very Unlikely", and sign-flips

`map_response_to_score` (`tracer.py:583`) returns `0.001` when nothing parses. `0.001` is also the
score for bucket `f`, "Very Unlikely (Below 10%)".

An unparsed **"Very Likely"** is therefore scored as **"Very Unlikely"** — not lossy, *sign-flipped*,
the maximum possible error. It also inflates apparent discrimination: a step where nothing parsed
looks like unanimous confident disbelief.

Compounding it, the answer span is extracted with `response.split("Answer:")[-1]`, which returns the
**entire response** when `Answer:` is absent — so the bucket matcher then scans full prose. In
practice models often answer `The final answer is $\boxed{f}$`, which contains no `Answer:` at all.

**Fix.** Return a distinct sentinel; record parse failures separately; exclude them from any ESS or
entropy computed over the weights rather than imputing a value. Harden the extractor for `\boxed{}`,
`**Answer**`, and `final answer is`.

### 1.4 Propagation and weighting read the same observation

The most consequential one, and it is not a bug in the ordinary sense — it is a departure from the
SMC semantics the method claims.

`setup_propagation` (`tracer.py:545-559`) puts the step-*k* action into `new_context`, so every
particle is rewritten **with sight of the action**. `weigh()` then scores those particles on how
well they predict **that same action**, with the action pruned from the evaluator's context
(`tracer.py:590`) — so the evaluator cannot tell the hypotheses already encode it.

Every particle is fitted to its own test. Unanimous "Very Likely" is the *correct* answer to the
question actually being asked.

**Reproduction**, one step, N=4:

| propagation sees the action | verdicts | normalized likelihood ESS |
|---|---|---|
| yes (upstream) | `a, a, a, a` | **1.000** — exactly uniform |
| no (held out) | `a, a, b, b` | 0.94 |

**Fix.** Build `new_context` from the state and its perception note only; the action still enters
`context_history`, so later steps see it. Correct SMC ordering is propagate on the prior, then
weight by the held-out observation.

**Caveat worth stating:** on a full 28-step run this fix closed **nothing measurable** — median
normalized likelihood ESS stayed at 1.000. It is a correctness fix, not a performance one.

---

## 2. The filter has a diversity half-life, and its own health metric cannot see it

We added lineage tracking: each particle carries a `root_id` for its ancestral hypothesis, so
distinct *commitments* can be counted rather than distinct particles.

**Root count decays multiplicatively and never recovers.** Roots are founded only at
initialization; thereafter they can only die. Across 5 clean runs (N=8, 28 steps):

```
final roots: 8->2, 8->3, 8->4, 8->1, 8->3      median 3
per-resample ratio: 0.375 - 0.50
monotonicity violations: 0/5
```

Worst case is **a single ancestor in 28 steps**, on a corpus whose gold contexts run longer than
that. The filter's diversity half-life is shorter than the trajectories it is meant to trace.

**The standard metric cannot detect this.** Particle ESS counts five copies of one winner as five
independent particles. Measured on the same steps:

| | particle ESS | root-mass ESS |
|---|---|---|
| late in a collapsed run | **0.82 – 0.98** ("healthy") | **0.16** (~1.3 effective hypotheses of 8) |

At a resample, particle ESS *jumps* (0.41 → 0.97, reading as recovery) while root-mass ESS *falls*
(0.53 → 0.27). Roughly **71-75% of the apparent post-resample recovery is a measurement artifact**
of counting duplicates as independent.

Two further notes for anyone measuring this:

- **Root count is a lower bound on diversity.** Copies sharing a root drift substantially in text —
  within-root divergence 0.53-0.70. "One ancestor" means one commitment explored several ways.
- **Resampling resets weights to uniform**, so per-particle weight trajectories restart at every
  resample and "strengthens over the run" is unobservable. Count reversals on *ancestral mass*,
  which is invariant to that encoding choice. Verified on a synthetic resample: root mass runs
  0.30 → 0.45 → 0.60 → 0.75 → 0.75 → 0.80 → 0.85 monotone across the boundary, while the lineage
  weight drops 0.75 → 0.25 and registers a false reversal.

---

## 3. Every inherited threshold landed inside the high-density region of its own distribution

Four constants, four independent recalibrations, same failure mode each time. This is the most
transferable result here.

| constant | inherited value | measured distribution | consequence |
|---|---|---|---|
| resample trigger | ESS < N/2 (0.50 normalized) | mass at 0.55-0.75 | crossings are near-ties; one run fired at **0.49 vs 0.50**. Resample count varied 1-2 on *identical* contexts. |
| merge | embedding cosine ≥ 0.90 | cosine spans **0.846-1.000** | absorbs **32/40** real pairs, including **26/34 known-different** — merge would destroy the diversity it is paired with. |
| split | weight > 3× step mean **and** mid-pack likelihood rank | corr(weight, rank) = **-0.866**; all 57 particles above 3× mean were rank 1 | fires **once in 476** particle-steps. Structurally impossible: weight *is* accumulated likelihood, so "carries mass" entails "top-ranked". |
| likelihood scoring | 6 buckets → softmax over {3, 2.5, 2, 1, 0.5, 0.001} | — | scores treated as logits; a:f ratio `e^2.999 ≈ 20` vs the 9:1 the prompt advertises. |

**The structural explanation.** These constants are set by intuition about what "degenerate" or
"duplicate" means in the abstract. But every quantity here is measured over a *population of
hypotheses about one agent in one scene, written in similar prose by the same model*. That
compresses the distributions hard — embedding cosine never drops below 0.85 — so a threshold
chosen to sound conservative lands mid-mass.

**What worked instead.** Derive the cut from the measured distribution and place it in a
low-density region, then log the resolved value per run:

- Resample: 0.25 sits in an empty band (0 observations within ±0.05, at **both** N=8 and N=12).
  0.333 is clean at N=8 and has **10** observations within ±0.05 at N=12 — the calibration does not
  transfer across N, because normalized ESS shifts down as particles compete for the same mass.
- Merge: use a **copy-history control** — resample duplicates are known-same-hypothesis, everything
  else known-different. Jaccard AUC **0.790** vs embedding cosine **0.716**. The intuitive choice of
  embeddings was the worse instrument on this data.
- Split: re-express the *intent* rather than tune the threshold. "Carries mass but the action
  doesn't confirm it" is prior/likelihood **disagreement** — top-quartile weight, below-median
  current likelihood. Fires 0.7-2.7/run.

**One trap specific to ESS.** It cannot be calibrated as a live percentile: the trajectory drifts as
the accumulator warms (~21 steps at α=0.85), so an expanding-window p10 reads **0.49-0.53** early
against a post-warm-up value of **0.152** — firing exactly when resampling is least warranted.
Calibrate offline from a prior run. Merge escapes this because its percentile is over within-step
pairwise similarities, which are stationary.

---

## 4. The null: diversity was preserved; belief tracking did not measurably improve

### What the modifications achieved

Accumulation (`w_t ∝ w_{t-1}^α · L_t^β`, α=0.85, β=1.0), a rank-first comparative scorer, goal-level
seeding, per-particle commitment anchors, and an anchored perturbation that mints new ancestral
roots on diversity collapse.

| | baseline | modified |
|---|---|---|
| final root count | 2, 3, 4, 1, 3 (median 3) | **7, 8, 8** |
| parse failures | 7 per run | **0** |
| rank stability (Kendall τ) | 0.687 | **0.926** |
| likelihood/posterior separation | 0 (identical on 28/28 steps) | **+0.30** |

Rank-first scoring was the only framing of four that discriminated at all: independent 0-100 scoring
returned `90,90,90,90`; sum-to-100 allocation returned `28,26,24,22` on *identical* inputs. Asking
the model to commit to a strict **order** before producing numbers was the difference — a cognitive
change, not a formatting one.

### What it did not achieve

Three ground-truth-checkable quality metrics, all saturated for the **baseline**:

| metric | ground truth | baseline | modified |
|---|---|---|---|
| false attribution | gold gaps: content the target was absent for | **0** | 0 |
| false belief | Odyssey `disguised_as`: Minerva speaking as Mentor | **10/10** | 10/10 |
| goal change | Strauss: "prevent the appeal" → "secure Commerce confirmation", mutually exclusive, both quotable | **10/10** | 10/10 |

The third is the informative one, because it was designed as a falsification test.

**The prediction.** A commitment nobody proposed at step 0 should be unrepresentable without
minting. We ran two routes on a 34-step trajectory: one that preserves its initial commitments and
mints nothing (verified: 0 roots minted), one that collapses and repairs (verified: 4-6 resamples,
**12-14 roots minted**).

**The result.** Both scored `tracks_late 10, stuck_early 0`, with cited evidence.

**The mechanism.** The anchor pins a particle's *commitment*, but the **belief text propagates
freely** with the context, and the final summary is built from texts. A particle anchored to
"prevent the appeal" can hold a belief that has drifted entirely to confirmation politics. The
within-root divergence of 0.65-0.70 reported in §2 is the same mechanism — we recorded it as a
measurement caveat before recognising it as the answer.

### Scope of the null

This is **not** "particle diversity doesn't matter". It is:

> On these corpora, for the three ground-truth-checkable dimensions we could construct, the work of
> belief tracking is done **upstream of the particle population** — by perception tracking, which
> bounds what a target can know, and by propagation, which updates belief content against context.
> The filter's population diversity operates where those checks do not reach.

Consistent with the paper's own ablation, which deletes the weight update for 1.8-4.3 points.

**What would change the conclusion.** A task where the correct belief is genuinely *ambiguous
mid-trajectory* and resolves late — so that carrying several live hypotheses is load-bearing rather
than redundant with a single well-propagated one. We did not find such a task in these corpora, and
the structure works against it: contexts with substantial unknowable content have an absent target,
who therefore speaks less, yielding short trajectories. Exactly **one** gold context in the corpus
cleared both thresholds.

---

## Method notes

Four evaluations produced invalid numbers before this was fixed structurally: the evaluator
re-derived its own context/target selection and picked differently from the runner — judging one
target's beliefs against another's ground truth. The fix is for the evaluator to read
`target_agent` and `context_id` from the run's own metadata and refuse to guess.

Related: a run that wrote **zero steps** still recorded a completion marker, so four failures were
invisible until an analysis found empty files. Guard on output length, not on exit status.

Superseded code paths were kept behind flags rather than deleted — the six-bucket scorer, the
flattening scorer variants, the paraphrase perturbation, the circular-propagation switch. Each is
the evidence for a finding, and deleting them would make the findings unreproducible.
