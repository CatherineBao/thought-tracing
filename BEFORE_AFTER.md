# ThoughtTracing particle filter: before and after

What changed on `nonlinear-particles`, measured. Every number is from a run on
`data/musing` narrative corpora, N=8 unless stated, and every "before" is the released
upstream code at `966eb27`.

---

## Summary

| | before | after |
|---|---|---|
| propagation sampling | frozen — **1/5** unique outputs at "temperature 0.7" | **5/5** unique |
| likelihood verdicts | **100% parse failure** (truncated before the answer) | **0** parse failures |
| verdict range | two values only: `a` or `f` | `a`, `b`, `d`, `e`, `f` |
| parse failure handling | scored `0.001` = "Very Unlikely" (**sign-flipped**) | distinct sentinel, excluded from ESS |
| weights across steps | **memoryless** — posterior ESS identical to likelihood ESS on 28/28 steps | accumulated; separation **+0.30** |
| rank stability (Kendall τ) | 0.687 | **0.926** |
| commitments at run end | **8 → 1–4** (median 3), monotone, **zero births** | **8 → 7–8**, with mid-run births |
| distinct commitments per run | 8, decaying | **16** (8 initial + 8 born) |
| lineage | `backtrack()` raised `AttributeError`; chain never grew | works; verified monotone across 5 runs |
| instrumentation | console text only; verdicts never reached disk | per-step JSONL, offline HTML viewer |
| thresholds | 4 inherited constants | 4 derived from measured distributions |
| **belief-tracking quality** | **at ceiling on 3 ground-truth metrics** | **unchanged — still at ceiling** |

The last row is the honest qualifier and is expanded at the end.

---

## 1. The method was partly inoperative, not just untuned

Four defects meant components did not do what the paper describes.

**Propagation ran greedy.** `batch_interact` accepted a `temperature` and called
`batch_generate(..., temperature=None)`, so the value never reached the API and the default of 0
applied. `rejuvenate_hypotheses` passes `temperature=1` specifically to diversify paraphrases; it
ran at 0 and returned identical text. The diversity-repair mechanism could not repair anything.

> five propagation calls on a real 5,206-char prompt — **before: 1/5 unique. after: 5/5.**

**The likelihood evaluator never answered.** `max_tokens=512` on a reasoning model: thinking
consumed 488 tokens, leaving 20 visible, truncating at ~88 characters — before the `Answer:` line
the parser needs. Every verdict was a parse failure. The scores that survived were the loose matcher
finding `(a)`-shaped substrings in truncated prose.

> a 28-step run produced exactly two verdict values, `3.0` and `0.001`, and never the four middle
> buckets. That reads as a decisive evaluator. It was noise.

**Parse failure was sign-flipped.** The failure return value `0.001` is also the score for bucket
`f`, "Very Unlikely". An unparsed **"Very Likely"** was scored as **"Very Unlikely"** — the maximum
possible error, and it inflated apparent discrimination.

**Propagation and weighting read the same observation.** `setup_propagation` put the step-*k* action
into the prompt that writes the hypotheses; `weigh()` then scored those hypotheses on predicting
that same action, with the action pruned from the evaluator's view. Every particle was fitted to its
own test, so unanimous "Very Likely" was the correct answer to the question being asked.

> one step, N=4 — **action visible: `a,a,a,a`, ESS 1.000 (exactly uniform). action held out:
> `a,a,b,b`, 0.94.**

Also fixed: `use_perception_only` (`AttributeError` on first rejuvenation), `backtrack()`
(`AttributeError` then `TypeError`, never callable), frozen lineage (resampling copied the *parent's*
pointer rather than the particle), `dump()` silently discarding every verdict and reasoning, and
`AsyncLlama3LogProbAgent` (`NameError` for all Llama models).

---

## 2. Weights now accumulate, so trends can exist

Before, `update_weights` **overwrote** the prior with each step's fresh likelihood. Strengthening and
weakening could not exist as trends, only as single-step blips the next step erased. Posterior ESS
was identical to likelihood ESS on **28 of 28 steps** — the two lines on the diagnostic plot were
the same line, and the gap that appeared occasionally was a parse-failure artifact, not memory.

After: `w_t ∝ w_{t-1}^α · L_t^β` (α=0.85, β=1.0, floor ε=0.12/N).

| | before | after |
|---|---|---|
| steps where posterior ≠ likelihood ESS | 0 / 28 | **27 / 28** (step 0 correctly locked) |
| median separation | 0.000 | **+0.30** (range 0.20–0.38, run-to-run spread 0.167) |

This works because the per-step signal is **ordinal, not magnitude**. Rank order is highly stable
(τ = 0.926) while the reported weights barely differ; accumulation is what converts consistent weak
ordering into usable separation.

A scorer change was needed first. Four framings were tested on identical inputs:

| framing | on genuinely distinct hypotheses |
|---|---|
| independent 0–100 scoring | `90, 90, 90, 90` |
| sum-to-100 allocation | `28, 26, 24, 22` |
| "eliminate the failures" | `100, 100, 100, 100` |
| **rank first, then score** | **`90, 70, 65, 60`** |

Committing to a strict **order** before producing numbers was the only framing that discriminated —
a cognitive difference, not a formatting one.

---

## 3. The filter had a diversity half-life, and its own health metric could not see it

Adding ancestral lineage (`root_id`) made it possible to count distinct *commitments* rather than
distinct particles.

**Before — commitments decay multiplicatively and never recover.** Roots are founded only at
initialization; thereafter they can only die.

```
final commitments across 5 runs:  8→2, 8→3, 8→4, 8→1, 8→3     median 3
per-resample ratio:               0.375 – 0.50
monotonicity violations:          0 / 5
```

Worst case: **a single ancestor in 28 steps**, on a corpus whose contexts run longer than that.

**The standard metric reports this as healthy**, because particle ESS counts five copies of one
winner as five independent particles:

| | particle ESS | root-mass ESS |
|---|---|---|
| late in a collapsed run | **0.82 – 0.98** | **0.16** (~1.3 effective hypotheses of 8) |

At a resample, particle ESS *jumps* (0.41 → 0.97, reading as recovery) while root-mass ESS *falls*
(0.53 → 0.27). Roughly **71–75%** of the apparent post-resample recovery is an artifact of counting
duplicates as independent.

**After** — an anchored perturbation that mints new commitments on measured collapse, plus split:

```
final commitments:   8→7, 8→8, 8→8
root-mass recovery:  5/5 repair events (+0.23 to +0.49)
```

---

## 4. Four inherited thresholds, all mis-placed, all recalibrated

The most transferable finding. Each constant was set by intuition about what "degenerate" or
"duplicate" means in general, and each landed inside the high-density region of this data.

| threshold | inherited | measured | consequence | now |
|---|---|---|---|---|
| resample | ESS < N/2 (0.50) | mass at 0.55–0.75 | crossings near-ties; one run fired at **0.49 vs 0.50**; resample count varied 1–2 on *identical* contexts | **0.25** (0 marginal crossings at N=8 **and** N=12) |
| merge | cosine ≥ 0.90 | cosine spans **0.846–1.000** | absorbs **26 / 34 known-different pairs** — merge would destroy the diversity it is paired with | **Jaccard p95**, resolved per run |
| split | weight > 3× mean **and** mid-pack rank | corr(weight, rank) = **−0.866**; all 57 particles above 3× mean were rank 1 | fires **once in 476** particle-steps — structurally impossible | prior/likelihood **disagreement**, ≈1.60× mean, **2.7 firings/run** |
| likelihood | 6 buckets → softmax over {3, 2.5, 2, 1, 0.5, 0.001} | — | scores treated as logits; a:f ratio `e^2.999 ≈ 20` vs the 9:1 advertised | linear normalization of a forced ranking |

**Why it happens.** Every quantity is measured over hypotheses about *one agent in one scene,
written in similar prose by the same model*. That compresses the distributions hard — embedding
cosine never drops below 0.85 — so a threshold chosen to sound conservative lands mid-mass.

Two cautions found the hard way. The **empty-band heuristic is distribution-dependent**: it worked
for ESS (bimodal — warm-up dips vs stable band) and not for weight (unimodal decaying, no gap), so
split had to be set by tail fraction instead. And **relative definitions do not transfer across N**:
`1/3` is clean at N=8 and has 10 observations within ±0.05 at N=12, because normalized ESS shifts
down as particles compete for the same mass.

---

## 5. What it looks like now

34 steps, full stack. Each letter is an ancestral commitment; bar width is share of belief mass.

```
  0 |AAAAAAAABBBBBBBCCCCCCDDDDDEEEEFFFFGGGHH  |                  +ABCDEFGH
  4 |BBBBBBBBBBBBBBBAAAAAAAAAAEEEEEDDDDDFFFFF | resample         -CGH
  9 |BBBBBBBBBBBBBBBBEEEEEEEEEAAAAAAADDDDDFI  | perturb          +I
 13 |EEEEEEEEEEEEEEEEJJJJJJJKKKKKKKBBBBBBIIFFD| split,merge      +JK -A
 25 |BBBBBBBBBBBBBBBBBJJJJJJJJJJJJJJJJJFFFFFF | resample,perturb -E
 29 |BBBBBBBBBBBBBBBBBMMMMMMMMJJJJJJFFFFFLLNN | perturb          +LMN
 31 |BBBBBBBBBBBOOOOOOPPPPPPQQQQQQFFFFFFMMMMMM| resample,perturb +OPQ -JLN
 33 |BBBBBBBBBBOOOOOOOOPPPPPPPQQQQQQFFFFFMMMM |
```

At step 0 every commitment concerns the appeal — *"To enable the appeal"*, *"To expedite the
appeal"*, *"To appear generous"*. By step 31 the dominant ones are *"To maintain a public image of
unwavering integrity"* and *"To secure lasting personal recognition"* — **born at step 31**, tracking
the subject's shift from managing an appeal to securing a confirmation.

Split fired at step 13, refining one commitment into two mutually exclusive children, with merge
absorbing another in the same step so the population cap never bound.

| non-linear progression | |
|---|---|
| reversals (long-lived commitments) | 7 |
| argmax changes | 6 |
| splits / merges | 1 / 1 |
| perturbations | 20 |
| commitments born mid-run | **8** |

**Before, all of these were zero by construction** — commitments could only die.

---

## 6. What did not change

Three quality metrics with ground truth in the corpus. **All were already at ceiling for the
baseline**, and remained there:

| metric | ground truth | before | after |
|---|---|---|---|
| false attribution | gold gaps: content the target was absent for | 0 | 0 |
| false belief | Odyssey `disguised_as` | 10/10 | 10/10 |
| goal change | subject's goal shifts, both sides quotable, mutually exclusive | 10/10 | 10/10 |

The third was built as a falsification test. The prediction: a commitment nobody proposed at step 0
should be unrepresentable without minting. Two routes were run — one preserving its initial
commitments with **0 roots minted**, one collapsing and repairing with **12–14 minted**. Both scored
identically.

**The mechanism:** the anchor pins a particle's *commitment*, but the **belief text propagates
freely** with the context, and the summary is built from texts. A particle anchored to "prevent the
appeal" can hold a belief that has drifted entirely to confirmation politics.

**Scope of the null.** On these corpora, for the dimensions checkable against ground truth, the work
of belief tracking happens **upstream of the particle population** — in perception tracking, which
bounds what a target can know, and in propagation, which updates belief content. The filter's
diversity operates where those checks do not reach. This is consistent with the paper's own ablation
deleting the weight update for 1.8–4.3 points.

What would change it: a task where the correct belief is genuinely *ambiguous mid-trajectory* and
resolves late, so carrying several live hypotheses is load-bearing rather than redundant. No such
task was found in these corpora, and the structure works against it — contexts with substantial
unknowable content have an absent target, who therefore speaks less, yielding short trajectories.
Exactly **one** gold context cleared both thresholds.

---

## Bottom line

The system is **correct where it was broken** — several components were inoperative rather than
badly tuned, and those fixes apply to anyone running this code. Its **internal dynamics are
substantially better**: weights accumulate, commitments survive, thresholds are derived rather than
assumed, and the whole run is now instrumented and inspectable.

**Whether those dynamics improve the output is unproven.** The honest position is a working filter
with a demonstrated capability — maintaining and generating hypothesis diversity across a
trajectory — and no evidence yet that the capability pays on these tasks.
