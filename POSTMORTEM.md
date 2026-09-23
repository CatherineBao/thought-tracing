# Two-timescale tracing: what was tried, and why it stopped

Short version for anyone picking this up later. Full evidence is in
`PRECHECKS.md` as numbered findings P-1 to P-14; this file is the map.

## The goal

Model the long-term mental state of people in an organisation, on
non-cherry-picked Slack transcripts. The motivating case is Bloomfield's
`blueberry_size` thread, where "size" means commercial caliber grading for
export on one side and green-berry sizing for harvest timing on the other, and
neither side notices.

The plan was a two-timescale filter: a slow **standing** board per person that
survives across the corpus, plus fast **episodic** boards that open when two
people diverge, so a misalignment could surface without evicting the long-term
picture.

## What was tried

| # | Idea | Why it failed | Finding |
|---|---|---|---|
| 1 | Episode trigger from the **mass** distribution over v4 standard categories | Measures a dyad *trait*, not an episode. Separation is constant on a stable board; persistence is free. | P-8 |
| 2 | Episode trigger from **likelihood margins** instead of mass | Margins move only 1.1–1.5x as much as mass, are heavily quantised, and the argmax sits in one bucket 44 times in 52. | P-8 |
| 3 | Episode trigger from the **undefined-lean rate** (no commitment beats the null) | Lift signs disagree by party: +0.16 / −0.18 / −0.07 against a 0.09 floor. Control dyad beat the test dyad. | P-10 |
| 4 | **Transcript-side** detector: compare the words each party puts around a shared term, with a permutation null | Raw version detected writing style, not meaning (test 47% vs control 48%). Normalised, `size` ranked 39/68 — *below* the median term. | P-11 |
| 5 | Raise **alpha** (the prior's exponent in the weight update) to stabilise the board | Churn went **up** at every alpha, all within noise. Seed spread at alpha=0.98 was 43.3 per 100, bigger than the effect. | P-14 |

## The one finding that explains the rest

The filter's leader is unstable, and the instability is **upstream of
everything that was built**.

```
argmax churn: 55.6 per 100 steps (median over 52 runs, zero runs at zero)

  operators account for            <= 30%   (P-13)
  reweighting accounts for            70%   (P-13)
  ...and inside that, the SCORER picks a different
     best hypothesis on 72% of steps           (P-14)

likelihood rank movement: 1.79 places/step (random shuffle = 2.7)
weight   rank movement: 1.03 places/step
```

The weight update is doing its job — it damps 1.79 places of likelihood
movement down to 1.03. But you cannot make a stable leader out of an input
that reshuffles at 72%, and pushing the damping harder only blunts surprise
detection.

So stratification, cross-run memory and alpha all sit *downstream* of the
problem. That is why five different designs each looked reasonable and each
came back null.

This is consistent with `FINDINGS.md` §4 (recoverable at
`git show a84d09a^:FINDINGS.md`): on these corpora "the work of belief
tracking is done upstream of the particle population". Rank-first scoring
raised Kendall tau from 0.687 to 0.926, but tau is rank *correlation* — an
argmax can flip freely while tau stays high, and it does.

## Two other things that did not survive contact with unselected data

**The v4 standard axis is not a trait.** On the hand-picked 6-set span Rovani
looked strongly `CUSTOMER` (0.65) and separated from Wolf at TV 0.70. On 42
unselected sets from the same topic, his `CUSTOMER` mass is 0.00–0.19 and the
separation drops to 0.22 — while the *control* pair rises to 0.31. The same
person moved more across spans (0.69) than two different people differed
within one (0.36). It is stable across **seeds** and unstable across **spans**;
seed spread is the floor every existing control checks against, and span
variation is roughly 10x larger. (P-2, P-9, P-10)

**Commitments do not recur across runs.** Two runs on the same person, same
config, adjacent spans: 44 commitments, **zero** exact overlap, zero stem
overlap, zero pairs clearing a 0.6 Jaccard floor. The one near-match is
"Ensure labeling quality" / "Ensure clear labeling" — the same commitment to a
reader, invisible to any lexical rule. So the memory store consolidates
nothing. (P-12)

## What was removed in the cleanup

Dead on the evidence above, so it is not sitting in the repo looking usable:

- **`memory.consolidate()`** and its thread-adjacency machinery. P-12 proved it
  can never fire. Rebuild only after semantic matching exists, not before.
- **The horizon / two-timescale machinery** in `hypothesis.py` (`horizon`,
  `can_transition`, `stratum_resample`, `indices_of`, `compute_ess(only=)`)
  and `test_horizon.py`. Stage 3 is closed, so nothing can construct an
  episodic particle and none of it was reachable. `hypothesis.py` is tracked,
  so `git show` recovers it if stratification is ever revisited — but P-14
  says it targets at most 30% of the churn, so it probably should not be.
- **`sweep_episode.py`** — analysed a closed stage.

## What still works and is worth keeping

- `trace_log.py` — **`margins` and `lean_undefined` are now logged.** They were
  computed on every baseline-scored step and thrown away, which is why no run
  on disk could be replayed to sweep a threshold. (8 tests)
- `memory.py` + wiring — a verified durable per-person store: cross-process
  accumulation, config isolation, atomic writes, no stored weight reachable as
  a prior, and restored commitments marked so they are not mistaken for
  something the filter found on its own. It is a correct **log**. (36 tests)
- `new_particle_id` widened to 64 bits (32 bits gives 25% collision at 50k ids).
- `sweep_alpha.py`, `referent_split.py`, `verify_memory.py` — analysis tools,
  runnable offline; the reproduction path for P-11, P-12 and P-14.

## If you pick this up

**Do not** retry a signal design against the current labels. Every attempt
above was scored against a positive region that is contaminated: the six
"color case" sets are mostly Agrovision hand-counts and a scrum blog link, not
the 2-Purple dispute. With a bad label you cannot tell "this design is bad"
from "this label cannot see it". This was routed around twice and both times
the result was uninterpretable.

**Do first:**

1. **Stage 0 labelling** — hand-label the split-bearing turns in
   `blueberry_size`, two labellers, kappa reported, frozen before any run.
   ~a day. It is the prerequisite for every other question here.
2. **Look at the scorer**, not the population. The open question is whether
   per-step comparative ranking of eight similar hypotheses is well-posed for
   an LLM at all, or whether the filter needs a different evidence signal.
   Everything downstream is blocked on that.

**Known-dead, do not rebuild:** episode detection from board fit or from
standard distributions; lexical matching for cross-run commitment identity;
alpha as a stability lever.

**Open, not tried:** semantic matching for commitment identity (embeddings or
a model judge). `merge_equivalent_anchors` already uses a model rather than a
threshold inside a run, for exactly this reason — the store never inherited
it. This is the cheapest untried thing, but note it only fixes P-12, not the
scorer problem underneath.
