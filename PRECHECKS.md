# Pre-check results: two-timescale tracing

Six cheap, read-only checks run before any code, each able to kill or redirect a
stage. Four of six came back against the plan. Measured 2026-09-22 against
`data/musing/bloomfield_dialogue.json` (402 sets / 2,190 turns) and the existing
`musing_out/runs/**` logs.

| # | check | result |
|---|---|---|
| P-1 | `source.topics` granularity | **FAIL** |
| P-2 | v4 categories separate the parties | **PASS** (purple); **unresolved** (size) |
| P-3 | stem canonicalization as an identity key | **FAIL** |
| P-4 | roles in `<corpus>_speakers.json` | confirmed absent |
| P-5 | gate drop rate >= 0.6 | **FAIL** |
| P-6 | turn counts and call budget | counted |

---

## P-1 — topic labels cannot partition the span. FAIL.

The whole corpus carries exactly two topic tuples: 328 sets `('purple_threshold',)`
and 74 sets `('blueberry_size',)`. Within the 74 there is **one** label and **one**
channel (`#south_america-faragrotech`). `bloomfield-0330` is pure ops noise
("turn it off and back on and connect that blue cable") and is still labeled
`blueberry_size`.

The label is **thread-level, not content-level**. It cannot separate routine from
on-topic traffic inside the evaluation span.

**Consequence.** The Stage-0-labels fallback is mandatory, not optional, and it now
carries *two* jobs rather than one: the Stage 5a partition **and** the 5f
precision/recall definition. Stage 0 moves onto the critical path for evaluation,
not just for gate recall.

## P-2 — categories separate strongly, but not along the hand-specified cut.

v4 compliance is total: every `cov4*` run parses at 1.00 reserved-word-leading.
Mass-weighted distributions over `COLLEAGUE / CUSTOMER / MEASUREMENT / DOCUMENT`,
purple thread, three seeds:

| pair | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| Wolf vs Rovani | 0.65 | 0.76 | 0.72 | **0.71** |
| Lyubovsky vs Rovani | 0.63 | 0.75 | 0.71 | **0.70** |
| Wolf vs Lyubovsky | 0.11 | 0.33 | 0.37 | 0.27 |

Rovani is `CUSTOMER`-dominant (0.65–0.77); Wolf and Lyubovsky are
`COLLEAGUE`/`DOCUMENT`-dominant with almost no `CUSTOMER`. The separation is large
and seed-stable, so the axis carries real signal.

**But the hand-specified coalition is the wrong cut.** `eval_motive_sep.py` groups
Wolf (management) against Lyubovsky+Rovani (engineering). Merging Lyubovsky with
Rovani averages two near-opposite distributions:

| | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| Wolf vs merged(Lyu+Rov) | 0.33 | 0.54 | 0.41 | **0.43** |

0.43 against 0.70 for the natural Rovani-vs-rest cut. This independently reproduces
the anomaly `eval_motive_sep.py` already recorded — *"management vs engineering
scored 0.33 while two ENGINEERS scored 0.42"* — and explains it: Rovani is the
outlier, and the hand coalition splits the real boundary.

**Unresolved for blueberry.** **No size-thread run carries v4 standards at all.**
The pre-registered blind spot — both referents landing in one bucket — cannot be
checked from existing logs. One cheap v4 size run is required before Stage 3 is
built.

## P-3 — stem is unusable as an identity key. FAIL.

`restatement.stem` is the leading verb + object head. Over 3,272 distinct
normalized anchors from every run on disk:

```
distinct normalized anchors : 3272
distinct stems              : 1758
stems holding >1 anchor     :  540
anchors that would MERGE    : 2054  (63%)
```

Worst collisions merge genuinely different commitments: `[ensure data]` absorbs 29
anchors including "Ensure data loads completely" and "Ensure data quality";
`[to maintain]` absorbs 31 including "maintain a facade of impartiality" and
"maintain his public image".

**Consequence.** Memory keys on `normalize_commitment` output exactly, as the plan's
fallback specified. Additionally — and beyond the fallback — 63% is too loose even
for **consolidation matching**, which would over-promote. Consolidation must require
stem match **plus** a similarity check, reusing the existing `MERGE_ABSOLUTE_FLOOR
= 0.40` Jaccard that the codebase already calibrated for "is this pair actually
alike".

## P-4 — no roles. Confirmed.

`bloomfield_speakers.json` is 32 entries of token -> full name
(`'Wolf' -> 'Hayden Wolf'`). "Channel-role" coalitions are not derivable; volume +
reply adjacency stands.

## P-5 — the >=0.6 drop-rate target is unreachable. FAIL.

Gate ceiling is `is_low_content` alone, before any of the five admit signals
(which only re-admit more):

| corpus | raw | after stripping Slack markup | minus bare-numeral | minus numeral+unit |
|---|---|---|---|---|
| blueberry_size (761 turns) | 0.32 | **0.35** | 0.33 | 0.35 |
| purple_threshold (1,429 turns) | 0.13 | **0.17** | 0.16 | 0.17 |

Three findings:

1. **Strip Slack markup (`<@U…>`, `<#C…|…>`, `<http…>`) before `is_low_content`.**
   Free gain (0.32 -> 0.35, 0.13 -> 0.17): user IDs and URLs inflate word counts and
   keep pure acknowledgements out of the gate.
2. **Use numeral+unit, never bare numeral.** Bare numeral re-admits 72 of 246
   low-content turns, almost all Slack user IDs and timestamps — "Thanks
   `<@U0459CR7Y59>`", "1 hour". Numeral+unit re-admits **zero**, so it costs no
   drop rate and stays available for "the 18s look fine".
3. **The >=0.6 target was set without measurement and cannot be met.** Ceiling is
   0.35 and 0.17.

**Consequence.** The absolute sub-target is replaced by a cost criterion (see the
plan's amended Stage 2). The *differential* abandon criterion — routine and on-topic
drop rates must separate — survives unchanged and is still the real test, but it
remains untestable until Stage 0 labels exist.

## P-6 — counts and budget.

```
bloomfield total      402 sets / 2,190 turns
  blueberry_size       74 sets /   761 turns   (1 channel)
  purple_threshold    328 sets / 1,429 turns
```

Speaker volume on blueberry_size: Letelier 275, McLafferty 118, Rodriguez 91,
Theofiledes 68, Kelley 35, Cisneros 30 … **Lyubovsky 9**, Rovani absent from the
top 12.

> Note for Stage 3: the hand-specified "Engineers" coalition
> (Lyubovsky, Rovani, Deskins, McLafferty) is **McLafferty-dominated** on this
> thread — Lyubovsky contributes 9 turns and Rovani none of the top volume. Taken
> with P-2, the hand coalition is doubly questionable here.

Scored steps after gating: Field 214 + Engineers 116 = **330 across both boards**.

| configuration | calls / run (1 seed, 1 arm) | 5 seeds x 5 arms |
|---|---|---|
| full (5+3) | ~2,970 | **~74,250** |
| flat N=5 | ~1,980 | ~49,500 |

Purple calibration, one 2-board run: ~3,447 calls. Cheaper than feared — purple is
1.9x blueberry in turns, not the 3–4x assumed — and the offline sweep means only
**one** purple run is needed for four of five episode thresholds.

---

## P-7 (added) — the headline is well-powered, and the target is blunter than assumed

Argmax churn over the 52 existing bloomfield runs with >=20 steps:

```
median 55.6 per 100 steps   mean 52.9   range 4.2 - 73.9
runs with ZERO churn: 0/52
```

At the median that is ~83 churn events on 150 routine scored steps. The
"counts will be 0-3, unresolvable at this n" concern is **refuted**.

It also reframes the goal. A board whose leader flips on **56% of steps** —
on spans hand-picked to be favourable — has no long-term state at all. The
deliverable is not a subtle differential but a reduction in gross instability,
so the absolute standing churn rate is reported as the headline number beside
the flat-vs-full delta.

## P-8 (added) — the trigger signal, measured

Only two runs on disk carry both the null scorer and v4 standards
(`rrarm_s0`, `rrctrl_s0`; 31 runs have v4 standards, 32 have the null, 2 have
both). On those, reconstructing margins from `allocation_raw` keyed by
`likelihood_rank`:

| | rrarm_s0 | rrctrl_s0 |
|---|---|---|
| scored steps | 79 | 79 |
| lean **undefined** (all margins <= 0) | 27 (34%) | 30 (38%) |
| distinct lean vectors | 25 / 52 | 35 / 49 |
| margin-lean mean per-step movement | 0.143 | 0.171 |
| mass-lean mean per-step movement | 0.096 | 0.150 |
| ratio margin/mass | **1.5x** | **1.1x** |
| argmax category | 44/52 one bucket | 26/49 one bucket |

The premise of reading `margins` rather than mass was that margins vary on a
stable board while mass does not. They vary **1.1-1.5x** as much, are heavily
quantized, and their argmax sits in one bucket -- so the persistence condition
stays nearly free, the exact failure the switch existed to cure.

Caveats, stated: these runs' standards are mostly prose rather than v4 (65 of
508 nonempty standards are reserved-word-leading), so the classifier is the
`eval_motive_sep` regex with its known ~39%-unbinned rate; and they are
single-target runs at N=8, not a 5-slot standing stratum. Indicative, not
conclusive -- but it points the wrong way, and the instrumented run is what
settles it.

**Consequence.** The 34-38% undefined steps are promoted from discarded
"holes" to the **primary trigger signal**; the v4 categories are demoted to a
reported dyad trait, which P-2 shows they are good at. `margins` and
`lean_undefined` are now instrumented so that no future run is unreplayable
for the same reason.

---

## P-9 (added) — the instrumented run. Both the trigger AND its fallback fail off the cherry-picked span.

One run per target, `cov4` config plus `--baseline-scorer`, seed 0, on **17
non-cherry-picked purple sets** (2024-02-19 to 2024-04-26; every set where >=2
of Wolf/Lyubovsky/Rovani speak, with all six hand-picked color sets excluded).
877 LLM calls total, against a 3,447 estimate.

### The episode trigger does not separate

Per-party rate of the draft-3 signal (`lean_undefined and not off_topic`):

| target | steps | undefined | off_topic | SIGNAL |
|---|---|---|---|---|
| Wolf | 31 | 4 (13%) | 4 | **3 (10%)** |
| Lyubovsky | 42 | 11 (26%) | 7 | **5 (12%)** |
| Rovani | 21 | 5 (24%) | 2 | **4 (19%)** |

| | pair | joint | asym |
|---|---|---|---|
| TEST | Rovani vs Wolf | 0.097 | 0.094 |
| CONTROL | Wolf vs Lyubovsky | **0.097** | 0.022 |

`joint` is **identical** on the test and control dyads. `asym` looks like a 4x
separation but is just `|rate_A - rate_B|` at run level -- a difference of two
per-person base rates, with no co-occurrence in time. That is a trait-shaped
quantity wearing an episode's clothes, which is the same confusion drafts 1
and 2 died of.

Windowed (w=8), TEST is above CONTROL on both rules (joint 0.125 vs 0.087;
asym 0.115 vs 0.076) but both peak at exactly 0.250, on **12 signal events in
total**. Noise-dominated; not a basis for a threshold.

The signal itself is not meaningless -- Wolf's three flagged turns are
"This is also likely due to the size of the plants" and "can someone share an
update on where bunch rot is?", both of him opening something the board was
not tracking, and `off_topic` correctly absorbed "can you send a calendar
invite". It fires on the right kind of turn. It does not distinguish a
misaligned dyad from an aligned one.

### The v4 trait does not replicate. This is the larger finding.

P-2 measured Rovani at `CUSTOMER` 0.65-0.77, seed-stable across s0/s1/s2, and
a 0.70 separation from Wolf. Same prompt, same seed, different span:

| target | span | COLL | CUST | MEAS | DOC | n |
|---|---|---|---|---|---|---|
| Wolf | cherry 6-set | 0.25 | 0.00 | 0.42 | 0.33 | 146 |
| Wolf | non-cherry 17-set | 0.22 | 0.02 | 0.41 | 0.35 | 221 |
| Lyubovsky | cherry 6-set | 0.34 | 0.02 | 0.37 | 0.27 | 143 |
| Lyubovsky | non-cherry 17-set | 0.26 | 0.00 | 0.66 | 0.08 | 278 |
| Rovani | cherry 6-set | 0.02 | **0.65** | 0.14 | 0.19 | 108 |
| Rovani | non-cherry 17-set | 0.08 | **0.00** | 0.77 | 0.15 | 160 |

```
WITHIN-PERSON TV across spans    Wolf 0.04   Lyubovsky 0.29   Rovani 0.69

BETWEEN-PERSON TV
  cherry 6-set        Rovani/Wolf 0.65   Lyubovsky/Rovani 0.63   Wolf/Lyubovsky 0.11
  non-cherry 17-set   Rovani/Wolf 0.36   Lyubovsky/Rovani 0.18   Wolf/Lyubovsky 0.29
```

**Within-person variation (0.69) exceeds between-person variation on the
honest span (0.36 max).** The same person's standard distribution moves more
across two conversations than two different people's distributions differ
within one. Rovani's `CUSTOMER` mass goes 0.65 -> 0.00.

The ordering inverts too: Lyubovsky and Rovani are near-opposite on the
cherry-picked span (0.63) and the **most similar pair** on the honest one
(0.18).

**The measurement is stable across SEEDS and unstable across SPANS.** Seed
spread was the floor every existing control checks against; span variation was
never checked, and it is roughly 10x larger. A seed-stable number was taken as
a stable property of a person when it is a property of the conversation.

Both spans are labeled `purple_threshold`, which is P-1 again: the label is
thread-level and does not describe content. The 6-set span argues about
defining "2-Purple" (where "ask the customer" is the live question); the
17-set slice is largely count reconciliation (where measurement is). The
STANDARD field tracks that shift faithfully -- it is behaving correctly and
simply is not a dispositional quantity.

### Consequence

The episode trigger has no demonstrated signal, and the dyad trait that was
named as its surviving fallback does not survive either. Stage 3 is not
supported by evidence and should not be built on this basis.

Untouched by any of this: **P-7**. Argmax churn median 55.6 per 100 steps over
52 runs, zero runs at zero. The standing board is grossly unstable, that
finding is independent of the standard axis and of the episode trigger, and it
is what Stage 1 addresses.

---

## P-10 — the sensitivity test. Stage 3 CLOSED.

Pre-registered before the run. Span: all 42 purple sets where >=2 of
Wolf/Lyubovsky/Rovani speak, **including** the six color sets this time. POS =
those six (the hand-picked "color case"); NEG = the other 36. 2,001 LLM calls.

DECISION RULE, fixed in advance: close Stage 3 if POS is not elevated over NEG
by more than the between-party spread of the NEG rate.

| target | POS n | POS rate | NEG n | NEG rate | lift |
|---|---|---|---|---|---|
| Wolf | 26 | 0.31 | 41 | 0.15 | **+0.16** |
| Lyubovsky | 20 | 0.05 | 69 | 0.23 | **-0.18** |
| Rovani | 12 | 0.08 | 40 | 0.15 | **-0.07** |

between-party spread of the NEG rate: 0.09

**Two of three lifts point the wrong way, and the largest magnitude is
negative.** Wolf's +0.16 clears the floor; Lyubovsky's -0.18 exceeds it in the
opposite direction. A detector whose sign flips by party is not detecting a
property of the region.

This is the failure `eval_motive_sep.py` already warned about -- *"measured on
two runs of identical input, Wolf and Lyubovsky SWAPPED which settlement mode
dominated ... A metric that reads one run therefore rewards noise."* An
interim read of Wolf alone showed +0.16 and looked like signal. It was the
same mistake at the level of parties rather than seeds.

Dyad separation, both combining rules, neither consistent:

| | pair | trait TV | joint | asym | windowed joint | windowed asym |
|---|---|---|---|---|---|---|
| TEST | Rovani vs Wolf | 0.22 | 0.130 | 0.070 | 0.092 | 0.155 |
| CONTROL | Wolf vs Lyubovsky | **0.31** | **0.179** | 0.021 | 0.062 | **0.167** |

CONTROL beats TEST on `joint` and on windowed `asym`. The only column where
TEST leads is run-level `asym`, which is `|rate_A - rate_B|` -- two per-person
base rates subtracted, with no co-occurrence in time.

The per-party undefined rate is 26-28% across all three, against 34-38% on
rrarm/rrctrl. Stable and near person-independent, which is itself evidence it
reflects the filter's fit rather than anything about these people.

### The trait fallback has inverted

| target | P-2 (cherry 6-set) | P-9 (17-set) | P-10 (42-set) |
|---|---|---|---|
| Rovani CUSTOMER | 0.65 | 0.00 | 0.19 |
| Wolf CUSTOMER | 0.07 | 0.02 | 0.21 |
| trait TV, Rovani vs Wolf | **0.70** | 0.36 | **0.22** |
| trait TV, Wolf vs Lyubovsky (control) | 0.11 | 0.29 | **0.31** |

On the widest, least-selected span the CONTROL dyad is more separated than the
TEST dyad. The 0.70 that made the trait look like a deliverable does not
survive contact with unselected data, and the ordering reverses.

### Caveats, stated rather than buried

* **The POS label is contaminated.** The six "color case" sets are mostly not
  about the 2-Purple dispute: 0006 opens on a scrum blog link, 0009 and 0039
  on Agrovision hand-count data. This is P-1 one level down -- even the
  hand-picked case span is not content-homogeneous. So this is a test of "can
  it see the hand-picked span", not "can it see the documented dispute".
* 11 of 219 steps could not be resolved to a set by text match.
* One seed.

The sign disagreement across parties is the damning part and does not depend
on label quality: if the signal were real, three people in the same region
should lean the same way. A cleaner POS label -- Stage 0 turn-level labels --
is what would be needed to reopen this, which is what the plan said before I
tried to route around it.

### Closed

Stage 3 is closed. Not built. The episode trigger has no demonstrated
sensitivity across three signal designs (mass lean, margin lean, undefined
rate) and its named fallback inverts on unselected data.

**Unaffected: P-7.** Argmax churn median 55.6 per 100 steps over 52 runs, zero
runs at zero. That is measured, independent of the standard axis and of
episode detection, and it is what Stages 1 and 4 address.

---

## P-11 — the transcript-side prototype fails too, on the target term itself.

`referent_split.py`: for every term both parties use >=4 times, compare each
party's bag of co-occurring words (Jensen-Shannon), scored against a
permutation null that reshuffles which party said which turn. No board, no
LLM. Built specifically because it samples a different layer from the three
designs FINDINGS §4 explains away.

### Raw version: detects writing style, not meaning

| | terms clearing p95 |
|---|---|
| TEST  Field vs Engineers | 32 / 68 (**47%**) |
| CONTROL  FieldA vs FieldB (same side) | 14 / 29 (**48%**) |

Identical rates, and `scans` is the top term in both. The permutation destroys
the speaker partition, so any systematic per-speaker style difference makes
every term significant. This is the failure mode of the three culture
detectors this repo already killed, caught immediately by the rule
`eval_motive_sep.py` states: a variant is only an improvement if it raises the
real score WITHOUT raising the control.

(First run also ranked `djrrsta` and `rdabe` top two -- Slack user-id
fragments from `<@U0DJRRSTA>`. Markup is now stripped before tokenising.)

### Normalised version: subtract the pair's own baseline divergence

| | top norm | 90th pct of norm |
|---|---|---|
| TEST | +0.093 | +0.044 |
| CONTROL | +0.069 | **+0.046** |

The control's 90th percentile is HIGHER than the test's. And the decisive row:

```
size    normalised excess -0.008   rank 39/68
sizes   normalised excess -0.025   rank 54/68
```

**The one term the entire case is about ranks in the bottom half and separates
LESS than the median term.** Commercial caliber grading for export and
green-berry sizing for harvest timing are lexically indistinguishable here --
both parties say "size" surrounded by millimetres, categories and blueberries.
The difference is in what the number is FOR, and purpose leaves no
co-occurrence footprint.

Consistent with `scene_player.py:341`, which already recorded that the topic
axis scores the sharpest line in the Wolf trace at exactly 0.0.

### Why this strengthens the Stage 3 close

The close previously rested on one family of designs, all reading the particle
population, with a contaminated positive label. It now rests on **two
independent families**, and the second does not touch the filter at all, uses
a clean permutation null, and was tested directly on the target term rather
than on a proxy region.

Four designs, two layers, one conclusion: the misalignment is a difference in
what a measurement is FOR. It is not in the board's fit, not in the standard
distribution, and not in co-occurrence vocabulary.

That is a statement about the phenomenon, not about the detectors, and it
narrows what a fifth attempt would have to do: represent purpose, and be
scored against labels that do not yet exist. Stage 0 remains the prerequisite.

---

## P-12 — cross-run memory accumulates, but nothing recurs.

Two date-split halves of Wolf's purple traffic, run as **separate process
invocations** so half 2 could only see what half 1 wrote to disk. Same person,
same corpus, same config, adjacent spans. 352 LLM calls.

### The mechanics pass

```
store: 44 commitments, 13 live / 31 retired
spans recorded: purple-h1 27, purple-h2 17
weight-like fields exposed to the caller: none
a different config sees an empty store: True
```

Accumulation across processes, config isolation, and the no-stored-weight rule
all hold.

### The purpose fails

```
h1 27 commitments, h2 17
EXACT overlap    : 0
STEM overlap     : 0
JACCARD >= 0.6   : 0
JACCARD >= 0.5   : 1
```

**Zero commitments recur across the two runs at any usable threshold.** The
single 0.50 pair is "Ensure labeling quality" / "Ensure clear labeling", which
a reader would call the same commitment and no lexical rule catches -- the
stem is `ensure labeling` against `ensure clear`, so even stem matching, which
P-3 measured as merging 63% of anchors WITHIN a run, finds nothing ACROSS two.

Consolidation requires the same commitment in >=2 non-adjacent threads. At 0
stem matches and 0 pairs clearing the 0.60 floor, the promotion path cannot
fire. The store is a log of 44 distinct strings, not a consolidating picture
of a person.

### This is P-9 one level down

P-9: the same person's v4 standard distribution moved MORE across two spans
(0.69) than two different people differed within one (0.36). P-12: the same
person's commitment TEXT does not repeat across two spans at all. Both say the
filter's output is shaped by the conversation rather than by the person -- the
thing a long-term memory would have to hold constant.

### What would have to change

Lexical matching is the wrong instrument for a generator that writes free
text. The candidates, in order of cost:

1. **Semantic matching** -- embeddings, or an LLM judge asked whether two
   commitments are the same aim. Cheap with embeddings, and the obvious fix.
   `merge_equivalent_anchors` already uses a MODEL rather than a threshold
   inside a run, for exactly this reason (tracer.py:2058-2072 records that
   "Avenge her mother" and "Forgive her mother's killer" score 0.17 lexically,
   the same as a synonym pair).
2. Constrain the commitment vocabulary at generation.
3. Accept the store as a log and drop consolidation.

Route 1 is the same fix the tracer already made internally; the store simply
never inherited it. Until then, Stage 4 persists correctly and consolidates
nothing.

---

## P-13 — churn attribution. Stratification was designed against the wrong cause.

P-7 measured argmax churn at a median 55.6 per 100 steps and named it the
problem Stages 1 and 4 address. This attributes it. No LLM calls: 6,160 step
transitions across every run on disk, using `operators_fired` and the
`pre_operator_particles` snapshot the log already carries.

```
transitions 6160   overall churn 41.1%

operators_fired        steps  churned  P(churn)    lift
(none)                  3864     1163     30.1%  -11.0%
merge                   1141      991     86.9%  +45.8%
split                    903      792     87.7%  +46.6%
expand                   302      262     86.8%  +45.7%
resample                 141       76     53.9%  +12.8%
perturb                  800      298     37.2%   -3.8%
expire                   534      198     37.1%   -4.0%
```

### The decisive number

```
Leader had already changed BEFORE any operator ran: 1775 / 5931 steps
  -> 70% of all churn is caused by REWEIGHTING ALONE
```

`pre_operator_particles` is logged before the step's operators fire. Comparing
its argmax to the previous step's shows the leader had already moved on 30% of
steps, which is 70% of every churn event in the corpus.

### What this does to the plan

**The two-timescale split addresses competition for mass between two kinds of
hypothesis.** That is not what is moving the leader. 70% of churn happens in
the likelihood reweighting, before any operator, and a standing stratum would
be reweighted every step by the same scorer and churn identically. Splitting
the population cannot fix instability that arises inside the weight update.

Note also that `perturb` and `expire` are BELOW baseline (-3.8%, -4.0%) --
the two operators most often blamed for destroying hypotheses are not the ones
moving the leader. `merge`, `split` and `expand` nearly guarantee churn when
they fire (~87%), but they fire on roughly a third of steps and account for
the minority 30%.

### The lever the measurement does support

`accumulate_weights` uses `w_t ∝ w_{t-1}^alpha · L_t^beta` with **alpha=0.85 on
every run on disk** (133 runs; the other 99 predate the field). That is a prior
half-life of **4.3 steps** -- the filter forgets half of what it believed
within four turns, so a single unusual turn can hand the lead to a different
root.

```
alpha=0.85  half-life  4.3 steps
alpha=0.90             6.6
alpha=0.95            13.5
alpha=0.98            34.3
```

A one-parameter sweep of alpha against argmax churn is the intervention this
attribution actually points at, it needs no new machinery, and no run has ever
varied it. That is a far better-founded next step than stratification, which
was designed against a cause that accounts for at most 30% of the effect.

---

## P-14 — the alpha sweep. Prediction falsified; the instability is upstream.

Pre-registered before running. alpha in {0.85, 0.90, 0.95, 0.98} x 3 seeds,
target Wolf, 17-set non-cherry purple span, everything else fixed. 12 runs,
3,395 calls.

P-13 predicted churn would fall monotonically as alpha rises, since 70% of it
is reweighting and alpha is the only term carrying belief across a step.

```
 alpha  half-life  churn/100   spread  roots  surpr  resamp  postESS
  0.85        4.3       55.6      6.7   28.0    5.3     1.0     4.78
  0.90        6.6       64.4     26.7   30.3    2.7     0.3     4.28
  0.95       13.5       64.4     10.0   31.3    4.0     1.0     5.06
  0.98       34.3       61.1     43.3   32.0    3.7     0.7     4.94
```

**Churn went UP at every alpha**, and every difference is within noise:
the baseline's own seed spread is 6.7 per 100, and alpha=0.98's is **43.3** --
larger than the entire effect being chased. The responsiveness check fired
anyway at 0.90 and 0.98 (surprise detection 2.7 and 3.7 against 5.3), so even
the non-effect carries a cost.

Per the pre-registered rule: **REJECT. No alpha is adopted.** This is the
between-run noise floor `eval_motive_sep.py` already recorded for this corpus,
met head-on.

### Why alpha cannot work, measured

If the weight update is not what moves the leader, the likelihood might be.
Over the same 12 runs:

```
mean |rank change| step-to-step, LIKELIHOOD : 1.79 places
mean |rank change| step-to-step, WEIGHT     : 1.03 places
   (population of 8; a random reshuffle gives ~2.7)

likelihood top-1 changes root: 258/356 = 72% of steps
```

**The scorer picks a different best hypothesis on 72% of steps, before any
weight update touches it.** The likelihood ranking moves 1.79 places per step
against a random-shuffle ceiling of 2.7 -- two thirds of the way to noise.

The weight update is in fact doing its job: it damps 1.79 places of likelihood
movement down to 1.03 of weight movement. Raising alpha damps harder, but it
cannot manufacture a stable leader out of an input that is reshuffling at 72%,
and pushing it further only blunts surprise detection.

### What this closes

P-7 framed churn of 55.6/100 as the problem Stages 1 and 4 address. P-13
showed operators account for at most 30% of it. P-14 shows the remaining 70%
originates in the SCORER, not in the weight update, and is not reachable by
any weight-update parameter.

So the instability is not a population-management problem at all. It is the
per-step likelihood being close to unstable, which is upstream of
stratification, of memory, and of alpha. Consistent with FINDINGS: rank-first
scoring raised Kendall tau from 0.687 to 0.926, but tau is rank CORRELATION --
an argmax can flip freely while tau stays high, and it does.

Anything aimed at "a stable long-term board" has to start there.
