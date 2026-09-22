"""Named hypothesis-GENERATION methods, as swappable prompt fragments.

Twelve structured analytic techniques, specified in prose in HYPOTHESES.md and
given code here. The claim that motivates them is HYPOTHESES.md's own:

    Twelve methods, run over every named person at every customer. Each entry
    is tagged with the method that produced it, BECAUSE METHOD PREDICTS FAILURE
    MODE.

    | Tag            | Method                   | What it does                                                          |
    |----------------|--------------------------|-----------------------------------------------------------------------|
    | anomaly        | Anomaly detection        | Start from the datum that does not fit and reason back                 |
    | role           | Role play                | Inhabit the person, then state their position in the third person      |
    | silence        | Absence analysis         | Reason from who and what is missing from the record                    |
    | devil          | Devil's advocate         | Argue against the best-supported claims, including my own              |
    | ach            | Competing hypotheses     | For one observation, enumerate rival explanations and keep survivors   |
    | assume         | Key assumptions check    | Surface what the analysis takes for granted and challenge it           |
    | premortem      | Premortem                | Assume the account is lost in six months, then explain why             |
    | presentation   | Self-presentation check  | Look for where a source is telling Musing what it wants to hear        |
    | crystal        | Crystal ball             | Ask what a perfect source would reveal that this data cannot           |
    | backcast       | Backcasting from outcomes| Fix a plausible end state, work backwards to what must be true now     |
    | signpost       | Indicators               | State what would have to be observed for a claim to be true            |
    | analogy        | Structured analogy       | Apply a pattern seen elsewhere, tested only against this org's evidence|

WHY THIS EXISTS. The filter's four generation sites all ask one question --
what does the target want? -- and DEBUG_HANDOFF.md records what that costs:
"Every hypothesis source is a READER, not a generator... This system has a weak
version of the testing half and none of the proposing half." A structured
analytic technique is precisely a device for proposing a cause that was never
said, which is the open problem: "In a Slack work thread nobody states why they
want a parameter."

WHY A SEPARATE MODULE, when STANDARD_PROMPTS lives inside tracer.py. Two
reasons. Twelve methods times three sites is an order of magnitude more prose
than the four STANDARD variants, and it would sit between is_low_content and
valid_commitment and bury both. More importantly the audit and the evals must
import the registry WITHOUT importing tracer, which constructs API clients --
audit_methods.py compares each method's PREDICTED failure mode against its
measured one, and that comparison needs the metadata, not the prompts.

THE DEFAULT PATH IS SACRED. With no method selected, method_rule() returns ""
at every site and every prompt in the repo is byte-identical to before this
module existed. FINDINGS.md's measurements are all against those prompts; a
one-character drift in the default path destroys more evidence than any
method can generate.
"""

from dataclasses import dataclass


# The three generation sites a method can speak at. A fragment is APPENDED to
# the site's existing prompt, never replaces it -- see standard_block() in
# tracer.py for why replacement would make the measurement circular.
SITES = ('seed', 'perturb', 'split', 'standard')

# Which field a method acts on.
#   anchor   -- it proposes a COMMITMENT, and its fragment lands in the
#               commitment block at the seed/perturb/split sites.
#   standard -- it proposes what would SETTLE the question, and its fragment is
#               appended to standard_rule()'s output instead. crystal and
#               signpost are about evidence that does not exist yet, which is
#               the standard field's subject and not the anchor's; routing them
#               through the anchor site would produce observable-condition
#               clauses where <=8-word aims belong.
AXES = ('anchor', 'standard')

FORM_RISKS = ('low', 'med', 'high')


@dataclass(frozen=True)
class Method:
    """One generation method: its prompt fragments, and its pre-registration.

    The metadata fields are not decoration. `needs_action` drives the mint-site
    chooser (a method that reasons from a datum that does not fit has nothing
    to work with on a stagnation step). `axis` decides which block the fragment
    lands in. `expected_failure` and `form_risk` are the PREDICTION that
    audit_methods.py scores the measured behaviour against -- which is the only
    thing that makes "method predicts failure mode" falsifiable rather than a
    slogan, and the whole reason this is a dataclass and not a dict of strings.
    """
    key: str
    name: str
    # Per-site prompt fragments. {t} is the target's name. An empty fragment
    # means "no change at this site"; a method with no fragment for a site
    # falls back to its perturb text, which is the site where the framing
    # matters most because perturbation is the only source term.
    seed: str = ""
    perturb: str = ""
    split: str = ""
    standard: str = ""
    # --- pre-registration ---
    axis: str = 'anchor'
    needs_action: bool = False
    needs_role_prior: bool = False
    expected_failure: str = ""
    form_risk: str = 'low'


# ---------------------------------------------------------------------------
# The twelve.
#
# Every anchor-axis fragment is spliced BEFORE the form block (the <=8 words /
# never a question / never an instruction rules). The form constraint must be
# the last thing the model reads, so a method that pulls against it loses the
# argument -- see the note on valid_commitment in tracer.py for why a rejected
# candidate is cheaper than a malformed one that founds a root.
# ---------------------------------------------------------------------------

_ANOMALY = Method(
    key='anomaly', name='Anomaly detection',
    axis='anchor', needs_action=True, form_risk='low',
    expected_failure="latches onto noise; a one-off oddity becomes a standing commitment",
    perturb=(
        "- METHOD -- anomaly detection. Begin from the single thing in the record that does "
        "NOT fit. The message sharper than its occasion, the question asked after it was "
        "already answered, the person brought into a thread who had no business in it, the "
        "answer that addressed something nobody asked. Name that one datum to yourself "
        "first. Then propose the commitment of {t}'s that would make it ordinary -- the aim "
        "under which that datum is the obvious thing to do.\n"),
    seed=(
        "Work from the detail that does not fit rather than from the gist. Find the part of "
        "{t}'s behaviour that the obvious reading does not account for, and let each "
        "hypothesis be an aim that would account for it. "),
)

_ROLE = Method(
    key='role', name='Role play',
    axis='anchor', needs_role_prior=True, form_risk='low',
    expected_failure=("first-person leakage, and sympathy -- inhabiting someone makes their "
                      "aim sound more coherent than the record supports"),
    perturb=(
        "- METHOD -- role play. Take {t}'s seat and reason from inside it: what they can see "
        "from there, who they answer to, what lands on them if this goes wrong, what they "
        "cannot say out loud in this room. Then step back OUT and state the answer in the "
        "third person, as {t}'s own aim. Never write \"I\" or \"my\". The reasoning is from "
        "inside; the answer is from outside.\n"),
    seed=(
        "Reason from inside {t}'s seat -- what they can see from there, who they answer to, "
        "what lands on them if this goes wrong -- then state each hypothesis in the third "
        "person as {t}'s own aim. Never write \"I\". "),
)

_SILENCE = Method(
    key='silence', name='Absence analysis',
    axis='anchor', form_risk='high',
    expected_failure="unfalsifiable -- an absence is consistent with almost any aim",
    perturb=(
        "- METHOD -- absence analysis. Reason from what is ABSENT rather than from what is "
        "present. Who is not in this thread who should be. What question nobody asked. What "
        "{t} raised once and has not mentioned since. Whose name {t} does not use. What {t} "
        "answered around rather than answering. Propose the commitment that would explain "
        "the absence.\n"
        "  FORM WARNING: state it as a positive aim -- what {t} is trying to bring about -- "
        "never as a negation (\"avoid...\", \"not...\") and never as a question. The absence "
        "is your evidence, not your answer.\n"),
    seed=(
        "Reason from what is missing: who is not here, what question nobody asked, what {t} "
        "has stopped mentioning. Each hypothesis is the aim that would explain the absence, "
        "stated as a positive aim rather than as a negation. "),
)

_DEVIL = Method(
    key='devil', name="Devil's advocate",
    axis='anchor', form_risk='low',
    expected_failure=("produces the lexical negation of the leading account, which merge "
                      "absorbs on the same step; highest CONTRADICTS rate at the coherence gate"),
    perturb=(
        "- METHOD -- devil's advocate. Take the account currently best supported -- including "
        "one you would otherwise defend -- and argue against it. Assume the leading reading "
        "is WRONG. Propose the aim the same record supports under that assumption.\n"
        "  It must be genuinely incompatible with the leading account, not its wording "
        "reversed: \"Earn Katara's trust\" -> \"Avoid earning Katara's trust\" is the same "
        "hypothesis stated backwards, not a rival one.\n"),
    seed=(
        "For each hypothesis, take the reading that seems most obvious and argue against it. "
        "Assume the obvious reading is wrong and state the aim the same record supports "
        "instead -- a genuinely different aim, not the obvious one negated. "),
)

_ACH = Method(
    key='ach', name='Competing hypotheses',
    axis='anchor', needs_action=True, form_risk='low',
    expected_failure=("nearest to what the comparative scorer already does, so it may "
                      "reproduce the filter's own loop and add nothing"),
    perturb=(
        "- METHOD -- competing hypotheses. Fix on the single observation you have been given. "
        "Enumerate the rival explanations for it to yourself BEFORE choosing any, including "
        "the dull ones. Discard every explanation the record already rules out. Propose only "
        "the survivors. Do not rank them -- that is not your job here. A survivor that is "
        "consistent with the record but unflattering stays in.\n"),
    seed=(
        "For the action above, enumerate the rival explanations to yourself before choosing, "
        "discard the ones the record already rules out, and give the survivors as your "
        "hypotheses -- including the unflattering ones. "),
)

_ASSUME = Method(
    key='assume', name='Key assumptions check',
    axis='anchor', form_risk='low',
    expected_failure="attacks assumptions in the prompt rather than assumptions in the record",
    perturb=(
        "- METHOD -- key assumptions check. Name to yourself what the accounts already in "
        "play take for granted about {t}: about what they care about, who they answer to, "
        "what they would find acceptable, what they already know. Pick the assumption that "
        "is most load-bearing and least evidenced. Then propose the commitment {t} holds if "
        "that assumption is FALSE.\n"),
    seed=(
        "Before answering, name to yourself what the obvious reading takes for granted about "
        "{t} -- what they care about, who they answer to, what they already know. Let each "
        "hypothesis be the aim {t} holds if one of those assumptions is false. "),
)

_PREMORTEM = Method(
    key='premortem', name='Premortem',
    axis='anchor', form_risk='low',
    expected_failure="imports a generic organisational failure story rather than reading this record",
    perturb=(
        "- METHOD -- premortem. Assume that six months from now this has gone badly for {t}: "
        "the relationship is broken, the work is abandoned, the trust is gone. Do not ask "
        "WHETHER that will happen -- assume it has. Explain how, using only what is in this "
        "record. Then work back to the aim {t} holds NOW that would have led there.\n"),
    seed=(
        "Assume that six months from now this has gone badly for {t}. Explain how, using only "
        "what is in this record, and work back to the aim {t} holds now that would have led "
        "there. "),
)

_PRESENTATION = Method(
    key='presentation', name='Self-presentation check',
    axis='anchor', needs_role_prior=True, form_risk='med',
    expected_failure=("trips the instruction-verb validator -- the natural phrasing starts "
                      "\"tell...\", which is a move against another character, not an aim"),
    perturb=(
        "- METHOD -- self-presentation check. Look for where {t} is saying what their audience "
        "wants to hear rather than what they think. Where the register shifts when a "
        "particular person is present. Where agreement arrives too fast. Where a concern is "
        "raised and softened in the same message. Where {t} reports a result rather than a "
        "problem. Propose the aim that the performance serves.\n"
        "  FORM WARNING: state it as {t}'s own aim, never as an instruction to anyone. "
        "\"Tell X what they want to hear\" is the wrong form; \"Keep the relationship "
        "intact\" is the right one.\n"),
    seed=(
        "Look for where {t} is saying what their audience wants to hear rather than what they "
        "think -- register shifts, agreement arriving too fast, a concern softened in the "
        "same message that raises it -- and let each hypothesis be the aim that performance "
        "serves, stated as {t}'s own aim rather than as an instruction. "),
)

_CRYSTAL = Method(
    key='crystal', name='Crystal ball',
    axis='standard', form_risk='low',
    expected_failure="proposes evidence so ideal it is unbounded by the corpus",
    standard=(
        "  METHOD -- crystal ball. Suppose a perfect source existed: someone who could answer "
        "any question about {t} honestly and completely. Say what that source would have to "
        "report for {t} to stop pressing. Name the source concretely EVEN IF this record "
        "could never contain it -- the private conversation, the number nobody has run, the "
        "thing {t}'s manager thinks but has not said. Do not retreat to evidence that happens "
        "to be available.\n"),
)

_BACKCAST = Method(
    key='backcast', name='Backcasting from outcomes',
    axis='anchor', form_risk='med',
    expected_failure=("produces an end state where a commitment belongs -- exactly the "
                      "\"commitment restated as an outcome\" the STANDARD prompts already forbid"),
    perturb=(
        "- METHOD -- backcasting. Fix a plausible end state: a concrete situation six months "
        "out that this record could credibly arrive at. Work backwards -- what would have to "
        "be true of {t} NOW for that end state to follow?\n"
        "  FORM WARNING: the end state is your REASONING, not your answer. The COMMITMENT is "
        "the aim {t} holds that pursues it. \"The grower accepts the delivered figures\" is "
        "an outcome; \"Keep the grower's trust\" is the commitment that pursues it.\n"),
    seed=(
        "Fix a plausible end state six months out that this record could credibly arrive at, "
        "and work backwards to what must be true of {t} now. Give the AIM that pursues the "
        "end state, not the end state restated. "),
)

_SIGNPOST = Method(
    key='signpost', name='Indicators',
    axis='standard', form_risk='low',
    expected_failure="names indicators so generic they would appear under any account",
    standard=(
        "  METHOD -- indicators. State what would have to be OBSERVED for this account to be "
        "true: something that could actually appear in a record of this kind -- a message "
        "someone would send, a meeting that would get scheduled, a number that would move, a "
        "person who would go quiet. It must be something that would NOT appear if the account "
        "were false. An indicator consistent with every account on the table is not an "
        "indicator.\n"),
)

_ANALOGY = Method(
    key='analogy', name='Structured analogy',
    axis='anchor', form_risk='med',
    expected_failure=("imports the other case's vocabulary, which inverts the boundness "
                      "metric and breaks the one-org scoping rule"),
    perturb=(
        "- METHOD -- structured analogy. Apply a pattern you have seen hold in organisations "
        "of this kind: how a new hire behaves before their first review, what someone does "
        "when their own manager has changed the requirement on them, how a person defends a "
        "decision they no longer believe in, what happens to a request that has been refused "
        "twice. Use the pattern as a GENERATOR only.\n"
        "  The clause you produce must be testable against THIS record alone. Do not import "
        "the other case's names, nouns, numbers or jargon. If your clause only makes sense "
        "given the analogy, discard it.\n"),
    seed=(
        "Generate from patterns that hold in organisations of this kind -- how someone behaves "
        "before a first review, when a requirement has changed under them, when defending a "
        "decision they no longer believe in. Use the pattern only as a generator: each "
        "hypothesis must be testable against this record alone, with none of the other case's "
        "names or jargon. "),
)


METHODS = {m.key: m for m in (
    _ANOMALY, _ROLE, _SILENCE, _DEVIL, _ACH, _ASSUME,
    _PREMORTEM, _PRESENTATION, _CRYSTAL, _BACKCAST, _SIGNPOST, _ANALOGY,
)}


# Appended to every seed-site fragment. The seed site's output contract is
# just "a numbered list of hypotheses" -- far looser than the COMMITMENT |
# BELIEF | STANDARD triple the perturb and split sites demand -- and most of
# these methods ask the model to reason privately before answering. Measured
# on the first live mixed run: devil wrote its scaffolding into the answer
# ("**Hypothesis:** ... **Argument against obvious reading:** ..."), which then
# propagates as the particle's text and is what the scorer reads. The guard is
# uniform and applied here rather than written into each fragment, so a
# thirteenth method cannot forget it.
SEED_OUTPUT_GUARD = (
    "Do that reasoning silently: each numbered item must be the hypothesis "
    "itself, one or two plain sentences about what {t} wants and believes, "
    "with no headings, labels or argument shown. ")


def method_rule(key, site, target):
    """The fragment `key` contributes at `site`, or "" for the default path.

    Mirrors standard_rule() in tracer.py deliberately: same tolerant default,
    so an unrecognised key degrades to today's prompt rather than raising deep
    inside a run. run_musing.py validates the key at PARSE time, which is where
    a typo should be caught.

    A method with no fragment for a site falls back to its perturb text. The
    fallback is a rule rather than a convention so it lives in exactly one
    place: perturbation is the only source term, so that is the phrasing that
    has had the most attention.
    """
    m = METHODS.get(key or "")
    if m is None:
        return ""
    if site == 'standard':
        # The standard axis is opt-in per method: an anchor-axis method must
        # not leak text into the settling-condition block, or the STANDARD
        # register shifts differentially by method and the mode lexicon that
        # bins it is measuring itself.
        frag = m.standard if m.axis == 'standard' else ""
    else:
        frag = getattr(m, site, "") or m.perturb or ""
        if frag and site == 'seed':
            frag = frag + SEED_OUTPUT_GUARD
    return frag.format(t=target) if frag else ""


def contributing_methods(method_list, site):
    """Those of `method_list` that actually contribute a frame at `site`.

    A LABEL MAY ONLY BE APPLIED WHERE THE FRAME WAS APPLIED. A standard-axis
    method has no seed, perturb or split fragment: it speaks through the
    settling condition, not the commitment. Letting it take a seeding slot or
    a mint event anyway generates those particles with the DEFAULT prompt and
    then stamps them with a method that had no part in writing them.

    That is not a cosmetic problem. Measured on the first long run, before
    this filter existed: signpost was chosen for two mint events and for the
    revive that brought back "Forgive Yon Rha" -- the one commitment this
    corpus was never able to propose -- so the audit would have credited a
    method with the best result in the run on the strength of an empty string.

    Returns [None] when nothing qualifies, which is the default path.
    """
    keys = [k for k in (method_list or []) if method_rule(k, site, 'x')]
    return keys or [None]


def seeding_methods(method_list):
    """Back-compat alias; seeding is just the seed site."""
    return contributing_methods(method_list, 'seed')


def parse_methods(raw):
    """'anomaly,silence' -> ['anomaly', 'silence']. Raises on an unknown key.

    Called from run_musing.py at argument-parse time. An unknown method must
    fail loudly here: method_rule() is tolerant by design, so a typo would
    otherwise produce an empty fragment and a run that looks like it applied a
    method for 40 steps without ever having done so.
    """
    if not raw:
        return None
    keys = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not keys:
        return None
    bad = [k for k in keys if k not in METHODS]
    if bad:
        raise ValueError(
            f"unknown method(s) {bad}; valid keys: {', '.join(sorted(METHODS))}")
    # dedupe, order-preserving: round-robin over a list with a repeat would
    # silently weight that method double.
    return list(dict.fromkeys(keys))


# Registry defects fail at import, not 40 steps into a run.
for _k, _m in METHODS.items():
    assert _k == _m.key, f"{_k} keyed as {_m.key}"
    assert _m.axis in AXES, f"{_k}: bad axis {_m.axis!r}"
    assert _m.form_risk in FORM_RISKS, f"{_k}: bad form_risk {_m.form_risk!r}"
    assert _m.expected_failure, f"{_k}: no expected_failure -- the audit has nothing to score against"
    if _m.axis == 'standard':
        assert _m.standard, f"{_k}: standard-axis method with no standard fragment"
        assert not (_m.seed or _m.perturb or _m.split), \
            f"{_k}: standard-axis method must not write to the commitment sites"
    else:
        assert _m.seed or _m.perturb or _m.split, f"{_k}: no fragment at any commitment site"
        assert not _m.standard, f"{_k}: anchor-axis method must not write to the standard block"
assert len(METHODS) == 12, f"expected the twelve of HYPOTHESES.md, found {len(METHODS)}"
del _k, _m
