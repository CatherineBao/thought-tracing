"""External baseline character profiles, as a seed-time prior.

WHAT THIS REPRESENTS. Every hypothesis source in this filter is a READER: it
sees one transcript and reports what the transcript says. That works when the
motive is spoken and fails when it is not -- on a Slack work thread nobody
states why they want a parameter, so the honest summary of what someone's
messages DO is "define numerical thresholds", which is a description of the
message and not a reason for sending it.

A baseline profile is the other half: the standing record an external system
already holds about a person, accumulated over months of watching them work
rather than read off the thread in front of it. Seat, stake, pressure, how they
carry themselves, where they sit in the influence structure, who they answer to.

SEED ONLY, ON PURPOSE. The profile founds the first population and is used
nowhere else. That is what it represents -- the understanding you walk into the
room already holding -- and it is also what makes the measurement clean: the
profile's entire effect is the founding set and whatever survives from it. The
mint site keeps using the transcript-derived role prior under --infer-motive,
so the two priors stay separable and the arms stay comparable.

WHY IT MIRRORS infer_role_prior()'s OUTPUT SHAPE. That function emits
SEAT/STAKE/PRESSURE one-clause fields, and the seeding prompt's existing
sentence -- "Let the hypotheses follow from that SEAT and STAKE" -- is written
against them. Rendering an external profile into the same register means the
prompt around it does not change, so the arm difference is the CONTENT of the
prior and not the sentence that introduces it.

WHY QUOTES ARE CARRIED AND NOT RENDERED. A real external payload contains
verbatim phrases ("in your words"). They are kept in the data so the file stays
faithful to what such a system actually hands over, and dropped at render time
because quotes are transcript SURFACE -- restatement.py's echo and bound metrics
exist precisely to penalise commitments that lift transcript wording, so feeding
quotes back would inflate the thing being measured by construction. Flipping
include_quotes=True is the experiment; it is deliberately not the default.

WHY read_confidence IS NOT A WEIGHT. The filter's weights are normalised
posteriors over a live population. An external confidence is a statement about
how well some other system thinks it knows this person, on nobody's normalising
scale. Routing it into the weights would silently rescale the whole filter, so
it is carried and displayed and otherwise inert.

THE DEFAULT PATH IS SACRED. With --character-profile off, render_profile is
never called and every prompt in the repo is byte-identical to before this
module existed. Same rule methods.py states, for the same reason: every measured
finding in musing_out is against those prompts.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "musing")

# Keys that are metadata about the file rather than a person. Anything starting
# with "_" is skipped by every reader here, so a provenance note can live in the
# same JSON object as the records without a parallel file to keep in step.
_META_PREFIX = "_"


def load_profiles(corpus: str, data_dir: str = None):
    """Profiles for a corpus, or None if that corpus has none.

    Tolerant by design: eight of the nine corpora have no profiles file and must
    keep loading exactly as they do now. run_musing.load_corpus's own _read()
    raises on a missing file, which is right for dialogue/gaps/speakers -- a
    corpus without those is broken -- and wrong for this one.
    """
    path = os.path.join(data_dir or DATA_DIR, f"{corpus}_profiles.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def people(profiles) -> dict:
    """The person records, with the file-level metadata keys dropped."""
    if not profiles:
        return {}
    return {k: v for k, v in profiles.items()
            if not k.startswith(_META_PREFIX) and isinstance(v, dict)}


def _clause(value) -> str:
    """One clause from either a string or a list of them."""
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


ROSTER_LIMIT = 6


def render_profile(profiles, target: str, roster=(), include_quotes: bool = False,
                   roster_limit: int = ROSTER_LIMIT, exclude=()) -> str:
    """The prior block for `target`, in infer_role_prior()'s register.

    `roster` is the other speakers present in the traced context, most central
    first -- the caller orders it, because only the caller knows who actually
    holds the floor in the sets being traced. The relational layer is here
    because who answers to whom, and who is relaying someone else's
    requirements, is the part a single thread most underdetermines and the part
    an external system is genuinely in a position to know.

    Each person appears ONCE, with the target's standing relation to them when
    there is one and their seat otherwise. Listing someone twice -- once under
    relations, once under a roster -- spends prompt on repetition and invites
    the proposer to treat one person as two.

    Capped at `roster_limit`. Uncapped, a busy channel renders a roster several
    times the length of the target's own profile, and the prior stops being a
    prior and becomes the bulk of what the seeder reads.

    Returns "" when the target has no profile, which is the signal to the caller
    that this run has no prior to inject. An unknown target is not an error:
    selection can land on a speaker nobody wrote a profile for, and that run
    should proceed as a no-profile run rather than crash a sweep at hour three.
    """
    recs = people(profiles)
    me = recs.get(target)
    if not me:
        return ""

    lines = []
    for label, key in (("SEAT", "seat"), ("STAKE", "stake"), ("PRESSURE", "pressure"),
                       ("STANCE", "stance"), ("STANDING", "standing")):
        text = _clause(me.get(key))
        if text:
            lines.append(f"{label}: {text}")

    # Only people actually in this context. A standing relationship to someone
    # who never appears is not evidence about this conversation, and naming them
    # invites the proposer to explain behaviour by a party who is not there.
    # `exclude` exists for the SCRAMBLE control, where the record handed over
    # belongs to someone other than the person being traced. Without it the
    # roster names the traced person inside a record written from a third
    # party's seat -- "Wolf -- Lyubovsky defers to them on priority" while
    # tracing Wolf -- which is self-referential in a way a genuinely wrong
    # record would not be, and tips the model off that something is off.
    skip = set(exclude) | {target}
    present = [r for r in roster if r not in skip and r in recs][:roster_limit]
    rel = me.get("relations") or {}
    others = []
    for r in present:
        # A relation is written as a verb phrase with the target as subject, so
        # it is rendered with the target's name in front of it. Without that the
        # line reads as a fact about the OTHER person ("Lyubovsky -- relies on
        # them for technical judgement") and reverses who depends on whom.
        relation = _clause(rel.get(r))
        clause = f"{target} {relation}" if relation else _clause(recs[r].get("seat"))
        if clause:
            others.append(f"- {r} -- {clause}")
    if others:
        lines.append("OTHERS PRESENT:\n" + "\n".join(others))

    # Off by default -- see the module docstring. Rendered last so that when it
    # IS on, everything above it is unchanged and the two renderings differ by
    # exactly this block.
    if include_quotes:
        qs = [q for q in (me.get("quotes") or []) if str(q).strip()]
        if qs:
            lines.append("IN THEIR WORDS: " + "; ".join(f'"{q}"' for q in qs))

    return "\n".join(lines)


def profile_meta(profiles, target: str, roster=(), traced: str = None) -> dict:
    """What the run log records about the prior it was given.

    Self-describing runs are how the evaluators partition arms. Recording the
    roster actually used -- not the roster requested -- means a run whose
    profile file was missing three people is visibly different from one whose
    file was complete, instead of both reading as "character_profile: true".
    """
    recs = people(profiles)
    me = recs.get(target) or {}
    skip = {target} | ({traced} if traced else set())
    present = [r for r in roster if r not in skip and r in recs]
    return {
        # WHOSE record was handed over. Equal to the traced agent on a normal
        # run and deliberately different under the scramble control, so an arm
        # cannot be mistaken for the other after the fact.
        "profile_target": target if me else None,
        "profile_scrambled": bool(traced and traced != target),
        "profile_roster": present,
        "profile_read_confidence": me.get("read_confidence"),
        "profile_source": me.get("source"),
    }
