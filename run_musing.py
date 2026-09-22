"""Driver for the data/musing narrative corpora.

These corpora are already in FANToM chat format -- `full_context` is
`Name: utterance` lines joined by newlines -- so the tracer consumes them with
--input-is-chat and no conversion.

Roles (see the plan's corpus section):
  discrimination gate : odyssey (disguised_as turns), or oppenheimer
  trajectory corpus   : oppenheimer, or atla filtered to target_is_prominent

INTERNAL DATA. These files carry real Slack messages from named employees and
copyrighted transcript material. Nothing here writes them anywhere but the local
output directory and the configured LLM backend.
"""
import argparse
import json
import random
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

from methods import METHODS, parse_methods
from profiles import load_profiles, render_profile, profile_meta

# A run writing far fewer steps than its trajectory has is a FAILURE, not a
# short run. A flat floor of 5 let a crash at step 7 of a 34-step context pass
# as success. Require most of the trajectory to have been traced.
MIN_STEPS = 5
MIN_FRACTION = 0.6

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "musing")


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

def load_corpus(corpus: str) -> Dict:
    def _read(suffix):
        path = os.path.join(DATA_DIR, f"{corpus}_{suffix}.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    sets = _read("dialogue")
    gaps = _read("gaps")
    if isinstance(gaps, dict):
        gaps = list(gaps.values())[0]
    return {
        "corpus": corpus,
        "sets": sets,
        "by_id": {s["set_id"]: s for s in sets},
        "gaps": gaps,
        "speakers": _read("speakers"),
        # Optional, and read through profiles.load_profiles rather than _read:
        # eight of the nine corpora have no profiles file, and _read raises on a
        # missing one. That is right for dialogue/gaps/speakers -- a corpus
        # without those is broken -- and wrong for this.
        "profiles": load_profiles(corpus),
    }


def stitch_gold(corpus: Dict, gap: Dict) -> Optional[Dict]:
    """[last_present] + [missed...] + [rejoins_at].

    The target demonstrably lacks information the context contains, so belief and
    reality can differ -- which is the whole point of a gold context.
    """
    ids = [gap["last_present"]] + list(gap["missed_set_ids"]) + [gap["rejoins_at"]]
    parts, used = [], []
    for sid in ids:
        s = corpus["by_id"].get(sid)
        if s is None:
            continue
        parts.append(s["full_context"])
        used.append(sid)
    if not parts:
        return None
    return {
        "context_id": f"{corpus['corpus']}:{gap['speaker']}:{gap['last_present']}->{gap['rejoins_at']}",
        "role": "gold",
        "target_agent": gap["speaker"],
        "full_context": "\n".join(parts),
        "set_ids": used,
        "missed_set_ids": list(gap["missed_set_ids"]),
        "gold_rank": gap.get("gold_rank"),
        "missed_turns": gap.get("missed_turns"),
        "stitched_chars": gap.get("stitched_chars"),
        "prominent": gap.get("target_is_prominent"),
    }


def silver_context(corpus: Dict, s: Dict, target_agent: str) -> Dict:
    """Control: everyone in the set heard everything, so belief should not diverge."""
    return {
        "context_id": f"{corpus['corpus']}:{target_agent}:{s['set_id']}",
        "role": "silver",
        "target_agent": target_agent,
        "full_context": s["full_context"],
        "set_ids": [s["set_id"]],
        "missed_set_ids": [],
        "n_turns": s["n_turns"],
        "stitched_chars": len(s["full_context"]),
    }


def select_gold(corpus: Dict, *, prominent_only=True, min_turns=8, max_turns=None,
                min_chars=8000, max_chars=30000, limit=None) -> List[Dict]:
    """Band-limited gold selection.

    Deliberately NOT `sort by gold_rank and take the head`: in atla the top ranks
    are past Avatars whose 198-set gap is a bookkeeping artifact of appearing once
    early and once late, not a belief trajectory.
    """
    out = []
    for gap in corpus["gaps"]:
        if prominent_only and not gap.get("target_is_prominent"):
            continue
        mt, sc = gap.get("missed_turns", 0), gap.get("stitched_chars", 0)
        if mt < min_turns or (max_turns and mt > max_turns):
            continue
        if sc < min_chars or (max_chars and sc > max_chars):
            continue
        ctx = stitch_gold(corpus, gap)
        if ctx:
            out.append(ctx)
    out.sort(key=lambda c: -(c.get("gold_rank") or 0))
    return out[:limit] if limit else out


def select_silver(corpus: Dict, limit=None) -> List[Dict]:
    out = []
    for s in corpus["sets"]:
        if s.get("tier") != "silver":
            continue
        speakers = s.get("speakers") or []
        if not speakers:
            continue
        # busiest speaker in the set is the most traceable target
        counts = {}
        for t in s.get("turns", []):
            counts[t["speaker"]] = counts.get(t["speaker"], 0) + 1
        target = max(speakers, key=lambda sp: counts.get(sp, 0))
        out.append(silver_context(corpus, s, target))
    out.sort(key=lambda c: -c["stitched_chars"])
    return out[:limit] if limit else out


# --------------------------------------------------------------------------
# tracer construction
# --------------------------------------------------------------------------

def context_roster(ctx: Dict, remap: Dict = None) -> List[str]:
    """Speakers in a context, most central first.

    Ordered by turn count because render_profile caps the roster: if it has to
    drop someone, the person who barely spoke is the right one to drop. Read off
    the stitched `full_context` rather than the source sets' `speakers` lists,
    so a coalition remap (--merge-speakers) is reflected -- the transcript the
    tracer sees is the one the roster has to describe.
    """
    counts = {}
    for line in (ctx.get("full_context") or "").split("\n"):
        name = line.split(":", 1)[0].strip() if ":" in line else ""
        # Speaker tokens are single, unique and prefix-free (see the corpus
        # README); anything with whitespace is a wrapped continuation line.
        if name and " " not in name:
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts, key=lambda s: -counts[s])


def make_args(**overrides) -> SimpleNamespace:
    args = dict(
        use_tracing=True,
        tracing_model="gemini-2.5-flash",
        model="gemini-2.5-flash",
        n_hypotheses=4,
        target_perceptions="sight",
        use_helper_llm=False,
        existing_traces=None,
        input_is_chat=True,      # musing full_context is FANToM chat format
        dataset="musing",
        likelihood_estimate="prompting",
        tracer_type="tracer",
        run_id="musing",
        output_dir="musing_out",
        print=False,
        use_cot=False,
        use_perception_only=False,  # referenced at tracer.py:656, defined by no parser
        hide_action_from_propagation=True,  # de-circularise: see setup_propagation
        scorer='comparative',  # 'isolated' restores the six-bucket per-particle scorer
        scorer_mode='rank',  # rank-first; 'independent'/'allocation' both flatten
        goal_seeding=False,  # PHASE 3a -- off by default so runs stay a clean baseline
        extract_anchors_flag=False,  # instrumentation: canonicalises roots
        use_anchor=False,    # PHASE 3b intervention -- thread anchor through propagation
        anchored_perturbation=False,  # PHASE 3e -- off by default
        root_mass_threshold=0.5,      # 3e fires below this
        baseline_scorer=False,        # null hypothesis on the slate -> absolute fit + surprise
        infer_motive=False,           # abductive proposer + whole-record role prior
        # EXTERNAL baseline profile, injected at SEEDING ONLY -- see profiles.py.
        # Distinct from infer_motive, which derives its prior from the transcript
        # and also speaks at the mint site. Keeping them separate is what lets
        # the two be run as separate arms.
        character_profile=False,
        profiles=None,                # corpus profile records, or None
        profile_roster=None,          # speakers present, most central first
        profile_as=None,              # whose profile to hand over (scramble control)
        legacy_form=False,            # restore pre-G2 propagation + no commitment validator
        standard_prompt='v1',         # STANDARD_PROMPTS variant; see eval_motive_sep.py
        methods=None,                 # methods.py keys; None = default prompts, unchanged
        revive_retired=False,         # expiry becomes a cache: bring a commitment back if it fits
        baseline_blend=0.15,          # uniform mixed back in so a null-level hypothesis survives
        surprise_perturb=False,       # mint when the null outranks every live commitment
        alpha=0.85,                   # prior exponent; ~6.7x steady-state amplification
        beta=1.0,                     # likelihood exponent
        eps_frac=0.12,                # weight floor, as a fraction of 1/N
        ess_divisor=4.0,              # PRODUCTION: resample below pop/4 (=0.25)
        stagnation_steps=3,           # 3e stagnation path: k consecutive low-mass steps
        enable_split=False,           # PHASE 4
        split_weight_quantile=0.20,   # calibrated at N=12: ~1.60x mean, 2.7 firings/run
        split_children=2,
        protect_leader=1,          # keep the heaviest copy of each root
        enable_expiry=False,       # weight-based retirement, off by default
        rebirth_at_fair_share=False,
        revival_rebirth=False,        # the same reset, REVIVED particles only
        expiry_weight_frac=0.5,
        expiry_steps=6,
        merge_percentile=95.0,        # Phase 4 merge cut, resolved per run (jaccard)
        merge_anchors=False,          # also merge particles whose COMMITMENTS are one aim
        merge_floor=0.40,             # absolute similarity floor; see MERGE_ABSOLUTE_FLOOR
        anchor_max_tokens=4096,       # thinking models spend this before writing
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def build_tracer(args):
    from tracer import Tracer, MultiTracer, TracerLight, MultiTracerLight
    cls = {
        "tracer": Tracer,
        "multi-tracer": MultiTracer,
        "tracer-light": TracerLight,
        "multi-tracer-light": MultiTracerLight,
    }[args.tracer_type]
    return cls(args)


def as_tracer_input(ctx: Dict) -> str:
    """The tracer splits on '\\nQuestion:' and extracts a question; give it one."""
    return f"{ctx['full_context']}\n\nQuestion: What is {ctx['target_agent']} thinking?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="oppenheimer")
    ap.add_argument("--role", default="silver", choices=["gold", "silver"])
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=None, help="cap context size to keep a run cheap")
    ap.add_argument("--min-chars", type=int, default=None)
    ap.add_argument("--set-id", default=None,
                    help="trace ONE named set directly (e.g. an odyssey disguise scene), "
                         "bypassing gold/silver selection")
    ap.add_argument("--target", default=None, help="target agent for --set-id")
    ap.add_argument("--merge-speakers", default=None,
                    help="trace a COALITION as one agent, e.g. "
                         "'Product=Deskins,Burdett;Customer=Rodriguez,Letelier'. Those "
                         "speakers are relabelled to the coalition name in the transcript, so "
                         "--target Product traces the group's shared position. Use when one "
                         "side of an argument is too thin to trace alone -- measured on the "
                         "blueberry_size topic, Deskins+Burdett hold 18 turns against "
                         "Rodriguez+Letelier's 366, so the product side cannot be traced "
                         "as an individual.")
    ap.add_argument("--set-ids", default=None,
                    help="comma-separated set ids stitched in order, e.g. a whole episode. "
                         "Joined the same way stitch_gold joins a gap, so the context format "
                         "is identical -- only the selection differs.")
    ap.add_argument("--min-target-turns", type=int, default=0,
                    help="minimum turns the TARGET speaks. High-unknowable contexts have\n                         an absent target, who therefore speaks less and yields a short\n                         trajectory -- 5 steps in one case. Both must be constrained.")
    ap.add_argument("--min-missed-chars", type=int, default=None,
                    help="select gold by UNKNOWABLE content (sum of missed sets), "
                         "which is what makes a context hard -- stitched size is a bad proxy")
    ap.add_argument("--n-hypotheses", type=int, default=4)
    ap.add_argument("--tracing-model", default="gemini-2.5-flash")
    ap.add_argument("--tracer-type", default="tracer")
    ap.add_argument("--output-dir", default="musing_out")
    ap.add_argument("--run-id", default="musing")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--scorer", default="comparative", choices=["comparative","isolated"])
    ap.add_argument("--scorer-mode", default="rank", choices=["rank","independent","allocation"])
    ap.add_argument("--goal-seeding", action="store_true", help="PHASE 3a")
    ap.add_argument("--extract-anchors", action="store_true",
                    help="instrumentation only: extract anchors so roots canonicalise")
    ap.add_argument("--use-anchor", action="store_true", help="PHASE 3b (implies --extract-anchors)")
    ap.add_argument("--anchored-perturbation", action="store_true", help="PHASE 3e (implies --use-anchor)")
    ap.add_argument("--root-mass-threshold", type=float, default=0.5)
    # A null hypothesis on the slate is the only way the system can say "none of
    # these explains it": likelihoods are normalized, so they can never all fall.
    ap.add_argument("--chronological", action="store_true",
                    help="interleave the listed sets' turns by timestamp instead of "
                         "concatenating whole sets, so step order matches real time")
    ap.add_argument("--methods", default=None,
                    help="comma-separated generation methods from methods.py "
                         "(round-robin across particles at seeding and across mint "
                         "events). Several names is the PRIMARY regime: it pairs the "
                         "comparison within one run. One name is a clean single-method "
                         "A/B. Unset leaves every prompt exactly as it was.")
    ap.add_argument("--method", default=None,
                    help="sugar for --methods with a single name")
    ap.add_argument("--standard-prompt", default="v1",
                    help="which STANDARD_PROMPTS variant the proposer/split/seeder use; "
                         "tuned against eval_motive_sep.py")
    ap.add_argument("--legacy-form", action="store_true",
                    help="restore the pre-G2 propagation question and disable the commitment "
                         "validator, so the context fix can be measured on its own")
    ap.add_argument("--revive-retired", action="store_true",
                    help="treat expiry as a cache rather than a delete: when a mint fires, first "
                         "check whether a previously retired commitment would have predicted the "
                         "action, and bring it back with its original root if it beats the null")
    ap.add_argument("--infer-motive", action="store_true",
                    help="derive a SEAT/STAKE prior from the whole record once, and ask the "
                         "proposer what would have to be TRUE for these messages to be worth "
                         "sending, instead of what the speaker wants")
    ap.add_argument("--character-profile", action="store_true",
                    help="seed hypotheses from an external baseline character profile "
                         "(data/musing/<corpus>_profiles.json) instead of reading the "
                         "prior off the transcript. SEEDING ONLY: it founds the first "
                         "population and is used nowhere else, which is what an "
                         "already-held understanding of someone represents. Not named "
                         "--baseline (taken by the scorer null) and not --seed (the RNG).")
    ap.add_argument("--profile-as", default=None, metavar="TOKEN",
                    help="SCRAMBLE CONTROL: trace --target but hand the seeder "
                         "TOKEN's profile instead of their own. If commitments "
                         "follow the profile rather than the person, the prior is "
                         "steering the output rather than carrying information "
                         "about who is being traced -- which is the difference "
                         "between an integration worth building and a prompt that "
                         "makes any motive-shaped prose look like insight.")
    ap.add_argument("--baseline-scorer", action="store_true",
                    help="add a null hypothesis to the comparative slate and score each "
                         "hypothesis by its MARGIN over it (absolute fit, not just relative)")
    ap.add_argument("--baseline-blend", type=float, default=0.15,
                    help="uniform fraction mixed back into margin-derived likelihoods")
    ap.add_argument("--surprise-perturb", action="store_true",
                    help="mint a new commitment when the null outranks every live one "
                         "(implies --baseline-scorer)")
    # Accumulator constants. tracer.accumulate has always read these via
    # getattr(), but nothing could set them and no run recorded which values it
    # used -- so every run before this flag existed is alpha=0.85, beta=1.0,
    # eps_frac=0.12 by construction. They are logged into the run meta now.
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="prior exponent in w_t ~ w_{t-1}^alpha * L_t^beta "
                         "(0 = no accumulation; steady-state amplification ~1/(1-alpha))")
    ap.add_argument("--beta", type=float, default=1.0, help="likelihood exponent")
    ap.add_argument("--eps-frac", type=float, default=0.12,
                    help="weight floor as a fraction of 1/N")
    ap.add_argument("--ess-divisor", type=float, default=4.0, help="resample below pop/DIV (production 4.0)")
    ap.add_argument("--stagnation-steps", type=int, default=3)
    ap.add_argument("--enable-split", action="store_true", help="PHASE 4 split+merge")
    ap.add_argument("--split-weight-quantile", type=float, default=0.20)
    ap.add_argument("--split-children", type=int, default=2)
    ap.add_argument("--protect-leader", type=int, default=1,
                    help="copies of each root kept safe from perturbation (0 = old behaviour)")
    ap.add_argument("--merge-anchors", action="store_true",
                    help="also absorb particles whose COMMITMENTS are the same aim, "
                         "not just whose belief prose is similar. Lexical overlap "
                         "picks candidates; one batched call per merge step decides, "
                         "because no threshold separates 'Avenge her mother' from "
                         "'Forgive her mother's killer'.")
    ap.add_argument("--merge-floor", type=float, default=0.40,
                    help="absolute text-Jaccard floor under the merge percentile. "
                         "0.0 restores the pre-fix behaviour, where the percentile "
                         "alone merged the top pair on 100%% of steps.")
    ap.add_argument("--merge-percentile", type=float, default=95.0)
    ap.add_argument("--rebirth-at-fair-share", action="store_true", default=False,
                    help="mints enter at 1/n instead of inheriting the replaced particle's "
                         "weight. MEASURED NET-NEGATIVE and OFF by default: exp_3 vs exp_4 "
                         "raised median birth mass 0.047->0.113 as intended, but taking that "
                         "mass from established hypotheses lowered root-mass ESS, tripped the "
                         "collapse trigger more often (9->13 firings, 3->6 consecutive), and "
                         "cut mint survival 59%%->32%% and argmax churn 8->5. Kept as the "
                         "evidence for that finding.")
    ap.add_argument("--revival-rebirth", action="store_true", default=False,
                    help="the fair-share reset for REVIVED particles only. OFF by default: "
                         "measured on bb_Rodriguez, 80 steps, two seeds a side, it lifts "
                         "revival birth mass from 0.61-0.77 of fair share to 0.94-0.95 as "
                         "designed, and exp_4's failure does NOT reproduce at this volume -- "
                         "root-mass ESS straddles the control (0.759/0.837 against "
                         "0.775/0.806) rather than falling. But nothing improves: survival "
                         "(0.53/0.78 vs 0.31/0.73) and churn (35/44 vs 50/42) straddle too. "
                         "Kept as the evidence that the revival-only cell is harmless and "
                         "pointless, which is not what exp_4 predicted for it. Needs "
                         "--revive-retired to do anything.")
    ap.add_argument("--enable-expiry", action="store_true",
                    help="retire a hypothesis held below EXPIRY_WEIGHT_FRAC/n for "
                         "EXPIRY_STEPS consecutive turns, minting a replacement")
    ap.add_argument("--expiry-weight-frac", type=float, default=0.5,
                    help="expiry threshold as a fraction of fair share 1/n (0.5 -> 0.0625 at n=8)")
    ap.add_argument("--expiry-steps", type=int, default=6,
                    help="consecutive INFORMATIVE turns (resample steps frozen) a root "
                         "must hold sub-threshold mass before retirement. Measured: regret "
                         "rate is flat at ~8%% across k, so k sets turnover volume, not "
                         "accuracy -- 6 gives ~5 retirements/run.")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed resampling draws. NOTE: fixes WHICH particles are duplicated, "
                         "not WHETHER a resample fires -- trigger crossings follow the ESS "
                         "trajectory, which varies with scorer non-determinism.")
    ap.add_argument("--show-action-to-propagation", action="store_true",
                    help="restore the circular pre-fix behaviour, for A/B")
    # select_gold's band lives in its own keyword defaults and main() called it
    # with no arguments, so --min-chars/--max-chars could only ever narrow the
    # already-banded output. These reach the selector itself.
    ap.add_argument("--no-prominent-only", action="store_true",
                    help="drop select_gold's prominent-target requirement")
    ap.add_argument("--min-missed-turns", type=int, default=8,
                    help="select_gold's min_turns; the default 8 is its own")
    ap.add_argument("--band-min-chars", type=int, default=8000,
                    help="select_gold's min_chars. Lower to widen the pool: the default "
                         "band yields 17 gold contexts at >=20 target turns, relaxed 44")
    ap.add_argument("--band-max-chars", type=int, default=30000,
                    help="select_gold's max_chars")
    ap.add_argument("--min-action-steps", type=int, default=0,
                    help="floor on trajectory length. The coverage gap only appears where\n"
                         "resampling fires and resampling needs steps, so a short context is\n"
                         "a guaranteed tie. Selection on a PROCESS property, independent of\n"
                         "the verdicts, so it does not bias which cases come out plausible")
    a = ap.parse_args()

    # Resolve and VALIDATE methods at parse time. method_rule() is tolerant by
    # design -- an unknown key degrades to the default prompt rather than
    # raising deep inside a run -- so a typo would otherwise produce a run that
    # looks like it applied a method for 40 steps without ever having done so.
    try:
        a.methods = parse_methods(a.methods or a.method)
    except ValueError as e:
        raise SystemExit(str(e))
    if a.methods:
        # A method label on a particle with no anchor attaches to nothing
        # measurable: roots do not canonicalise without the extractor, and
        # every method-keyed metric in the audit reads `anchor`.
        a.extract_anchors = True
        # WARN, do not enable. With the source term off, methods affect seeding
        # only -- a real experiment, and a cheaper one, but a different one.
        # Silently switching on 3e would change the population dynamics and
        # confound the method sweep with the operator.
        if not a.anchored_perturbation:
            print("[yellow]--methods without --anchored-perturbation: methods will "
                  "affect SEEDING only; no minted commitment will carry a method.[/yellow]")
        needs_prior = [k for k in a.methods if METHODS[k].needs_role_prior]
        if needs_prior and not a.infer_motive:
            print(f"[yellow]{', '.join(needs_prior)} read from a role prior that is off; "
                  f"add --infer-motive or expect them to underperform.[/yellow]")

    if a.seed is not None:
        random.seed(a.seed)
        import numpy as _np
        _np.random.seed(a.seed)
    corpus = load_corpus(a.corpus)
    if a.set_ids:
        ids = [x.strip() for x in a.set_ids.split(",") if x.strip()]
        # coalition map: speaker -> coalition name
        remap = {}
        if a.merge_speakers:
            for grp in a.merge_speakers.split(";"):
                if "=" not in grp:
                    continue
                name, members = grp.split("=", 1)
                for m in members.split(","):
                    if m.strip():
                        remap[m.strip()] = name.strip()
        parts, used, nt = [], [], 0
        line = lambda t: f"{remap.get(t['speaker'], t['speaker'])}: {t.get('text') or ''}"
        if a.chronological:
            # Interleave every turn by timestamp instead of concatenating whole
            # sets. Sorting the SETS is not enough: bloomfield-0329 alone runs
            # 2024-10-28 to 2025-01-08, so listing it first puts January content
            # ahead of October content from every later set. Measured on the
            # blueberry_size span: the filter met Burdett's 7 Jan "no" at turn 24
            # and only reached October traffic at turn 39, which makes any
            # "before the reveal" claim about step order meaningless.
            #
            # This mixes threads, which is a real cost -- a Slack set is one
            # conversation and interleaving breaks its adjacency. It is the right
            # trade only when the question is about WHEN something became
            # knowable, which is exactly what early-detection asks.
            all_turns = []
            for sid in ids:
                st = corpus["by_id"].get(sid)
                if st is None:
                    raise SystemExit(f"set {sid} not found in {a.corpus}")
                all_turns.extend(st["turns"]); used.append(sid); nt += st["n_turns"]
            all_turns.sort(key=lambda t: t.get("sent") or "")
            parts = ["\n".join(line(t) for t in all_turns)]
            print(f"[chronological] {nt} turns interleaved, "
                  f"{(all_turns[0].get('sent') or '')[:10]} -> {(all_turns[-1].get('sent') or '')[:10]}")
        else:
            for sid in ids:
                st = corpus["by_id"].get(sid)
                if st is None:
                    raise SystemExit(f"set {sid} not found in {a.corpus}")
                if remap:
                    # rebuild the transcript with coalition labels. Verified lossless
                    # against full_context on 200/200 sets when remap is empty.
                    parts.append("\n".join(line(t) for t in st["turns"]))
                else:
                    parts.append(st["full_context"])
                used.append(sid); nt += st["n_turns"]
        if not a.target:
            raise SystemExit("--set-ids requires --target")
        blob = "\n".join(parts)
        contexts = [{"context_id": f"{a.corpus}:{a.target}:{used[0]}->{used[-1]}",
                     "role": "scene", "target_agent": a.target, "full_context": blob,
                     "set_ids": used, "missed_set_ids": [],
                     "stitched_chars": len(blob), "n_turns": nt}]
    elif a.set_id:
        st = corpus["by_id"].get(a.set_id)
        if st is None:
            raise SystemExit(f"set {a.set_id} not found in {a.corpus}")
        tgt = a.target or max(st.get("speakers") or [],
                              key=lambda sp: sum(1 for t in st["turns"] if t["speaker"] == sp))
        contexts = [{"context_id": f"{a.corpus}:{tgt}:{a.set_id}", "role": "scene",
                     "target_agent": tgt, "full_context": st["full_context"],
                     "set_ids": [a.set_id], "missed_set_ids": [],
                     "stitched_chars": len(st["full_context"]), "n_turns": st["n_turns"]}]
    else:
        contexts = (select_gold(corpus,
                                prominent_only=not a.no_prominent_only,
                                min_turns=a.min_missed_turns,
                                min_chars=a.band_min_chars,
                                max_chars=a.band_max_chars)
                    if a.role == "gold" else select_silver(corpus))
    if a.max_chars:
        contexts = [c for c in contexts if c["stitched_chars"] <= a.max_chars]
    if a.min_chars:
        contexts = [c for c in contexts if c["stitched_chars"] >= a.min_chars]
    if a.min_missed_chars:
        def _missed(ctx):
            return sum(len(corpus["by_id"][s]["full_context"])
                       for s in ctx["missed_set_ids"] if s in corpus["by_id"])
        contexts = [c for c in contexts if _missed(c) >= a.min_missed_chars]
        contexts.sort(key=lambda c: _missed(c))   # smallest qualifying = cheapest
    if a.min_action_steps:
        # A trajectory step carries an action only when the target speaks, and
        # only those steps reweight, so the target's own turn count -- not the
        # context's total -- is what bounds how many times the filter can
        # resample. Same quantity --min-target-turns measures; kept separate so
        # the process-property floor is legible as its own pre-registered choice.
        def _acts(ctx):
            n = 0
            for sid in ctx["set_ids"]:
                st = corpus["by_id"].get(sid)
                if st:
                    n += sum(1 for t in st["turns"] if t["speaker"] == ctx["target_agent"])
            return n
        before = len(contexts)
        contexts = [c for c in contexts if _acts(c) >= a.min_action_steps]
        print(f"  --min-action-steps {a.min_action_steps}: {before} -> {len(contexts)} contexts")
    if a.min_target_turns:
        def _tturns(ctx):
            n = 0
            for sid in ctx["set_ids"]:
                st = corpus["by_id"].get(sid)
                if st:
                    n += sum(1 for t in st["turns"] if t["speaker"] == ctx["target_agent"])
            return n
        contexts = [c for c in contexts if _tturns(c) >= a.min_target_turns]
    contexts = contexts[: a.limit]
    if not contexts:
        raise SystemExit(f"no {a.role} contexts matched in {a.corpus}")

    args = make_args(tracing_model=a.tracing_model, model=a.tracing_model,
                     n_hypotheses=a.n_hypotheses, tracer_type=a.tracer_type,
                     output_dir=a.output_dir, run_id=a.run_id, print=a.print,
                     hide_action_from_propagation=not a.show_action_to_propagation,
                     scorer=a.scorer, scorer_mode=a.scorer_mode,
                     goal_seeding=a.goal_seeding,
                     # 3e is meaningless without an anchor to replace
                     extract_anchors_flag=a.extract_anchors or a.use_anchor or a.anchored_perturbation,
                     use_anchor=a.use_anchor or a.anchored_perturbation,
                     anchored_perturbation=a.anchored_perturbation,
                     root_mass_threshold=a.root_mass_threshold,
                     alpha=a.alpha, beta=a.beta, eps_frac=a.eps_frac,
                     baseline_scorer=a.baseline_scorer or a.surprise_perturb,
                     infer_motive=a.infer_motive,
                     character_profile=a.character_profile,
                     profiles=corpus.get("profiles"),
                     profile_as=a.profile_as,
                     revive_retired=a.revive_retired,
                     legacy_form=a.legacy_form,
                     standard_prompt=a.standard_prompt,
                     methods=a.methods,
                     baseline_blend=a.baseline_blend,
                     surprise_perturb=a.surprise_perturb,
                     ess_divisor=a.ess_divisor, merge_percentile=a.merge_percentile,
                     merge_anchors=a.merge_anchors, merge_floor=a.merge_floor,
                     stagnation_steps=a.stagnation_steps,
                     enable_split=a.enable_split,
                     split_weight_quantile=a.split_weight_quantile,
                     split_children=a.split_children,
                     protect_leader=a.protect_leader,
                     enable_expiry=a.enable_expiry,
                     rebirth_at_fair_share=a.rebirth_at_fair_share,
                     revival_rebirth=a.revival_rebirth,
                     expiry_weight_frac=a.expiry_weight_frac,
                     expiry_steps=a.expiry_steps)
    tracer = build_tracer(args)

    from trace_log import RunLogger

    for i, ctx in enumerate(contexts):
        print(f"[{ctx['role']}] {ctx['context_id']}  chars={ctx['stitched_chars']}")
        run_id = f"{a.run_id}-{a.corpus}-{a.role}-{i:03d}"
        # Per-context, because the roster is a property of the transcript being
        # traced, not of the run. The tracer renders from args at
        # set_tracer_variables, so this has to be in place before trace().
        args.profile_roster = context_roster(ctx)
        pmeta = (profile_meta(corpus.get("profiles"),
                              a.profile_as or ctx["target_agent"],
                              args.profile_roster, traced=ctx["target_agent"])
                 if a.character_profile else {})
        logger = RunLogger(a.output_dir, run_id, meta={
            "corpus": a.corpus,
            "role": ctx["role"],
            "corpus_role": "trajectory" if a.role == "gold" else "control",
            "context_id": ctx["context_id"],
            "target_agent": ctx["target_agent"],
            "n_hypotheses": a.n_hypotheses,
            "tracer_type": a.tracer_type,
            "model": a.tracing_model,
            "stitched_chars": ctx["stitched_chars"],
            "hide_action_from_propagation": not a.show_action_to_propagation,
            "scorer": a.scorer,
            "seed": a.seed,
            "goal_seeding": a.goal_seeding,
            "extract_anchors": a.extract_anchors or a.use_anchor or a.anchored_perturbation,
            "use_anchor": a.use_anchor or a.anchored_perturbation,
            "anchored_perturbation": a.anchored_perturbation,
            "root_mass_threshold": a.root_mass_threshold,
            "alpha": a.alpha,
            "baseline_scorer": a.baseline_scorer or a.surprise_perturb,
            "infer_motive": a.infer_motive,
            "revive_retired": a.revive_retired,
            "legacy_form": a.legacy_form,
            "standard_prompt": a.standard_prompt,
            "methods": a.methods,
            "method_assignment": ("mixed" if a.methods and len(a.methods) > 1
                                  else ("single" if a.methods else None)),
            "baseline_blend": a.baseline_blend,
            "surprise_perturb": a.surprise_perturb,
            "beta": a.beta,
            "eps_frac": a.eps_frac,
            "ess_divisor": a.ess_divisor,
            "stagnation_steps": a.stagnation_steps,
            "enable_expiry": a.enable_expiry,
            "rebirth_at_fair_share": a.rebirth_at_fair_share,
            "revival_rebirth": a.revival_rebirth,
            "expiry_weight_frac": a.expiry_weight_frac,
            "expiry_steps": a.expiry_steps,
            "enable_split": a.enable_split,
            "merge_percentile": a.merge_percentile,
            "merge_anchors": a.merge_anchors,
            "merge_floor": a.merge_floor,
            # self-describing runs: the evaluator reconstructs the judge window
            # from these rather than re-deriving selection, which is how four
            # earlier evaluations produced invalid numbers.
            "set_ids": ctx.get("set_ids"),
            "missed_set_ids": ctx.get("missed_set_ids"),
            "scorer_mode": a.scorer_mode,
            # Arm identity. audit_profile.py partitions on this; a run whose
            # profile file was missing the target reads as character_profile
            # true with profile_target null, which is visibly different from a
            # run that had one -- rather than both reporting the same thing.
            "character_profile": a.character_profile,
            **pmeta,
        })
        tracer.attach_logger(logger)
        try:
            tracer.trace(as_tracer_input(ctx), target_agent=ctx["target_agent"])
        finally:
            summary = logger.close()
            tracer.run_logger = None
        # FILE-LENGTH GUARD. Four runs once wrote zero-length step files while
        # their wrapper still recorded completion, so the failure was invisible
        # until the analysis found empty globs. A run that produced no steps is
        # a failed run and must say so loudly and exit non-zero.
        written = 0
        try:
            with open(logger.path, encoding="utf-8") as fh:
                written = sum(1 for line in fh if line.strip())
        except OSError:
            written = 0
        expected = getattr(tracer, "_last_trajectory_len", 0) or 0
        floor = max(MIN_STEPS, int(expected * MIN_FRACTION)) if expected else MIN_STEPS
        if written < floor:
            print(f"!!! RUN FAILED: {run_id} wrote {written} steps "
                  f"(minimum {floor}) -> {logger.path}")
            raise SystemExit(2)
        print(f"[ok] {run_id}: {written} steps written")
        print(json.dumps({k: summary[k] for k in (
            "run_id", "steps", "median_likelihood_ess", "median_posterior_ess",
            "reversals_median", "argmax_churn", "split_count", "merge_count",
            "effective_temperatures", "total_llm_calls") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
