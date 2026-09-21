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
        ess_divisor=4.0,              # PRODUCTION: resample below pop/4 (=0.25)
        stagnation_steps=3,           # 3e stagnation path: k consecutive low-mass steps
        enable_split=False,           # PHASE 4
        split_weight_quantile=0.20,   # calibrated at N=12: ~1.60x mean, 2.7 firings/run
        split_children=2,
        protect_leader=1,          # keep the heaviest copy of each root
        merge_percentile=95.0,        # Phase 4 merge cut, resolved per run (jaccard)
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
    ap.add_argument("--ess-divisor", type=float, default=4.0, help="resample below pop/DIV (production 4.0)")
    ap.add_argument("--stagnation-steps", type=int, default=3)
    ap.add_argument("--enable-split", action="store_true", help="PHASE 4 split+merge")
    ap.add_argument("--split-weight-quantile", type=float, default=0.20)
    ap.add_argument("--split-children", type=int, default=2)
    ap.add_argument("--protect-leader", type=int, default=1,
                    help="copies of each root kept safe from perturbation (0 = old behaviour)")
    ap.add_argument("--merge-percentile", type=float, default=95.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed resampling draws. NOTE: fixes WHICH particles are duplicated, "
                         "not WHETHER a resample fires -- trigger crossings follow the ESS "
                         "trajectory, which varies with scorer non-determinism.")
    ap.add_argument("--show-action-to-propagation", action="store_true",
                    help="restore the circular pre-fix behaviour, for A/B")
    a = ap.parse_args()

    if a.seed is not None:
        random.seed(a.seed)
        import numpy as _np
        _np.random.seed(a.seed)
    corpus = load_corpus(a.corpus)
    if a.set_id:
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
        contexts = (select_gold(corpus) if a.role == "gold" else select_silver(corpus))
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
                     ess_divisor=a.ess_divisor, merge_percentile=a.merge_percentile,
                     stagnation_steps=a.stagnation_steps,
                     enable_split=a.enable_split,
                     split_weight_quantile=a.split_weight_quantile,
                     split_children=a.split_children,
                     protect_leader=a.protect_leader)
    tracer = build_tracer(args)

    from trace_log import RunLogger

    for i, ctx in enumerate(contexts):
        print(f"[{ctx['role']}] {ctx['context_id']}  chars={ctx['stitched_chars']}")
        run_id = f"{a.run_id}-{a.corpus}-{a.role}-{i:03d}"
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
            "ess_divisor": a.ess_divisor,
            "stagnation_steps": a.stagnation_steps,
            "enable_split": a.enable_split,
            "merge_percentile": a.merge_percentile,
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
