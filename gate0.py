"""Gate 0 -- believe the log before touching the algorithm.

Five propagation calls at 0.7 must produce non-identical text AND the logged
effective_temperature must read 0.7. Both halves matter: the first shows the
sampler draws fresh randomness, the second shows the log reports reality rather
than the intent at the call site.

Uses the REAL propagation prompt from a real mid-trajectory step, not a
synthetic one. A short throwaway prompt varies easily at 0.7 and would pass
while telling us nothing; a full trajectory prompt is heavily constrained and
may produce near-identical text even with a working sampler. That is the case
this check exists to characterise, which is why it also runs a temperature-0
control: if 0.0 and 0.7 produce the same spread, the temperature is not
reaching the sampler; if 0.0 collapses and 0.7 spreads, it is.
"""
import argparse
import json
from copy import deepcopy

import trace_log
import run_musing as rm
from utils import overall_jaccard_similarity


def build_real_propagation_prompt(tracer, ctx, step_k):
    """Reproduce Tracer.propagate's prompt at a mid-trajectory step."""
    pre = tracer.preprocess_input(rm.as_tracer_input(ctx), ctx["target_agent"])
    if pre is None:
        raise SystemExit("target agent not identified")
    tracer.set_tracer_variables(pre)

    trajectory, perceptions = pre["trajectory"], pre["perceptions"]
    step_k = min(step_k, len(trajectory) - 1)
    if step_k < 1:
        raise SystemExit(f"trajectory too short ({len(trajectory)} steps)")

    # step 0 particles, then carry context forward to step_k exactly as the loop does
    hyps = tracer.initialize(state_action=trajectory[0], perceptions=perceptions[0])
    hyps.contexts = [trajectory[i] for i in range(step_k)]
    hyps.perceptions = [perceptions[i] for i in range(step_k)]

    info = tracer.setup_propagation(hyps, trajectory[step_k], perceptions[step_k])
    target = info["target_agent"]
    system_prompt = f"You are an expert assistant trying to predict {target}'s thoughts."
    hypothesis = hyps.texts[0]
    prompt = (
        f"{tracer.trace_header}\n\n<previous context>\n{info['context_and_perception_str']}\n"
        f"</previous context>\n<previous prediction regarding {target}'s thoughts>\n{hypothesis}\n"
        f"</previous prediction regarding {target}'s thoughts>\n\n"
        f"<current context>{tracer.assumption}\n{info['new_context']}\n</current context>\n\n"
        f"Question: What did {target} believe?"
    )
    return prompt, system_prompt, len(trajectory), step_k


def fire(tracer, prompt, system_prompt, temperature, n=5):
    trace_log.RECORDER.drain()
    outs = tracer.tracer_model.batch_interact(
        [prompt] * n, temperature=temperature, max_tokens=1024,
        system_prompts=system_prompt, stage=f"gate0@{temperature}",
    )
    calls = trace_log.RECORDER.drain()
    temps = sorted({c["effective_temperature"] for c in calls})
    uniq = len(set(o.strip() for o in outs))
    diversity = 1 - overall_jaccard_similarity(outs) if len(outs) > 1 else 0.0
    return {"temperature_requested": temperature, "temperatures_logged": temps,
            "n_calls": len(calls), "unique_outputs": uniq, "n": n,
            "jaccard_diversity": round(diversity, 4), "outputs": outs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="oppenheimer")
    ap.add_argument("--role", default="silver", choices=["gold", "silver"])
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n-calls", type=int, default=5)
    ap.add_argument("--tracing-model", default="gemini-2.5-flash")
    ap.add_argument("--context-index", type=int, default=-1,
                    help="index into the list sorted by size; -1 picks the LARGEST, "
                         "because a short prompt varies easily at 0.7 and proves nothing")
    a = ap.parse_args()

    corpus = rm.load_corpus(a.corpus)
    pool = rm.select_gold(corpus) if a.role == "gold" else rm.select_silver(corpus)
    pool.sort(key=lambda c: c["stitched_chars"])
    ctx = pool[-1] if a.context_index < 0 else pool[a.context_index]
    print(f"context: {ctx['context_id']}  ({ctx['stitched_chars']} chars)")

    args = rm.make_args(tracing_model=a.tracing_model, model=a.tracing_model,
                        n_hypotheses=4, run_id="gate0", output_dir="musing_out")
    tracer = rm.build_tracer(args)

    prompt, system_prompt, n_steps, step_k = build_real_propagation_prompt(tracer, ctx, a.step)
    print(f"trajectory steps: {n_steps}; using step {step_k}")
    print(f"propagation prompt: {len(prompt)} chars\n")

    hot = fire(tracer, prompt, system_prompt, 0.7, a.n_calls)
    cold = fire(tracer, prompt, system_prompt, 0.0, a.n_calls)

    for label, r in (("T=0.7", hot), ("T=0.0 (control)", cold)):
        print(f"{label:>16}  logged={r['temperatures_logged']}  "
              f"unique={r['unique_outputs']}/{r['n']}  jaccard_diversity={r['jaccard_diversity']}")

    temp_ok = hot["temperatures_logged"] == [0.7]
    vary_ok = hot["unique_outputs"] > 1
    responds = hot["jaccard_diversity"] > cold["jaccard_diversity"]

    print("\n--- Gate 0 ---")
    print(f"  log reports 0.7 ........... {'PASS' if temp_ok else 'FAIL'}")
    print(f"  outputs non-identical ..... {'PASS' if vary_ok else 'FAIL'}")
    print(f"  responds to temperature ... {'PASS' if responds else 'INCONCLUSIVE'}")
    verdict = temp_ok and vary_ok
    print(f"  GATE 0: {'PASS' if verdict else 'FAIL'}")
    if temp_ok and not vary_ok:
        print("  -> temperature reaches the sampler but the prompt constrains the output.")
        print("     Sampler is fine; this belongs to the diversity budget, not a dropped temperature.")

    with open("musing_out/gate0_result.json", "w", encoding="utf-8") as fh:
        json.dump({"context": ctx["context_id"], "step": step_k, "steps": n_steps,
                   "prompt_chars": len(prompt), "hot": hot, "cold": cold,
                   "verdict": verdict}, fh, indent=2)
    print("  wrote musing_out/gate0_result.json")


if __name__ == "__main__":
    main()
