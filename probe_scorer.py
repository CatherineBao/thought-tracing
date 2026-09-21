"""Fixed test case for the likelihood scorer.

Feeds the scorer N textually IDENTICAL hypotheses and, as contrast, N genuinely
distinct ones, on the same real mid-trajectory step.

This exists before Phase 1 rewrites the scorer, because it is the guard against
the Phase 1 gate passing on manufactured differences. Identical inputs carry
zero information, so any spread the scorer reports on them is noise. A
comparative scorer under forced allocation is *structurally obliged* to produce
a spread there, which is exactly the failure the gate cannot see. Re-run this
against the new scorer: the identical case should stay flat, or the gate is
measuring its own constraint.

The context is cached so repeat runs cost only the scoring calls.
"""
import argparse
import json
import os

import trace_log
import run_musing as rm

CACHE = "musing_out/probe_context.json"


def prepare(tracer, ctx, step):
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            c = json.load(fh)
        if c.get("context_id") == ctx["context_id"] and c.get("step") == step:
            return c
    pre = tracer.preprocess_input(rm.as_tracer_input(ctx), ctx["target_agent"])
    tracer.set_tracer_variables(pre)
    traj, perc = pre["trajectory"], pre["perceptions"]
    idx = next((i for i in range(min(step, len(traj) - 1), len(traj)) if traj[i]["action"]), None)
    if idx is None:
        raise SystemExit("no action step found")
    hyps = tracer.initialize(state_action=traj[0], perceptions=perc[0])
    # Hypotheses CONTEMPORANEOUS with the action: propagated up to step idx the
    # way the real filter does. Step-0 hypotheses against a step-8 action is a
    # mismatch -- every hypothesis is irrelevant, the scorer correctly answers
    # "none of these predict it", and the fixture cannot separate scorer designs.
    from hypothesis import HypothesesSetV3
    cur = hyps
    for j in range(1, idx + 1):
        cur = tracer.propagate(cur, state_action=traj[j], perceptions=perc[j])
    c = {"context_id": ctx["context_id"], "step": step, "action_step": idx,
         "contexts": traj[: idx + 1], "perceptions": perc[: idx + 1],
         "action": traj[idx]["action"], "target_agent": ctx["target_agent"],
         "distinct": list(hyps.texts), "contemporaneous": list(cur.texts),
         "assumption": tracer.assumption}
    os.makedirs("musing_out", exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=2)
    return c


def score(tracer, c, hypotheses, label):
    trace_log.RECORDER.drain()
    fn = (tracer.prompt_likelihood_comparative
          if getattr(tracer.args, "scorer", "comparative") == "comparative"
          else tracer.prompt_likelihood)
    res = fn(hypotheses, c["contexts"], c["perceptions"],
             c["action"], target_agent=c["target_agent"])
    w = [float(x) for x in res["weights"]]
    letters = res.get("letters")
    return {
        "label": label,
        "n": len(hypotheses),
        "identical_inputs": len(set(h.strip() for h in hypotheses)) == 1,
        "letters": letters,
        "parse_failures": res.get("parse_failures"),
        "raw_scores": [None if x is None else float(x) for x in (res["raw_scores"] or [])],
        "weights": [round(x, 4) for x in w],
        "likelihood_ess": trace_log.ess(w, mask=res.get("parse_mask")),
        "likelihood_ess_norm": trace_log.ess_norm(w, mask=res.get("parse_mask")),
        "allocation": res.get("allocation"),
        "all_zero": res.get("all_zero", False),
        "ties": res.get("ties"),
        "answers": [a[:60] for a in (res.get("answers") or [])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="oppenheimer")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default="musing_out/probe_scorer.json")
    ap.add_argument("--scorer", default="comparative", choices=["comparative","isolated"])
    ap.add_argument("--scorer-mode", default="rank", choices=["rank","independent","allocation"])
    a = ap.parse_args()

    corpus = rm.load_corpus(a.corpus)
    pool = sorted(rm.select_silver(corpus), key=lambda c: c["stitched_chars"])
    ctx = pool[-1]
    args = rm.make_args(tracing_model=a.model, model=a.model, n_hypotheses=a.n,
                        run_id="probe", output_dir="musing_out", scorer=a.scorer, scorer_mode=a.scorer_mode)
    tracer = rm.build_tracer(args)
    tracer.target_agent = ctx["target_agent"]

    c = prepare(tracer, ctx, a.step)
    tracer.assumption = c["assumption"]
    tracer.target_agent = c["target_agent"]
    print(f"context {c['context_id']}  action step {c['action_step']}")
    print(f"action: {c['action'][:110]}...\n")

    contemp = (c.get("contemporaneous") or c["distinct"])[: a.n]
    identical = [contemp[0]] * a.n
    mismatched = c["distinct"][: a.n]

    results = [score(tracer, c, identical, "IDENTICAL x%d" % a.n),
               score(tracer, c, contemp, "CONTEMP x%d" % a.n),
               score(tracer, c, mismatched, "MISMATCH x%d" % a.n)]

    gate = 0.75 * a.n
    for r in results:
        print(f"{r['label']:>14}  letters={r['letters']}  parse_fail={r['parse_failures']}")
        print(f"{'':>14}  raw={r['raw_scores']}  weights={r['weights']}")
        en = r.get('likelihood_ess_norm')
        print(f"{'':>14}  alloc={r.get('allocation')}  ties={r.get('ties')}  all_zero={r.get('all_zero')}")
        print(f"{'':>14}  likelihood_ess_norm={en if en is None else round(en,3)}  (gate 0.75)")
        for i, ans in enumerate(r["answers"] or []):
            print(f"{'':>14}    answer[{i}]: {ans!r}")
        print()

    ident = results[0]
    verdict = {
        "identical_is_flat": (ident.get("likelihood_ess_norm") or 0) > 0.95,
        "identical_norm": ident.get("likelihood_ess_norm"),
        "contemp_norm": results[1].get("likelihood_ess_norm"),
        "separation": ((ident.get("likelihood_ess_norm") or 0)
                       - (results[1].get("likelihood_ess_norm") or 0)),
        "middle_buckets_used": sorted({l for r in results for l in (r["letters"] or []) if l in ("b","c","d","e")}),
    }
    print("--- probe ---")
    en = ident.get('likelihood_ess_norm')
    print(f"  identical inputs -> flat? {'YES' if verdict['identical_is_flat'] else 'NO'} "
          f"(normalized ess {en if en is None else round(en,3)}, 1.0 = flat)")
    if not verdict["identical_is_flat"]:
        print("  -> the scorer reports a spread over inputs that carry zero information.")
        print("     Any Phase 1 gate pass has to be checked against this case.")
    print(f"  middle buckets (b/c/d/e) used anywhere: {verdict['middle_buckets_used'] or 'NONE'}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"context": c["context_id"], "action_step": c["action_step"],
                   "results": results, "verdict": verdict}, fh, indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
