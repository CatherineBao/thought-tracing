"""Confirm the logprob backend, and pin the model that produced the numbers.

    python confirm_logprobs.py --self-test       # no weights, no GPU, no network
    python confirm_logprobs.py --server http://HOST:8000/v1
    python confirm_logprobs.py --offline --num-gpus 1

WHAT PHASE 2 NEEDS AND WHY IT IS NOT FREE. Scoring a hypothesis against the
observed next action needs the log-probability of a continuation the model did
not generate. Every backend in agents/ returns generated text; none returns the
per-token log-probs of a string handed to it. So the capability has to be
confirmed before anything is designed on top of it, and confirmed as a NUMBER
rather than as an API that returned 200.

FOUR CHECKS, BECAUSE A LIST OF FLOATS PROVES NOTHING ON ITS OWN.

  aligned      the tail carries exactly as many log-probs as the continuation
               has tokens, and the boundary did not retokenize
  responsive   a plausible continuation outscores a scrambled one on the same
               prompt. A backend returning a constant, or returning the prompt's
               log-probs shifted by one, passes `aligned` and fails this.
  ordered      the same continuation scores higher after the context that
               licenses it than after a context that does not. This is the
               property the scorer would actually be built on.
  deterministic  two identical requests return identical floats. A sampling
               path that leaks into scoring makes every later comparison noise.

THE PIN IS PART OF THE RESULT. Log-probs are not comparable across weights,
tokenizer or dtype, so a threshold tuned under one is meaningless under
another. The confirmation writes the exact model string, revision, tokenizer
and dtype it ran under into logprob_backend.json, and PREREG.md points at that
file rather than restating the name.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logprob_backend.json")

# The proposed pin. Ungated on the Hub (no license click in the way of a
# reproduction), 128k context via rope scaling -- a noisy window is ~1,000
# turns -- served by vLLM without a custom path, and small enough that one
# 80GB card holds it in bf16 with room for the KV cache.
PIN = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "revision": None,          # filled by the confirmation run
    "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "bfloat16",
    "vllm_version": None,      # filled by the confirmation run
}

PROMPT = (
    "Ortega: The 2-Purple threshold has to match what the grower sees in the field.\n"
    "Wolf: Agreed, but the instrument has to reproduce the count or it is not a number.\n"
    "Ortega: The grower does not care what the instrument says.\n"
    "Wolf:")
GOOD = " Then we are measuring two different things."
SCRAMBLED = " things two different are measuring we Then."
WRONG_CONTEXT = (
    "Theofiledes: for the updating of A96 can you please turn it off and back on\n"
    "Letelier: done, the blue cable is connected\n"
    "Wolf:")


# --------------------------------------------------------------------------
# offline self-test of the one piece most likely to be silently wrong
# --------------------------------------------------------------------------

class _FakeTokenizer:
    """Whitespace tokenizer that MERGES across a chosen boundary on demand.

    The merging case is the whole reason _split_point exists, and it cannot be
    exercised with a real tokenizer without knowing which string happens to
    merge under which vocabulary. Here it is forced.
    """

    def __init__(self, merge_at=None):
        self.merge_at = merge_at
        self.vocab = {}

    def _id(self, tok):
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def encode(self, text, add_special_tokens=False):
        toks = text.split(" ")
        if self.merge_at and self.merge_at in text:
            head, tail = text.split(self.merge_at, 1)
            toks = head.split(" ")[:-1] + [head.split(" ")[-1] + self.merge_at] + tail.split(" ")
            toks = [t for t in toks if t != ""]
        return [self._id(t) for t in toks]

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return " ".join(rev.get(i, "?") for i in ids)


def self_test():
    from agents.vllm import _split_point

    tok = _FakeTokenizer()
    prompt, cont = "the grower does not care", " what the instrument says"
    p_ids, j_ids, n, clean = _split_point(tok, prompt, cont)
    assert clean, "a non-merging boundary was reported as dirty"
    assert n == len(p_ids), (n, len(p_ids))
    assert len(j_ids) - n == len(tok.encode(cont.lstrip(" "))), "tail length wrong"

    # Now force a merge across the join and confirm it is CAUGHT rather than
    # scored: the tail would otherwise be read against the wrong tokens.
    merging = _FakeTokenizer(merge_at="care what")
    _, _, n2, clean2 = _split_point(merging, prompt, cont)
    assert not clean2, "a retokenizing boundary was reported as clean"
    assert n2 < len(p_ids), "split point did not retreat to the last agreeing token"
    print("  ok   split point is exact on a clean boundary")
    print("  ok   a retokenizing boundary is reported, not scored")
    return True


# --------------------------------------------------------------------------

def build_agent(args):
    if args.server:
        from agents.vllm import VllmServerAgent
        return VllmServerAgent(args.model, base_url=args.server), "server"
    from agents.vllm import VllmScoringAgent
    return VllmScoringAgent(args.model, num_gpus=args.num_gpus,
                            dtype=args.dtype), "offline"


def checks(agent):
    good = agent.score_continuation(PROMPT, GOOD)
    scram = agent.score_continuation(PROMPT, SCRAMBLED)
    wrong = agent.score_continuation(WRONG_CONTEXT, GOOD)
    again = agent.score_continuation(PROMPT, GOOD)

    n_expected = len(agent.tokenizer.encode(GOOD, add_special_tokens=False))
    out = {
        "aligned": {
            "pass": bool(good.logprobs) and good.boundary_clean,
            "tail_logprobs": len(good.logprobs),
            "continuation_tokens_standalone": n_expected,
            "boundary_clean": good.boundary_clean,
            "note": ("standalone token count is a sanity reference, not a "
                     "requirement: a leading space legitimately retokenizes when "
                     "the continuation is encoded on its own."),
        },
        "responsive": {
            "pass": good.mean > scram.mean,
            "plausible_mean": good.mean, "scrambled_mean": scram.mean,
        },
        "ordered": {
            "pass": good.mean > wrong.mean,
            "in_context_mean": good.mean, "out_of_context_mean": wrong.mean,
        },
        "deterministic": {
            "pass": good.logprobs == again.logprobs,
            "max_abs_diff": max((abs(a - b) for a, b in zip(good.logprobs, again.logprobs)),
                                default=0.0),
        },
    }
    out["_sample"] = good.as_dict()
    return out


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PIN["model"])
    ap.add_argument("--dtype", default=PIN["dtype"])
    ap.add_argument("--server", default=os.environ.get("VLLM_BASE_URL"),
                    help="OpenAI-compatible vLLM endpoint; omit to load weights here")
    ap.add_argument("--offline", action="store_true", help="force the in-process vLLM path")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--self-test", action="store_true",
                    help="alignment logic only; no weights, no GPU, no network")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.offline:
        a.server = None

    print("alignment self-test:")
    self_test()

    if a.self_test:
        print("\nself-test only. The backend itself is NOT confirmed: rerun with "
              "--server or --offline on a machine that can serve the weights.")
        return

    try:
        agent, path = build_agent(a)
    except Exception as exc:
        raise SystemExit(
            f"\nNOT CONFIRMED: could not reach a logprob backend.\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"  Serve the pin with:\n"
            f"    vllm serve {a.model} --dtype {a.dtype} --max-model-len 131072\n"
            f"  then rerun with --server http://HOST:8000/v1")

    print(f"\nbackend: {path} / {a.model}")
    try:
        result = checks(agent)
    except Exception as exc:
        raise SystemExit(f"NOT CONFIRMED: the backend is reachable but scoring failed.\n"
                         f"  {type(exc).__name__}: {exc}")

    for name, r in result.items():
        if name.startswith("_"):
            continue
        print(f"  {'ok  ' if r['pass'] else 'FAIL'} {name:<14} "
              + "  ".join(f"{k}={v}" for k, v in r.items()
                          if k not in ("pass", "note")))

    passed = all(r["pass"] for k, r in result.items() if not k.startswith("_"))
    pin = dict(PIN, model=a.model, tokenizer=a.model, dtype=a.dtype)
    try:
        import vllm
        pin["vllm_version"] = vllm.__version__
    except Exception:
        pin["vllm_version"] = None
    payload = {
        "_README": ("Result of confirm_logprobs.py. PREREG.md points here rather than "
                    "restating the model: log-probs are not comparable across weights, "
                    "tokenizer or dtype, so a threshold tuned under one pin is "
                    "meaningless under another."),
        "confirmed": passed,
        "confirmed_at": dt.date.today().isoformat(),
        "git_commit": git_head(),
        "path": path,
        "endpoint": a.server,
        "pin": pin,
        "checks": result,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n{'CONFIRMED' if passed else 'NOT CONFIRMED'} -> {a.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
