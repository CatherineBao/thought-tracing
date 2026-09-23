"""One cached, batched model handle for the choice-point harness.

WHY A CACHE AND NOT JUST A MODEL. This harness makes the same call many times:
extraction reruns while the schema is still moving, a scoring bug means
re-running the forecast leg, and a spot-check means re-reading what the model
already said. Paying for those again is the reason people stop re-running
things, and a measurement nobody re-runs is a measurement nobody checks. Keyed
on the exact (model, temperature, system, prompt) so a changed prompt is a
different entry and never a stale hit.

WHY THE KEY INCLUDES TEMPERATURE. Everything here runs at 0, but a cache that
ignored temperature would serve a greedy answer to a sampled call, which is the
kind of silent contamination that makes a control arm stop being one.

THE FAKE IS NOT A MOCK OF THE ANSWER. FakeModel exists so the pipeline -- prompt
construction, parsing, keying, scoring, the permutation test -- can be exercised
with no key and no spend. Its answers are a hash. Nothing it produces is
evidence about anything, and every report built on it says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "musing_out", "choice_cache")


def _key(model, system, prompt, temperature, max_tokens):
    blob = json.dumps([model, system or "", prompt, temperature, max_tokens],
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedModel:
    """batch_interact with a disk cache and a call counter.

    The counter is reported by every driver so a run's cost is on the record
    beside its result -- `llm_calls` in the output is calls actually MADE, not
    calls requested, and the gap is what the cache saved.
    """

    def __init__(self, model_name="gemini-2.5-flash", cache_dir=CACHE_DIR,
                 read_only=False, verbose=True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.read_only = read_only
        self.verbose = verbose
        self.calls_made = 0
        self.cache_hits = 0
        self._model = None
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    # -- the model is built lazily so a fully cached run needs no API key ----

    def _ensure(self):
        if self._model is None:
            try:
                import dotenv
                dotenv.load_dotenv(os.path.join(HERE, ".env"))
            except Exception:
                pass
            from agents.load_model import load_model
            self._model = load_model(self.model_name)
        return self._model

    def _path(self, k):
        return os.path.join(self.cache_dir, k[:2], k + ".json")

    def _read(self, k):
        p = self._path(k)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)["response"]
        except Exception:
            return None

    def _write(self, k, prompt, system, response):
        p = self._path(k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"model": self.model_name, "system": system,
                       "prompt": prompt, "response": response,
                       "at": time.time()}, fh, ensure_ascii=False)
        os.replace(tmp, p)

    def batch_interact(self, prompts, system_prompts=None, temperature=0,
                       max_tokens=1024, chunk=24):
        systems = system_prompts or [None] * len(prompts)
        keys = [_key(self.model_name, s, p, temperature, max_tokens)
                for s, p in zip(systems, prompts)]
        out = [self._read(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        self.cache_hits += len(prompts) - len(todo)

        if todo and self.read_only:
            raise RuntimeError(f"{len(todo)} uncached prompts and read_only=True")

        for start in range(0, len(todo), chunk):
            idxs = todo[start:start + chunk]
            model = self._ensure()
            if self.verbose:
                print(f"    llm {start + len(idxs)}/{len(todo)}", end="\r",
                      file=sys.stderr, flush=True)
            got = model.batch_interact([prompts[i] for i in idxs],
                                       system_prompts=[systems[i] for i in idxs],
                                       temperature=temperature,
                                       max_tokens=max_tokens)
            self.calls_made += len(idxs)
            for i, resp in zip(idxs, got):
                out[i] = resp
                self._write(keys[i], prompts[i], systems[i], resp)
        if todo and self.verbose:
            print(" " * 40, end="\r", file=sys.stderr)
        return out

    def interact(self, prompt, system_prompt=None, temperature=0, max_tokens=1024):
        return self.batch_interact([prompt], [system_prompt], temperature, max_tokens)[0]

    def stats(self):
        return {"model": self.model_name, "llm_calls": self.calls_made,
                "cache_hits": self.cache_hits}


class FakeModel:
    """Deterministic stand-in. Exercises the pipeline; proves nothing about it.

    Answers are a hash of the prompt, so every number downstream is chance --
    which is exactly what the offline tests need to assert against, because a
    harness whose baseline and hypothesis arms BOTH sit at chance under a hash
    is a harness with no leak in it.
    """

    def __init__(self, mode="hash"):
        self.mode = mode
        self.calls_made = 0
        self.cache_hits = 0
        self.model_name = f"fake:{mode}"

    def batch_interact(self, prompts, system_prompts=None, temperature=0,
                       max_tokens=1024, chunk=24):
        self.calls_made += len(prompts)
        return [self._answer(p) for p in prompts]

    def interact(self, prompt, system_prompt=None, temperature=0, max_tokens=1024):
        return self.batch_interact([prompt])[0]

    # Turn lines look like "[12] Wolf: ...". The fake has to read them to
    # produce an extraction that survives validate(), or the offline tests
    # exercise the reject path and nothing else -- which is what the first
    # version of this class did.
    _TURN = re.compile(r"^\[(\d+)\]\s+([^:]+):", re.M)
    _ACTIONS = ("CONCEDE", "HOLD", "ESCALATE", "DROP", "IGNORE", "RE_RAISE", "TRADE")

    def _answer(self, prompt):
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        opts = re.findall(r"^\s*\[([A-Z_]+)\]", prompt, flags=re.M)
        if opts:
            return f"ANSWER: {opts[h % len(opts)]}\nWHY: deterministic stand-in, not evidence."
        if "decision point" in prompt:
            return json.dumps(self._fake_points(prompt, h))
        if "MOTIVE" in prompt or "motive" in prompt:
            return json.dumps([{"motive": f"fake aim {h % 7}",
                                "because": "deterministic stand-in"}])
        return json.dumps({"note": "fake", "h": h})

    def _fake_points(self, prompt, h):
        turns = self._TURN.findall(prompt)
        if len(turns) < 4:
            return []
        out = []
        for k in (len(turns) // 2, len(turns) - 2):
            idx, who = turns[k]
            acts = [self._ACTIONS[(h + k + j) % len(self._ACTIONS)] for j in range(3)]
            acts = list(dict.fromkeys(acts))
            while len(acts) < 3:
                acts.append(next(a for a in self._ACTIONS if a not in acts))
            out.append({
                "person": who.strip(), "at_turn": int(idx),
                "situation": "deterministic stand-in situation",
                "why_real": "stand-in",
                "alternatives": [{"action": a, "gist": "stand-in", "gives_up": "stand-in"}
                                 for a in acts],
                "actual": acts[h % len(acts)],
            })
        return out

    def stats(self):
        return {"model": self.model_name, "llm_calls": self.calls_made,
                "cache_hits": 0, "WARNING": "FakeModel output is a hash, not evidence"}


def extract_json(raw):
    """First JSON object or array in a model reply, or None.

    Models fence JSON, prepend a sentence, or both. Parsing the whole reply
    fails on all of that; a regex for the outermost braces fails on nested
    ones. This walks the brackets, which handles both and returns None rather
    than raising -- an unparseable reply is a datum (it is counted) and not a
    crash halfway through a paid run.
    """
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    # WHICHEVER BRACKET COMES FIRST. Trying "{" before "[" finds the first
    # object INSIDE an array and returns it alone, so a caller expecting a list
    # of extractions silently receives one dict and reports the reply as
    # unparseable. That is what it did.
    openers = sorted((("{", "}"), ("[", "]")),
                     key=lambda oc: (s.find(oc[0]) if oc[0] in s else len(s) + 1))
    for opener, closer in openers:
        start = s.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None
