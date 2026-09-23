"""vLLM backends: generation, and log-probabilities of a given continuation.

The vllm import is guarded for the same reason load_model's are: vllm pulls in
torch and expects a GPU, and this module now also holds VllmServerAgent, which
talks to a server and needs neither. An unguarded top-level import made the
server client unimportable on exactly the machines it exists for. Nothing about
VllmAgent changes -- it still fails at construction when vllm is absent, only
with a sentence saying so instead of an ImportError on the module.
"""
from typing import List

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:                     # pragma: no cover - env-dependent
    LLM = SamplingParams = None
    # Bound to a module-level string: Python deletes the `except ... as` name at
    # the end of the block, so a closure over it raises NameError at the moment
    # it is needed -- reporting a missing name instead of a missing package.
    _VLLM_IMPORT_ERROR = str(exc)

    def _need_vllm(*_args, **_kwargs):
        raise ImportError(
            "this path needs the `vllm` package (and a GPU to serve weights on). "
            "For log-probabilities against a model served elsewhere, use "
            f"VllmServerAgent, which needs neither. Original error: {_VLLM_IMPORT_ERROR}")

class VllmAgent():
    def __init__(self, model_name, num_gpus=2, max_tokens=1024, **kwargs):
        if LLM is None:
            _need_vllm()
        self.model_name = model_name
        self.model = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 1
        self.cot_prompt = "\nLet's think step by step."
        
    def preprocess_input(self, text, system_prompt=None, history=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": f"{system_prompt}"})
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{text}"})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return prompt

    def postprocess_output(self, output):
        return output.outputs[0].text.strip()

    def interact(self, prompt, temperature=1, max_tokens=None, system_prompt: str = None, history: str = None):
        return self.batch_interact([prompt], temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, histories=[history])[0]

    def batch_interact(self, prompts, temperature=1, max_tokens=None, system_prompt: str = None, histories: List[List] = None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        if histories is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, system_prompt, history) for prompt, history in zip(prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt, system_prompt) for prompt in prompts]

        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens)
        # prompts = [self.preprocess_input(text) for text in texts]
        outputs = self.model.generate(message_batch, sampling_params=sampling_params)
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def batch_cot(self, prompts, temperature=None, max_tokens=None):
        cot_prompts = [prompt.removesuffix("\nAnswer:") + self.cot_prompt for prompt in prompts]
        cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
        return cot_responses

    # def cot(self, prompt, temperature=None, max_tokens=None):
    #     q_prompt = prompt.split("\nAnswer:")[0].strip()
    #     cot_prompt = f"{q_prompt}\nLet's think step by step before answering the question above."
    #     cot_response = self.interact(cot_prompt, temperature, max_tokens)
    #     prompt_with_cot = f"{q_prompt}\n{cot_response}\nTherefore, the answer is:"
    #     final_response = self.interact(prompt_with_cot, temperature, max_tokens)
    #     return final_response

    # def batch_cot(self, prompts, temperature=None, max_tokens=None):
    #     cot_prompts = [prompt.split("\nAnswer:")[0].strip() + "\nLet's think step by step before answering the question above." for prompt in prompts]
    #     cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
    #     prompts_with_cot = [prompt.split("\nAnswer:")[0].strip() + f"\n{cot_response}\nTherefore, the answer is:" for prompt, cot_response in zip(prompts, cot_responses)]
    #     return self.batch_interact(prompts_with_cot, temperature, max_tokens)

class NemoAgent(VllmAgent):
    def __init__(self, model_name, num_gpus=4, max_tokens=1024, **kwargs):
        if LLM is None:
            _need_vllm()
        self.model_name = model_name
        self.model = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95, max_model_len=416810)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 0
        self.cot_prompt = "\nLet's think step by step."

    def interact(self, prompt, temperature=0, max_tokens=None, system_prompt=None, history=None):
        return super().interact(prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, history=history)

    def batch_interact(self, prompts, temperature=0, max_tokens=256, system_prompt: str = None, histories: List[List] = None):
        return super().batch_interact(prompts, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, histories=histories)
    

# ==========================================================================
# Continuation scoring
#
# Phase 2 needs the log-probability a model assigns to a continuation IT DID
# NOT GENERATE -- the observed next action, scored under each hypothesis --
# which is a different request from every other backend in this directory.
# Nothing here changes the generation path above: the tracer's existing calls
# go through interact/batch_interact byte for byte, and these methods are only
# reachable from code that asks for them by name.
#
# THE PITFALL THIS CODE EXISTS TO AVOID. The obvious implementation tokenizes
# the prompt, tokenizes the continuation, and reads the last len(continuation)
# log-probs. That is wrong whenever the tokenizer merges across the boundary --
# "...size" + " definition" can retokenize so that the first continuation token
# absorbs the prompt's last character, and the log-probs then line up against
# the wrong tokens for the whole tail. So the join is tokenized ONCE and the
# split point is found by prefix agreement, and a boundary that does not agree
# is reported rather than silently scored.
#
# WHY A SERVER CLIENT SITS BESIDE THE OFFLINE ONE. vllm imports torch and wants
# a GPU at construction time. Most of this repo's work happens where neither is
# available, and a backend that can only be exercised on the box that has the
# weights cannot be checked by the people writing against it. VllmServerAgent
# speaks the same two methods over an OpenAI-compatible endpoint.
# ==========================================================================

import os
from typing import Optional


class ContinuationScore:
    """Per-token log-probabilities for a continuation, and their provenance.

    `tokens` and `logprobs` are the CONTINUATION's, in order. `boundary_clean`
    records whether the join retokenized: False means the numbers describe a
    different tokenization than the caller asked about, and a caller that
    ignores it is scoring noise.
    """

    __slots__ = ("tokens", "token_ids", "logprobs", "boundary_clean",
                 "prompt_tokens", "model", "detail")

    def __init__(self, tokens, token_ids, logprobs, boundary_clean,
                 prompt_tokens, model, detail=None):
        self.tokens = tokens
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.boundary_clean = boundary_clean
        self.prompt_tokens = prompt_tokens
        self.model = model
        self.detail = detail or {}

    @property
    def total(self):
        return sum(self.logprobs)

    @property
    def mean(self):
        """Length-normalised. Two continuations of different length are not
        comparable on `total`, and the scorer compares exactly that."""
        return self.total / len(self.logprobs) if self.logprobs else float("nan")

    def as_dict(self):
        return {"model": self.model, "n_tokens": len(self.logprobs),
                "prompt_tokens": self.prompt_tokens, "total": self.total,
                "mean": self.mean, "boundary_clean": self.boundary_clean,
                "tokens": self.tokens, "logprobs": self.logprobs,
                "detail": self.detail}


def _split_point(tokenizer, prompt, continuation):
    """Tokenize the join once; return (joined_ids, split_index, boundary_clean).

    The split index is where the prompt's own tokenization stops agreeing with
    the join's. When they agree all the way it is len(prompt_ids), which is the
    clean case; when they do not, the index is the last agreeing position and
    boundary_clean is False so the caller can refuse the number.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    joined_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    n = 0
    for a, b in zip(prompt_ids, joined_ids):
        if a != b:
            break
        n += 1
    return prompt_ids, joined_ids, n, n == len(prompt_ids)


class _ScoringMixin:
    def score_continuation(self, prompt: str, continuation: str) -> ContinuationScore:
        return self.batch_score_continuation([(prompt, continuation)])[0]


class VllmScoringAgent(VllmAgent, _ScoringMixin):
    """VllmAgent plus prompt_logprobs. Generation behaviour is unchanged.

    It takes its own dtype and max_model_len because VllmAgent swallows both in
    **kwargs and never reaches LLM() with them -- which would have silently
    served the pinned model at the default dtype, and log-probs are not
    comparable across dtypes.
    """

    def __init__(self, model_name, num_gpus=1, max_tokens=1024,
                 dtype="bfloat16", max_model_len=None, gpu_memory_utilization=0.90,
                 **kwargs):
        if LLM is None:
            _need_vllm()
        self.model_name = model_name
        opts = dict(model=model_name, tensor_parallel_size=num_gpus, dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization)
        if max_model_len:
            opts["max_model_len"] = max_model_len
        self.model = LLM(**opts)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 0
        self.cot_prompt = "\nLet's think step by step."
        self.dtype = dtype

    def batch_score_continuation(self, pairs):
        joins, splits = [], []
        for prompt, continuation in pairs:
            _, joined_ids, n, clean = _split_point(self.tokenizer, prompt, continuation)
            joins.append(joined_ids)
            splits.append((n, clean, len(joined_ids)))

        # prompt_logprobs=0 asks for the log-prob of the token that is actually
        # there and nothing else. max_tokens=1 because vLLM will not accept a
        # request that generates nothing; the sampled token is discarded.
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
        outs = self.model.generate(prompt_token_ids=joins, sampling_params=params)

        results = []
        for (prompt, continuation), out, (n, clean, total) in zip(pairs, outs, splits):
            plp = out.prompt_logprobs or []
            ids = out.prompt_token_ids
            tokens, token_ids, lps = [], [], []
            for i in range(n, total):
                entry = plp[i] if i < len(plp) else None
                if not entry:
                    continue
                tid = ids[i]
                lp = entry.get(tid)
                if lp is None:
                    continue
                lps.append(float(getattr(lp, "logprob", lp)))
                token_ids.append(tid)
                tokens.append(getattr(lp, "decoded_token", None)
                              or self.tokenizer.decode([tid]))
            results.append(ContinuationScore(
                tokens, token_ids, lps, clean, n, self.model_name,
                {"joined_tokens": total, "requested_continuation": continuation}))
        return results


class VllmServerAgent(_ScoringMixin):
    """The same two methods against an OpenAI-compatible vLLM server.

    No torch, no GPU, no weights in this process -- which is the only way the
    scoring path can be exercised from a machine that is not the one serving
    the model. The tokenizer still has to match the served weights exactly, so
    it is loaded from the same model string and the server's own token ids are
    checked against it on the first call.
    """

    def __init__(self, model_name, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 120.0, **kwargs):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL")
                         or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _post(self, path, body):
        import json as _json
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as fh:
            return _json.loads(fh.read().decode("utf-8"))

    def batch_score_continuation(self, pairs):
        results = []
        for prompt, continuation in pairs:
            _, joined_ids, n, clean = _split_point(self.tokenizer, prompt, continuation)
            # Token ids rather than text: the server would retokenize a string
            # and the split index computed here would no longer point at the
            # same place. echo/max_tokens=0 asks for the prompt's own log-probs.
            body = {"model": self.model_name, "prompt": joined_ids,
                    "max_tokens": 0, "echo": True, "logprobs": 0,
                    "temperature": 0}
            out = self._post("/completions", body)
            lp = (out["choices"][0].get("logprobs") or {})
            token_lps = lp.get("token_logprobs") or []
            toks = lp.get("tokens") or []
            tokens, lps = [], []
            for i in range(n, len(joined_ids)):
                if i >= len(token_lps) or token_lps[i] is None:
                    continue
                lps.append(float(token_lps[i]))
                tokens.append(toks[i] if i < len(toks)
                              else self.tokenizer.decode([joined_ids[i]]))
            results.append(ContinuationScore(
                tokens, joined_ids[n:], lps, clean, n, self.model_name,
                {"endpoint": self.base_url, "joined_tokens": len(joined_ids),
                 "requested_continuation": continuation}))
        return results
