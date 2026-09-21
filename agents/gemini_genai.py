"""Gemini backend on the current google-genai SDK.

Replaces the legacy google.generativeai client in agents/gemini.py. The reason
this file exists rather than a patch to that one: the old batch path hardcoded
`temperature=None` into batch_generate, so every batched call silently ran at
the agent default of 0 no matter what the call site asked for. Propagation and
rejuvenation both went through that path, which is why particles were frozen.

Two invariants here:
  1. temperature and max_tokens are forwarded on BOTH the single and batch paths.
  2. every call records the temperature it actually sent via trace_log, so the
     log reports reality rather than the intent at the call site.
"""
import asyncio
import os
import time
from types import SimpleNamespace
from typing import List

import backoff
import httpx

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# google-genai speaks httpx, not requests. Backing off on
# requests.exceptions.HTTPError (as the legacy client did) catches nothing here:
# a transient `RemoteProtocolError: Server disconnected` killed a 44-step run.
RETRYABLE = (
    httpx.HTTPError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectError,
    genai_errors.APIError,
)

from .base import AsyncBaseAgent

try:
    from trace_log import record_llm_call
except Exception:  # instrumentation is optional; the agent still works without it
    def record_llm_call(**kwargs):
        pass


SAFETY_SETTINGS = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


class AsyncGeminiGenAIAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        if not os.environ.get("GOOGLE_API_KEY"):
            try:  # repo keeps GOOGLE_API_KEY in a gitignored .env
                from dotenv import load_dotenv
                load_dotenv()
            except Exception:
                pass
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        # None leaves the model default (thinking on); 0 disables it.
        self.thinking_budget = getattr(self.args, "thinking_budget", 0)

    # -- the only place temperature is resolved -----------------------------
    def _resolve(self, temperature, max_tokens):
        t = self.args.temperature if temperature is None else temperature
        m = self.args.max_tokens if max_tokens is None else max_tokens
        return t, m

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, stage=None, attempt=0):
        t, m = self._resolve(temperature, max_tokens)
        # Thinking tokens count against max_output_tokens. On a 2.5 model with
        # the upstream max_tokens=512, reasoning consumed 488 of the budget and
        # the visible answer was truncated at 20 tokens -- before the "Answer:"
        # the parser needs. Every verdict came back a parse failure and scored
        # 0.001, which the loose matcher then sometimes read as bucket 'a' off a
        # prose fragment. Default thinking off so max_tokens means what the
        # upstream prompts assume: budget for the visible chain of thought they
        # explicitly ask for.
        cfg = dict(
            system_instruction=system_prompt,
            temperature=t,
            max_output_tokens=m,
            safety_settings=SAFETY_SETTINGS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        config = types.GenerateContentConfig(**cfg)
        last = None
        for retry in range(4):
            try:
                resp = self.client.models.generate_content(
                    model=self.args.model, contents=prompt, config=config
                )
                self._note_finish(resp, t, m, stage, attempt + retry)
                return resp
            except RETRYABLE as exc:
                last = exc
                time.sleep(min(2 ** retry, 8))
        raise last

    def _note_finish(self, resp, t, m, stage, attempt):
        """A silent MAX_TOKENS is exactly the class of failure Gate 0 exists to catch."""
        try:
            fr = str(resp.candidates[0].finish_reason)
            u = resp.usage_metadata
            record_llm_call(temperature=t, max_tokens=m, model=self.args.model,
                            attempt=attempt, stage=stage, finish_reason=fr,
                            truncated=fr.endswith("MAX_TOKENS"),
                            thoughts_tokens=getattr(u, "thoughts_token_count", None),
                            visible_tokens=getattr(u, "candidates_token_count", None))
            if fr.endswith("MAX_TOKENS"):
                import trace_log as _tl
                _stage = stage or _tl.RECORDER._stage
                print(f"[gemini_genai] TRUNCATED at max_tokens={m} (stage={_stage}); "
                      f"visible={getattr(u,'candidates_token_count',None)} "
                      f"thoughts={getattr(u,'thoughts_token_count',None)}")
        except Exception:
            pass

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None, stage=None):
        message = self.preprocess_chat(prompt, history=history) if history else self.preprocess_input(prompt)
        output = self.generate(message, system_prompt, temperature=temperature, max_tokens=max_tokens, stage=stage)
        return self.postprocess_output(output)

    @backoff.on_exception(backoff.expo, RETRYABLE, max_tries=6, max_time=180)
    async def batch_generate(self, prompts, system_prompts, temperature=None, max_tokens=None, stage=None):
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*[
            loop.run_in_executor(self.executor, self.generate, prompt, system_prompt, temperature, max_tokens, stage)
            for prompt, system_prompt in zip(prompts, system_prompts)
        ])

    def batch_interact(self, prompts, temperature=0, max_tokens=256, system_prompts=None, histories: List[List] = None, stage=None):
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        else:
            assert len(prompts) == len(system_prompts)

        if histories is not None and histories[0] is not None:
            assert len(prompts) == len(histories)
            batch = [self.preprocess_chat(p, history=h) for p, h in zip(prompts, histories)]
        else:
            batch = [self.preprocess_input(p) for p in prompts]

        # temperature and max_tokens forwarded -- this is the line the legacy
        # client got wrong.
        outputs = asyncio.run(
            self.batch_generate(batch, system_prompts, temperature=temperature, max_tokens=max_tokens, stage=stage)
        )
        return [self.postprocess_output(o) for o in outputs]

    def preprocess_input(self, text, system_prompt=None, history=None):
        return text

    def preprocess_chat(self, text, system_prompt=None, history=None):
        messages = []
        if history:
            for idx, msg in enumerate(history):
                role = "user" if idx % 2 == 0 else "model"
                messages.append(types.Content(role=role, parts=[types.Part(text=str(msg))]))
        messages.append(types.Content(role="user", parts=[types.Part(text=str(text))]))
        return messages

    def postprocess_output(self, output):
        text = getattr(output, "text", None)
        if text is None:
            # a blocked or empty candidate: surface it rather than crashing the run
            reason = getattr(getattr(output, "prompt_feedback", None), "block_reason", None)
            print(f"[gemini_genai] empty response (block_reason={reason})")
            return ""
        return text
