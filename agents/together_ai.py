import os
import together
import backoff
import requests
from types import SimpleNamespace
from .base import AsyncBaseAgent
from transformers import AutoTokenizer
from together import Together

client = Together(api_key=os.getenv('TOGETHERAI_API_KEY'))

class AsyncTogetherAIAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.api_key = together.api_key = os.getenv('TOGETHERAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.args.model = self.args.model.removesuffix("-tg")

    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError, together.error.ResponseError))
    def generate(self, prompt, temperature=None, max_tokens=None):
        output = together.Complete.create(
            prompt=prompt,
            model=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature,
        )

        return output

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
        try:
            responses = [c['text'].strip() for c in output['output']['choices']]
        except:
            responses = [c['text'].strip() for c in output['choices']]
        return responses[0]

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
        message = self.preprocess_input(prompt, system_prompt, history)
        output = self.generate(message, temperature=temperature, max_tokens=max_tokens)
        response = self.postprocess_output(output)

        return response

class AsyncLlama3Agent(AsyncTogetherAIAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        hf_togetherai_map = {
            "meta-llama/Llama-3-70b-chat-hf": "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Llama-3-8b-chat-hf": "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": "meta-llama/Llama-3.3-70B-Instruct",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(hf_togetherai_map[self.args.model])

    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError, together.error.ResponseError))
    def generate(self, prompt, temperature=None, max_tokens=None):
        output = together.Complete.create(
            prompt=prompt,
            model=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature,
            stop=[self.tokenizer.eos_token, "<|eot_id|>"]
        )

        return output

class AsyncQwenAgent(AsyncTogetherAIAgent):
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
        # prompt = self.tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        return messages

    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError, together.error.ResponseError, together.error.ServiceUnavailableError))
    def generate(self, messages, temperature=None, max_tokens=None):
        response = client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            temperature=self.args.temperature if temperature is None else temperature,
            max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
        )
        return response.choices[0].message.content

    def postprocess_output(self, output):
        return output

class AsyncDeepSeekAgent(AsyncQwenAgent):
    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError, together.error.ResponseError, together.error.ServiceUnavailableError, together.error.APIError, together.error.InvalidRequestError))
    def generate(self, messages, temperature=None, max_tokens=None):
        response = client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            temperature=self.args.temperature if temperature is None else temperature
        )
        return response.choices[0].message.content