"""Backend dispatch.

Imports are lazy: a missing SDK for one backend must not stop another from
loading. Previously every backend was imported at module scope, so running the
Gemini path required the OpenAI, Together and vLLM packages to be installed.
"""


def _missing(name, exc):
    def _raise(*args, **kwargs):
        raise ImportError(f"backend '{name}' is unavailable: {exc}")
    return _raise


def _gpt():
    from .gpt import AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent, O1BaseAgent, O1MiniAgent
    return AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent, O1BaseAgent, O1MiniAgent


def load_model(model_name, num_gpus=2, mode="async", **kwargs):
    if model_name in ["o1-preview-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"]:
        _, _, O1BaseAgent, _ = _gpt()
        model = O1BaseAgent({'model': model_name, **kwargs})
    elif model_name in ["o1-mini-2024-09-12"]:
        _, _, _, O1MiniAgent = _gpt()
        model = O1MiniAgent({'model': model_name, **kwargs})
    elif model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-0125-preview", "gpt-4-0613", "gpt-3.5-turbo-0125"]:
        AsyncGPT, SyncGPT, _, _ = _gpt()
        if mode == "async":
            model = AsyncGPT({'model': model_name, **kwargs})
        elif mode == "non-async":
            model = SyncGPT({'model': model_name, **kwargs})
    elif model_name in ["gpt-4-turbo-nonasync", "gpt-4o-nonasync"]:
        _, SyncGPT, _, _ = _gpt()
        model = SyncGPT({'model': model_name.removesuffix("-nonasync"), **kwargs})
    elif model_name.startswith("gemini-") and model_name.endswith("-legacy"):
        from .gemini import AsyncGeminiAgent
        model = AsyncGeminiAgent({'model': model_name.removesuffix("-legacy"), **kwargs})
    elif model_name.startswith("gemini-"):
        from .gemini_genai import AsyncGeminiGenAIAgent
        model = AsyncGeminiGenAIAgent({'model': model_name, **kwargs})
    elif model_name in ["meta-llama/Llama-3-70b-chat-hf-tg", "meta-llama/Llama-3-8b-chat-hf-tg", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
        # was AsyncLlama3LogProbAgent, which is never imported anywhere -> NameError
        from .together_ai import AsyncLlama3Agent
        model = AsyncLlama3Agent({'model': model_name, 'temperature': 0, 'max_tokens': 1024, **kwargs})
    elif model_name in ["meta-llama/Llama-2-13b-chat-hf", "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B-Instruct"]:
        from .vllm import VllmAgent
        model = VllmAgent(model_name, num_gpus=num_gpus, **kwargs)
    elif model_name in ["Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen/QwQ-32B-Preview"]:
        from .together_ai import AsyncQwenAgent
        model = AsyncQwenAgent({'model': model_name, 'temperature': 0, 'max_tokens': 1024, **kwargs})
    elif model_name in ["deepseek-ai/DeepSeek-R1"]:
        from .together_ai import AsyncDeepSeekAgent
        model = AsyncDeepSeekAgent({'model': model_name, 'temperature': 0, 'max_tokens': 5000, **kwargs})
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model
