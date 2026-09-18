"""
Thin, provider-agnostic LLM wrapper.

The team has not locked an LLM vendor yet, so the agent depends on this
interface instead of on any SDK. Switching provider = changing one env var.

Supported values of ``LLM_PROVIDER``:

  openai      — any OpenAI-compatible endpoint (OpenAI, Groq, Together,
                NVIDIA NIM, OpenRouter, local vLLM). Set LLM_BASE_URL too.
  gemini      — Google Generative AI.
  rule        — no network: deterministic stub used by unit tests, CI and the
                offline Docker build. The agent degrades gracefully (rule-based
                routing + templated answers) instead of crashing.
"""

from __future__ import annotations

import os
import time

try:  # make .env loading independent of import order — get_llm() must not
      # silently fall back to "rule" just because src.utils.config wasn't
      # imported first in whatever script called it.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 800) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI  # lazy import

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, model: str, api_key: str):
        self.model_name = model
        self.api_key = api_key
        try:
            from google import genai  # new official SDK

            self._client = genai.Client(api_key=api_key)
            self._use_new_sdk = True
        except ImportError:
            import google.generativeai as genai  # legacy SDK fallback

            genai.configure(api_key=api_key)
            self._genai = genai
            self._use_new_sdk = False

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        # Gemini 3.x / 2.x "thinking" models spend part of max_output_tokens on an
        # internal reasoning step before writing the visible answer. A small
        # budget (the router call uses ~150) can be entirely consumed by that
        # reasoning step, leaving response.text with nothing to return, and
        # accessing .text in that state raises instead of returning "".
        # A hard floor keeps every call — router, SQL generation, answer —
        # from silently crashing the agent on a truncated response.
        effective_max_tokens = max(max_tokens, 512)

        if getattr(self, "_use_new_sdk", False):
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=effective_max_tokens,
            )
            if system:
                config.system_instruction = system

            for attempt in range(3):
                try:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    return (response.text or "").strip()
                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if attempt < 2:
                            time.sleep(6.5)
                            continue
                    try:
                        return (getattr(response, "text", "") or "").strip()
                    except Exception:
                        pass
                    print(f"[llm] Gemini returned no usable text ({exc}); "
                          "treating as empty so the caller's fallback applies.")
                    return ""

        model = self._genai.GenerativeModel(
            self.model_name, system_instruction=system
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": effective_max_tokens,
            },
        )
        try:
            return (response.text or "").strip()
        except (ValueError, AttributeError) as exc:
            pieces: list[str] = []
            for candidate in getattr(response, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        pieces.append(text)
            if pieces:
                return "".join(pieces).strip()
            print(f"[llm] Gemini returned no usable text ({exc}); "
                  "treating as empty so the caller's fallback applies.")
            return ""


class RuleBasedLLM(BaseLLM):
    """No-network stub. Returns empty text so callers use their fallbacks."""

    name = "rule"

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        return ""


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseLLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "rule")).lower()
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    base_url = base_url or os.getenv("LLM_BASE_URL") or None

    if provider in {"rule", "none", "offline"} or not api_key:
        if provider not in {"rule", "none", "offline"}:
            print("[llm] No LLM_API_KEY found — falling back to rule-based mode.")
        return RuleBasedLLM()

    try:
        if provider == "gemini":
            return GeminiLLM(model, api_key)
        return OpenAICompatibleLLM(model, api_key, base_url)
    except Exception as exc:
        print(f"[llm] Could not initialise provider '{provider}' ({exc}); using rule mode.")
        return RuleBasedLLM()


def timed_complete(llm: BaseLLM, *args, **kwargs) -> tuple[str, float]:
    start = time.perf_counter()
    text = llm.complete(*args, **kwargs)
    return text, round((time.perf_counter() - start) * 1000, 2)