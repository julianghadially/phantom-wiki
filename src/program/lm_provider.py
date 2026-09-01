"""Task LM for the benchmark programs: DeepSeek-V4-Flash on GMI Cloud, with a
per-call DeepInfra fallback on provider errors.

Why this exists
---------------
GMI intermittently answers a single request with a 4xx -- observed as
``Error code: 402 - {'error': 'Insufficient balance', 'reason':
'model_access_denied'}`` -- and then serves the very next request normally.
LiteLLM retries 408/409/429/5xx but treats every other 4xx as permanent, so the
call fails, the row scores 0.0 and the eval aggregate is silently depressed.

``GMIWithDeepInfraFallbackLM`` keeps GMI as the primary (fast) provider and,
when the GMI request fails with a provider/API error, re-issues the SAME
request -- same model, same messages, same kwargs, same ``reasoning_effort`` --
on DeepInfra, which serves the identical model (much slower, but correct).
Every diverted error is logged to stderr with a ``[WARNING]`` tag, and the
fallback outcome is stamped on the current OpenTelemetry span so it is visible
in the trace files.

What is diverted (see ``should_fallback``)
------------------------------------------
Everything caught here comes from the LM request path -- never from program
code -- so the question is only whether a second provider could plausibly
answer the same request. It is diverted when the error is an API error
(``openai.APIError`` is the base of every LiteLLM exception): any 4xx
(401/402/403/404/422/429, generic 400), any 5xx (``InternalServerError``,
``ServiceUnavailableError``, ``APIError``), ``Timeout`` and
``APIConnectionError`` -- all AFTER LiteLLM's own retries for the transient
ones have run out. Provider-specific rejections such as content moderation are
diverted too.

It is NOT diverted when the request itself is the problem and the same model
elsewhere would fail identically -- ``ContextWindowExceededError`` (the prompt
does not fit the model) and ``UnsupportedParamsError`` (an invalid request
parameter, raised before any network call). Those are the program's own
faults and must stay visible to whoever is evolving it. Non-API exceptions
(a ``ValueError`` from DSPy, a program bug) are never diverted either.

This module is deliberately outside the evolvable program packages: the
benchmark constraints pin the task model, and the provider wiring is
infrastructure, not part of the program under optimization.

Keys: the GMI key is passed explicitly (otherwise LiteLLM's ``openai/`` route
would fall back to ``OPENAI_API_KEY``); the DeepInfra key is read by LiteLLM
from ``DEEPINFRA_API_KEY`` at call time so it never lands in ``dump_state`` or
the trace files.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import dspy
import litellm
import openai

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
# Primary: GMI Cloud through LiteLLM's OpenAI-compatible route
# (model="openai/<id>" + api_base=<GMI endpoint>).
GMI_MODEL = f"openai/{MODEL_ID}"
GMI_API_BASE = "https://api.gmi-serving.com/v1"
# Fallback: the same model on DeepInfra through LiteLLM's native provider.
DEEPINFRA_MODEL = f"deepinfra/{MODEL_ID}"
# Reasoning is enabled via the standard OpenAI `reasoning_effort` param. Neither
# route allows it by default, so it must be forwarded via
# allowed_openai_params=[...] (BerriAI/litellm#14039). Do NOT use
# thinking={"type": "enabled"}: DeepInfra's endpoint rejects the kwarg.
REASONING_EFFORT = "high"
_ALLOWED_OPENAI_PARAMS = ["reasoning_effort"]

_LOG_ERROR_CHARS = 300


# Request-is-the-problem errors: the identical request on the identical model
# fails the same way on DeepInfra, so diverting only hides a program fault.
NON_FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    litellm.ContextWindowExceededError,
    litellm.UnsupportedParamsError,
)


def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by a LiteLLM/OpenAI exception, or None."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def should_fallback(exc: BaseException) -> bool:
    """True for a provider/API error a second provider could plausibly answer.

    ``openai.APIError`` is the common base of every LiteLLM exception
    (4xx, 5xx, timeouts, connection errors); anything else is not an API
    error and is left alone. ``NON_FALLBACK_ERRORS`` carves out the
    request-is-the-problem cases.
    """
    if not isinstance(exc, openai.APIError):
        return False
    return not isinstance(exc, NON_FALLBACK_ERRORS)


def _describe(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _LOG_ERROR_CHARS:
        text = text[:_LOG_ERROR_CHARS] + "..."
    return f"{type(exc).__name__}: {text}"


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr, flush=True)


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def _mark_span(**attrs: Any) -> None:
    """Best-effort: stamp the fallback on the active OTel span (never raises)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        for key, value in attrs.items():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 -- tracing must never break a call
        pass


class GMIWithDeepInfraFallbackLM(dspy.LM):
    """DeepSeek-V4-Flash on GMI; a provider error from GMI re-issues the call on DeepInfra.

    Hooks ``forward``/``aforward`` (the request layer) so ``dspy.LM.__call__``'s
    response processing, history and usage tracking -- and the OpenInference
    ``LM.__call__`` span -- stay exactly as for a plain ``dspy.LM``.
    """

    def __init__(self, **overrides: Any):
        super().__init__(
            GMI_MODEL,
            api_base=GMI_API_BASE,
            api_key=os.environ["GMI_API_KEY"],
            reasoning_effort=REASONING_EFFORT,
            allowed_openai_params=list(_ALLOWED_OPENAI_PARAMS),
            **overrides,
        )
        self._fallback = dspy.LM(
            DEEPINFRA_MODEL,
            reasoning_effort=REASONING_EFFORT,
            allowed_openai_params=list(_ALLOWED_OPENAI_PARAMS),
            **overrides,
        )

    # -- sync ---------------------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):
        try:
            return super().forward(prompt=prompt, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- classified below
            if not should_fallback(exc):
                raise
            code = _status_code(exc)
            started = self._log_diverted(code, exc)
            try:
                result = self._fallback.forward(prompt=prompt, messages=messages, **kwargs)
            except Exception as exc2:  # noqa: BLE001
                self._log_fallback_failed(code, exc2, started)
                raise exc2 from exc
            self._log_fallback_ok(code, started)
            return result

    # -- async --------------------------------------------------------------

    async def aforward(self, prompt=None, messages=None, **kwargs):
        try:
            return await super().aforward(prompt=prompt, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- classified below
            if not should_fallback(exc):
                raise
            code = _status_code(exc)
            started = self._log_diverted(code, exc)
            try:
                result = await self._fallback.aforward(
                    prompt=prompt, messages=messages, **kwargs
                )
            except Exception as exc2:  # noqa: BLE001
                self._log_fallback_failed(code, exc2, started)
                raise exc2 from exc
            self._log_fallback_ok(code, started)
            return result

    # -- logging ------------------------------------------------------------

    def _log_diverted(self, code: int | None, exc: BaseException) -> float:
        desc = _describe(exc)
        _warn(
            f"GMI error on {self.model} (status={code}, {desc}) "
            f"-- retrying this call on DeepInfra ({self._fallback.model})"
        )
        _mark_span(
            **{
                "lm.fallback.provider": "deepinfra",
                "lm.fallback.model": self._fallback.model,
                "lm.fallback.gmi_status": code if code is not None else -1,
                "lm.fallback.gmi_error": desc,
            }
        )
        return time.monotonic()

    def _log_fallback_ok(self, code: int | None, started: float) -> None:
        elapsed = time.monotonic() - started
        _warn(f"DeepInfra fallback succeeded in {elapsed:.1f}s (gmi_status={code})")
        _mark_span(**{"lm.fallback.outcome": "ok", "lm.fallback.seconds": elapsed})

    def _log_fallback_failed(
        self, code: int | None, exc: BaseException, started: float
    ) -> None:
        elapsed = time.monotonic() - started
        desc = _describe(exc)
        _error(
            f"DeepInfra fallback failed after {elapsed:.1f}s "
            f"(gmi_status={code}, fallback_status={_status_code(exc)}): {desc}"
        )
        _mark_span(
            **{
                "lm.fallback.outcome": "failed",
                "lm.fallback.seconds": elapsed,
                "lm.fallback.error": desc,
            }
        )


def build_task_lm(**overrides: Any) -> dspy.LM:
    """The benchmark's task LM. ``overrides`` (e.g. ``cache=False``) apply to
    both the GMI primary and the DeepInfra fallback."""
    return GMIWithDeepInfraFallbackLM(**overrides)
