"""Unit tests for the GMI -> DeepInfra provider-error fallback LM. No network:
the base ``dspy.LM.forward`` is monkeypatched to simulate each provider."""

import asyncio

import dspy
import litellm
import openai
import pytest

from src.program import lm_provider
from src.program.lm_provider import (
    DEEPINFRA_MODEL,
    GMI_API_BASE,
    GMI_MODEL,
    GMIWithDeepInfraFallbackLM,
    build_task_lm,
)

GMI_402_TEXT = (
    "OpenAIException - Error code: 402 - "
    "{'error': 'Insufficient balance', 'reason': 'model_access_denied'}"
)


def _api_error(status_code: int, text: str = GMI_402_TEXT) -> litellm.APIError:
    return litellm.APIError(
        status_code=status_code, message=text, llm_provider="openai", model=GMI_MODEL
    )


def _response(content: str = "hi") -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[{"index": 0, "finish_reason": "stop",
                  "message": {"role": "assistant", "content": content}}],
        model="deepseek-ai/DeepSeek-V4-Flash",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class FakeProviders:
    """Records every base-LM forward call and answers per model string."""

    def __init__(self, gmi, deepinfra):
        self.gmi = gmi
        self.deepinfra = deepinfra
        self.calls = []

    def __call__(self, lm, prompt=None, messages=None, **kwargs):
        self.calls.append((lm.model, prompt, messages, dict(kwargs)))
        behaviour = self.gmi if lm.model == GMI_MODEL else self.deepinfra
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeProviders as ``dspy.LM.forward``/``aforward``; configure via
    ``fake.gmi`` / ``fake.deepinfra`` (an exception instance or a response)."""
    fp = FakeProviders(gmi=_response("from-gmi"), deepinfra=_response("from-deepinfra"))

    def forward(lm, prompt=None, messages=None, **kwargs):  # plain function -> binds `lm`
        return fp(lm, prompt=prompt, messages=messages, **kwargs)

    async def aforward(lm, prompt=None, messages=None, **kwargs):
        return fp(lm, prompt=prompt, messages=messages, **kwargs)

    monkeypatch.setattr(dspy.LM, "forward", forward)

    monkeypatch.setattr(dspy.LM, "aforward", aforward)
    return fp


def test_construction_pins_both_routes_to_the_same_model(keys):
    lm = build_task_lm(cache=False)
    assert isinstance(lm, GMIWithDeepInfraFallbackLM)
    assert lm.model == GMI_MODEL
    assert lm.kwargs["api_base"] == GMI_API_BASE
    assert lm.kwargs["api_key"] == "test-gmi-key"
    assert lm.kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" in lm.kwargs["allowed_openai_params"]
    assert lm.cache is False

    fb = lm._fallback
    assert fb.model == DEEPINFRA_MODEL
    assert "api_key" not in fb.kwargs  # LiteLLM reads DEEPINFRA_API_KEY at call time
    assert "api_base" not in fb.kwargs
    assert fb.kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" in fb.kwargs["allowed_openai_params"]
    assert fb.cache is False  # overrides reach the fallback too


def test_missing_gmi_key_still_fails_at_construction(monkeypatch):
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        build_task_lm()


def test_gmi_402_reissues_the_identical_request_on_deepinfra(keys, fake, capsys):
    fake.gmi = _api_error(402)
    lm = build_task_lm(cache=False)
    messages = [{"role": "user", "content": "Say hi"}]

    result = lm.forward(messages=messages, temperature=0.3)

    assert result.choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]
    _, _, fb_messages, fb_kwargs = fake.calls[1]
    assert fb_messages == messages
    assert fb_kwargs == {"temperature": 0.3}

    err = capsys.readouterr().err
    assert "[WARNING] GMI error on openai/deepseek-ai/DeepSeek-V4-Flash (status=402" in err
    assert "Insufficient balance" in err
    assert f"retrying this call on DeepInfra ({DEEPINFRA_MODEL})" in err
    assert "[WARNING] DeepInfra fallback succeeded in" in err


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
def test_every_http_status_is_diverted(keys, fake, status):
    fake.gmi = _api_error(status, text=f"Error code: {status}")
    lm = build_task_lm()
    result = lm.forward(prompt="x")
    assert result.choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]


@pytest.mark.parametrize(
    "exc",
    [
        litellm.RateLimitError(message="rate limited", llm_provider="openai", model=GMI_MODEL),
        litellm.AuthenticationError(message="bad key", llm_provider="openai", model=GMI_MODEL),
        litellm.InternalServerError(message="500", llm_provider="openai", model=GMI_MODEL),
        litellm.ServiceUnavailableError(message="503", llm_provider="openai", model=GMI_MODEL),
        litellm.Timeout(message="timed out", llm_provider="openai", model=GMI_MODEL),
        litellm.APIConnectionError(message="conn reset", llm_provider="openai", model=GMI_MODEL),
        litellm.BadRequestError(message="generic 400", llm_provider="openai", model=GMI_MODEL),
        litellm.ContentPolicyViolationError(message="moderated", llm_provider="openai", model=GMI_MODEL),
        openai.APIConnectionError(request=None),  # raw openai error, no status_code
    ],
    ids=lambda e: type(e).__name__,
)
def test_every_api_error_kind_is_diverted(keys, fake, exc, capsys):
    fake.gmi = exc
    lm = build_task_lm()
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]
    assert "[WARNING] GMI error on" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exc",
    [
        litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model=GMI_MODEL),
        litellm.UnsupportedParamsError(status_code=400, message="bad param"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_request_is_the_problem_errors_are_not_diverted(keys, fake, exc, capsys):
    """Same request on the same model fails identically elsewhere -- surface it."""
    fake.gmi = exc
    lm = build_task_lm()
    with pytest.raises(type(exc)):
        lm.forward(prompt="x")
    assert [c[0] for c in fake.calls] == [GMI_MODEL]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("exc", [ValueError("dspy/program bug"), KeyError("x"), RuntimeError("no status")],
                         ids=lambda e: type(e).__name__)
def test_non_api_errors_are_not_diverted(keys, fake, exc, capsys):
    fake.gmi = exc
    lm = build_task_lm()
    with pytest.raises(type(exc)):
        lm.forward(prompt="x")
    assert [c[0] for c in fake.calls] == [GMI_MODEL]
    assert capsys.readouterr().err == ""


def test_fallback_failure_reraises_chained_from_the_gmi_error(keys, fake, capsys):
    fake.gmi = _api_error(402)
    fake.deepinfra = litellm.APIError(
        status_code=503, message="deepinfra down", llm_provider="deepinfra",
        model=DEEPINFRA_MODEL,
    )
    lm = build_task_lm()
    with pytest.raises(litellm.APIError) as info:
        lm.forward(prompt="x")
    assert info.value.status_code == 503
    assert isinstance(info.value.__cause__, litellm.APIError)
    assert info.value.__cause__.status_code == 402
    err = capsys.readouterr().err
    assert "[WARNING] GMI error" in err
    assert "[ERROR] DeepInfra fallback failed" in err
    assert "gmi_status=402, fallback_status=503" in err


def test_call_path_processes_the_fallback_response(keys, fake):
    """``lm(...)`` goes through dspy's response processing + history."""
    fake.gmi = _api_error(402)
    lm = build_task_lm(cache=False)
    out = lm("Say hi")
    assert out == ["from-deepinfra"]
    assert lm.history and lm.history[-1]["outputs"] == ["from-deepinfra"]


def test_async_path_falls_back_too(keys, fake, capsys):
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    result = asyncio.run(lm.aforward(prompt="x"))
    assert result.choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]
    assert "[WARNING] GMI error" in capsys.readouterr().err


def test_copy_keeps_the_fallback(keys):
    lm = build_task_lm(cache=False)
    clone = lm.copy(temperature=0.7)
    assert isinstance(clone, GMIWithDeepInfraFallbackLM)
    assert clone._fallback.model == DEEPINFRA_MODEL
    assert clone.kwargs["temperature"] == 0.7


def test_should_fallback_classifier():
    assert lm_provider.should_fallback(_api_error(402))
    assert lm_provider.should_fallback(_api_error(500))
    assert lm_provider.should_fallback(
        litellm.Timeout(message="t", llm_provider="openai", model=GMI_MODEL)
    )
    assert not lm_provider.should_fallback(
        litellm.ContextWindowExceededError(message="c", llm_provider="openai", model=GMI_MODEL)
    )
    assert not lm_provider.should_fallback(RuntimeError("no status"))
    assert lm_provider.NON_FALLBACK_ERRORS == (
        litellm.ContextWindowExceededError, litellm.UnsupportedParamsError,
    )
