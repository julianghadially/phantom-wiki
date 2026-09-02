"""Unit tests for the cross-provider fallback task LM. No network: the base
``dspy.LM.forward`` is monkeypatched to simulate each provider."""

import asyncio

import dspy
import litellm
import openai
import pytest

from src.infra import lm_provider
from src.infra.lm_provider import (
    DEFAULT_FALLBACK,
    DEFAULT_PROVIDER,
    GMI_API_BASE,
    GMIWithDeepInfraFallbackLM,
    MODEL_ROUTES,
    PROVIDER_PREFERENCE,
    TASK_MODEL,
    ProviderBreaker,
    ProviderFallbackLM,
    Route,
    build_task_lm,
    route_for,
)

# Every model string comes from the one routing table -- the tests read it the
# same way the module does, so a model swap is still a one-row edit.
GMI_MODEL = MODEL_ROUTES[TASK_MODEL]["gmi"].model
DEEPINFRA_MODEL = MODEL_ROUTES[TASK_MODEL]["deepinfra"].model
DEEPSEEK_ROUTE = MODEL_ROUTES[TASK_MODEL]["deepseek"].model

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

    def __init__(self, gmi, deepinfra, deepseek=None):
        self.gmi = gmi
        self.deepinfra = deepinfra
        self.deepseek = deepseek
        self.calls = []

    def __call__(self, lm, prompt=None, messages=None, **kwargs):
        self.calls.append((lm.model, prompt, messages, dict(kwargs)))
        behaviour = {GMI_MODEL: self.gmi, DEEPINFRA_MODEL: self.deepinfra}.get(
            lm.model, self.deepseek
        )
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour


@pytest.fixture(autouse=True)
def fresh_breakers():
    """Provider health is process-global; no test may inherit another's."""
    lm_provider.reset_breakers()
    yield
    lm_provider.reset_breakers()


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    # The routing env vars must not leak in from the shell running pytest.
    for var in ("LM_MODEL", "LM_PROVIDER", "LM_FALLBACK", "LM_BREAKER",
                "LM_BREAKER_FAILURES", "LM_BREAKER_COOLDOWN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def gmi_primary(keys, monkeypatch):
    """The GMI arm: primary GMI, covered by DeepInfra.

    Pinned explicitly because the default primary is currently pinned past GMI
    (its API is down) and GMI's preferred cover is DeepSeek. The divert tests
    below are provider-agnostic; they are written against the GMI 402 this
    module was built for."""
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    monkeypatch.setenv("LM_FALLBACK", "deepinfra")


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeProviders as ``dspy.LM.forward``/``aforward``; configure via
    ``fake.gmi`` / ``fake.deepinfra`` (an exception instance or a response)."""
    fp = FakeProviders(
        gmi=_response("from-gmi"),
        deepinfra=_response("from-deepinfra"),
        deepseek=_response("from-deepseek"),
    )

    def forward(lm, prompt=None, messages=None, **kwargs):  # plain function -> binds `lm`
        return fp(lm, prompt=prompt, messages=messages, **kwargs)

    async def aforward(lm, prompt=None, messages=None, **kwargs):
        return fp(lm, prompt=prompt, messages=messages, **kwargs)

    monkeypatch.setattr(dspy.LM, "forward", forward)

    monkeypatch.setattr(dspy.LM, "aforward", aforward)
    return fp


def test_construction_pins_both_routes_to_the_same_model(gmi_primary):
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


def test_missing_gmi_key_still_fails_at_construction(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        build_task_lm()


def test_gmi_402_reissues_the_identical_request_on_deepinfra(gmi_primary, fake, capsys):
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
    assert "[WARNING] gmi error on openai/deepseek-ai/DeepSeek-V4-Flash (status=402" in err
    assert "Insufficient balance" in err
    assert f"retrying this call on deepinfra ({DEEPINFRA_MODEL})" in err
    assert "[WARNING] deepinfra fallback succeeded in" in err


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 502, 503, 504])
def test_every_http_status_is_diverted(gmi_primary, fake, status):
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
def test_every_api_error_kind_is_diverted(gmi_primary, fake, exc, capsys):
    fake.gmi = exc
    lm = build_task_lm()
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]
    assert "[WARNING] gmi error on" in capsys.readouterr().err


@pytest.mark.parametrize(
    "exc",
    [
        litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model=GMI_MODEL),
        litellm.UnsupportedParamsError(status_code=400, message="bad param"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_request_is_the_problem_errors_are_not_diverted(gmi_primary, fake, exc, capsys):
    """Same request on the same model fails identically elsewhere -- surface it."""
    fake.gmi = exc
    lm = build_task_lm()
    with pytest.raises(type(exc)):
        lm.forward(prompt="x")
    assert [c[0] for c in fake.calls] == [GMI_MODEL]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("exc", [ValueError("dspy/program bug"), KeyError("x"), RuntimeError("no status")],
                         ids=lambda e: type(e).__name__)
def test_non_api_errors_are_not_diverted(gmi_primary, fake, exc, capsys):
    fake.gmi = exc
    lm = build_task_lm()
    with pytest.raises(type(exc)):
        lm.forward(prompt="x")
    assert [c[0] for c in fake.calls] == [GMI_MODEL]
    assert capsys.readouterr().err == ""


def test_fallback_failure_reraises_chained_from_the_gmi_error(gmi_primary, fake, capsys):
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
    assert "[WARNING] gmi error" in err
    assert "[ERROR] deepinfra fallback failed" in err
    assert "primary_status=402, fallback_status=503" in err


def test_call_path_processes_the_fallback_response(gmi_primary, fake):
    """``lm(...)`` goes through dspy's response processing + history."""
    fake.gmi = _api_error(402)
    lm = build_task_lm(cache=False)
    out = lm("Say hi")
    assert out == ["from-deepinfra"]
    assert lm.history and lm.history[-1]["outputs"] == ["from-deepinfra"]


def test_async_path_falls_back_too(gmi_primary, fake, capsys):
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    result = asyncio.run(lm.aforward(prompt="x"))
    assert result.choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPINFRA_MODEL]
    assert "[WARNING] gmi error" in capsys.readouterr().err


def test_copy_keeps_the_fallback(gmi_primary):
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


# -- routing env vars: LM_PROVIDER / LM_FALLBACK ----------------------------


def test_default_routing_is_deepseek_primary_covered_by_deepinfra(keys):
    """GMI's API is down, so DEFAULT_PROVIDER starts one step down the order."""
    lm = build_task_lm()
    assert lm.provider == "deepseek"
    assert lm.model == DEEPSEEK_ROUTE
    assert lm.fallback_provider == "deepinfra"
    assert lm._fallback.model == DEEPINFRA_MODEL


def test_the_default_run_does_not_need_a_gmi_key(monkeypatch):
    """Neither the default primary nor its cover is GMI while GMI is down."""
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.delenv("LM_PROVIDER", raising=False)
    monkeypatch.delenv("LM_FALLBACK", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")
    lm = build_task_lm()
    assert lm.model == DEEPSEEK_ROUTE and lm._fallback.model == DEEPINFRA_MODEL


def test_the_preference_order_is_gmi_then_deepseek_then_deepinfra():
    """GMI is cheapest and fastest, so it leads once its API is healthy."""
    assert PROVIDER_PREFERENCE == ("gmi", "deepseek", "deepinfra")
    assert DEFAULT_PROVIDER in PROVIDER_PREFERENCE


def test_each_provider_is_covered_by_the_next_one_down_the_order():
    assert DEFAULT_FALLBACK == {
        "gmi": "deepseek", "deepseek": "deepinfra", "deepinfra": "gmi",
    }


def test_gmi_primary_is_covered_by_deepseek(keys, monkeypatch, fake, capsys):
    """The configured preference for when GMI comes back: GMI -> DeepSeek."""
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    assert lm.fallback_provider == "deepseek"
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepseek"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPSEEK_ROUTE]
    assert f"retrying this call on deepseek ({DEEPSEEK_ROUTE})" in capsys.readouterr().err


def test_lm_provider_deepinfra_makes_deepinfra_primary_and_gmi_the_fallback(keys, monkeypatch):
    """DeepInfra is last in the order, so its cover wraps around to GMI."""
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    lm = build_task_lm(cache=False)
    assert lm.provider == "deepinfra"
    assert lm.model == DEEPINFRA_MODEL
    assert "api_base" not in lm.kwargs
    assert lm.kwargs["reasoning_effort"] == "high"
    assert lm.fallback_provider == "gmi"
    assert lm._fallback.model == GMI_MODEL
    assert lm._fallback.kwargs["api_base"] == GMI_API_BASE
    assert lm._fallback.kwargs["api_key"] == "test-gmi-key"


def test_lm_provider_deepinfra_sends_the_request_to_deepinfra(keys, monkeypatch, fake):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    lm = build_task_lm()
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL]


def test_deepinfra_primary_falls_back_to_gmi(keys, monkeypatch, fake, capsys):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    fake.deepinfra = _api_error(503, text="deepinfra unavailable")
    lm = build_task_lm()
    assert lm.forward(prompt="x").choices[0].message.content == "from-gmi"
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL, GMI_MODEL]
    err = capsys.readouterr().err
    assert "[WARNING] deepinfra error on" in err
    assert f"retrying this call on gmi ({GMI_MODEL})" in err


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none", "FALSE", " 0 "])
def test_lm_fallback_off_builds_a_single_provider_lm(keys, monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    lm = build_task_lm()
    assert lm.provider == DEFAULT_PROVIDER
    assert lm._fallback is None
    assert lm.fallback_provider is None


def test_lm_fallback_off_lets_the_provider_error_propagate(gmi_primary, monkeypatch, fake, capsys):
    """The point of a single-provider run: the primary's failures stay visible."""
    monkeypatch.setenv("LM_FALLBACK", "0")
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    with pytest.raises(litellm.APIError) as info:
        lm.forward(prompt="x")
    assert info.value.status_code == 402
    assert [c[0] for c in fake.calls] == [GMI_MODEL]
    assert capsys.readouterr().err == ""


def test_lm_fallback_off_on_the_async_path_too(gmi_primary, monkeypatch, fake):
    monkeypatch.setenv("LM_FALLBACK", "0")
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    with pytest.raises(litellm.APIError):
        asyncio.run(lm.aforward(prompt="x"))
    assert [c[0] for c in fake.calls] == [GMI_MODEL]


def test_deepinfra_only_run_does_not_need_a_gmi_key(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = build_task_lm()
    assert lm.model == DEEPINFRA_MODEL and lm._fallback is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_lm_fallback_truthy_values_keep_it_armed(keys, monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    assert build_task_lm()._fallback is not None


def test_explicit_arguments_beat_the_environment(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = ProviderFallbackLM(provider="gmi", fallback=True)
    assert lm.provider == "gmi" and lm.fallback_provider == "deepseek"


def test_unknown_routing_values_fail_loudly(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "togetherai")
    with pytest.raises(ValueError, match="LM_PROVIDER"):
        build_task_lm()
    monkeypatch.delenv("LM_PROVIDER")
    monkeypatch.setenv("LM_FALLBACK", "maybe")
    with pytest.raises(ValueError, match="LM_FALLBACK"):
        build_task_lm()


def test_copy_keeps_the_routing(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    monkeypatch.setenv("LM_FALLBACK", "0")
    clone = build_task_lm(cache=False).copy(temperature=0.7)
    assert isinstance(clone, ProviderFallbackLM)
    assert clone.model == DEEPINFRA_MODEL and clone._fallback is None


# -- deepseek: the model's first-party API as a third provider --------------


def test_deepseek_is_a_first_class_provider(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    lm = build_task_lm(cache=False)
    assert lm.provider == "deepseek"
    assert lm.model == DEEPSEEK_ROUTE
    assert "api_key" not in lm.kwargs  # LiteLLM reads DEEPSEEK_API_KEY at call time
    assert "api_base" not in lm.kwargs
    assert lm.kwargs["reasoning_effort"] == "high"


def test_deepseek_sends_the_request_to_deepseek(keys, monkeypatch, fake):
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    lm = build_task_lm()
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepseek"
    assert [c[0] for c in fake.calls] == [DEEPSEEK_ROUTE]


def test_deepseek_route_is_the_confirmed_first_party_id(keys, monkeypatch):
    """Listed by GET https://api.deepseek.com/models -- not a guessed alias."""
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    assert build_task_lm().model == "deepseek/deepseek-v4-flash"


def test_deepseek_defaults_to_covering_with_deepinfra(keys, monkeypatch, fake, capsys):
    """DeepSeek is today's primary; the next provider down the order covers."""
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    fake.deepseek = _api_error(500, text="deepseek down")
    lm = build_task_lm()
    assert lm.fallback_provider == "deepinfra"
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [DEEPSEEK_ROUTE, DEEPINFRA_MODEL]
    assert "[WARNING] deepseek error on" in capsys.readouterr().err


def test_lm_fallback_can_name_the_covering_provider(keys, monkeypatch, fake):
    """With three providers, 'the other one' is ambiguous -- allow naming it."""
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    monkeypatch.setenv("LM_FALLBACK", "deepseek")
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    assert lm.provider == "gmi" and lm.fallback_provider == "deepseek"
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepseek"
    assert [c[0] for c in fake.calls] == [GMI_MODEL, DEEPSEEK_ROUTE]


def test_a_provider_cannot_cover_for_itself(keys, monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    monkeypatch.setenv("LM_FALLBACK", "deepinfra")
    with pytest.raises(ValueError, match="cannot be its own fallback"):
        build_task_lm()


def test_default_fallback_pairs_are_declared_for_every_provider():
    from src.infra.lm_provider import PROVIDERS

    assert set(DEFAULT_FALLBACK) == set(PROVIDERS)
    assert all(v in PROVIDERS and v != k for k, v in DEFAULT_FALLBACK.items())


def test_deepseek_only_run_needs_neither_gmi_nor_deepinfra(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = build_task_lm()
    assert lm.model == DEEPSEEK_ROUTE and lm._fallback is None


# -- the routing table: one row per model, one column per provider ----------


def test_every_preferred_provider_has_a_route_for_the_task_model():
    row = MODEL_ROUTES[TASK_MODEL]
    assert set(PROVIDER_PREFERENCE) <= set(row)
    assert all(isinstance(r, Route) for r in row.values())


@pytest.mark.parametrize("provider", PROVIDER_PREFERENCE)
def test_each_provider_serves_the_model_string_the_table_gives_it(keys, monkeypatch, provider):
    """The table is the only place a model id lives -- the LM just reads it."""
    monkeypatch.setenv("LM_PROVIDER", provider)
    monkeypatch.setenv("LM_FALLBACK", "0")
    assert build_task_lm().model == MODEL_ROUTES[TASK_MODEL][provider].model


def test_only_the_openai_style_route_carries_a_base_url_and_an_explicit_key(keys):
    gmi = route_for(TASK_MODEL, "gmi").lm_kwargs()
    assert gmi["api_base"] == GMI_API_BASE and gmi["api_key"] == "test-gmi-key"
    for native in ("deepseek", "deepinfra"):
        kwargs = route_for(TASK_MODEL, native).lm_kwargs()
        # LiteLLM knows these endpoints and reads their keys at call time, so
        # nothing lands in dump_state or the trace files.
        assert set(kwargs) == {"model"}


def test_serving_a_different_model_is_a_one_row_edit(keys, monkeypatch):
    """The whole point of the table: add a row, point $LM_MODEL at it, done."""
    monkeypatch.setitem(MODEL_ROUTES, "toy-1", {
        "gmi": Route("openai/vendor/Toy-1", api_base=GMI_API_BASE, api_key_env="GMI_API_KEY"),
        "deepseek": Route("deepseek/toy-1"),
        "deepinfra": Route("deepinfra/vendor/Toy-1"),
    })
    monkeypatch.setenv("LM_MODEL", "toy-1")
    lm = build_task_lm()
    assert lm.task_model == "toy-1"
    assert lm.model == "deepseek/toy-1"          # primary: deepseek
    assert lm._fallback.model == "deepinfra/vendor/Toy-1"


def test_an_unknown_model_names_the_rows_that_exist(keys, monkeypatch):
    monkeypatch.setenv("LM_MODEL", "gpt-9")
    with pytest.raises(ValueError, match="MODEL_ROUTES"):
        build_task_lm()


def test_a_provider_missing_from_the_row_fails_at_construction(keys, monkeypatch):
    """Better than discovering it on the first divert, mid-eval."""
    monkeypatch.setitem(MODEL_ROUTES, "toy-2", {"deepseek": Route("deepseek/toy-2")})
    monkeypatch.setenv("LM_MODEL", "toy-2")
    with pytest.raises(ValueError, match="no route for model 'toy-2'"):
        build_task_lm()  # deepseek primary is fine; its deepinfra cover is not


# -- retry budget -----------------------------------------------------------


def test_the_primary_gets_one_retry_and_the_cover_the_full_budget(keys):
    """Two attempts, then the cover -- the third attempt's backoff is dead time
    when an equivalent provider is idle."""
    lm = build_task_lm()
    assert lm.num_retries == 1
    assert lm._fallback.num_retries == 3


def test_a_single_provider_run_keeps_the_full_retry_budget(keys, monkeypatch):
    """With nowhere to divert, retrying is the only thing that can save the row."""
    monkeypatch.setenv("LM_FALLBACK", "0")
    assert build_task_lm().num_retries == 3


def test_an_explicit_num_retries_wins_on_both_routes(keys):
    lm = build_task_lm(num_retries=7)
    assert lm.num_retries == 7 and lm._fallback.num_retries == 7


# -- circuit breaker: the state machine -------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def breaker(clock, monkeypatch):
    """A breaker on a fake clock, installed as the shared one for gmi."""
    b = ProviderBreaker("gmi", threshold=3, cooldown=60.0, clock=clock)
    monkeypatch.setitem(lm_provider._BREAKERS, "gmi", b)
    return b


def test_a_breaker_starts_closed_and_ignores_isolated_failures(breaker):
    assert breaker.state == "closed" and breaker.allow()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()  # never 3 IN A ROW


def test_consecutive_failures_open_the_breaker(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()


def test_an_open_breaker_lets_exactly_one_probe_through_after_the_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(59)
    assert not breaker.allow()
    clock.advance(2)
    assert breaker.allow()          # the probe
    assert breaker.state == "probing"
    assert not breaker.allow()      # everyone else keeps going to the cover


def test_a_failed_probe_restarts_the_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(61)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(59)
    assert not breaker.allow()
    clock.advance(2)
    assert breaker.allow()


def test_a_successful_probe_closes_the_breaker(breaker, clock, capsys):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(61)
    breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()
    assert "marking it healthy again" in capsys.readouterr().err


# -- circuit breaker: through the LM ----------------------------------------


def _fail_primary(lm, fake, times):
    """Drive ``times`` diverted primary failures through the LM."""
    for _ in range(times):
        lm.forward(prompt="x")
    return [c[0] for c in fake.calls]


def test_an_outage_stops_costing_primary_attempts_once_the_breaker_opens(
    gmi_primary, fake, breaker, capsys
):
    fake.gmi = _api_error(503, text="gmi down")
    lm = build_task_lm()
    _fail_primary(lm, fake, 3)                       # 3 diverted calls
    assert [c[0] for c in fake.calls].count(GMI_MODEL) == 3

    fake.calls.clear()
    for _ in range(10):
        assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    # Not one attempt on the dead provider: no request, no retries, no backoff.
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL] * 10
    err = capsys.readouterr().err
    assert "gmi failed 3 calls in a row -- marking it unhealthy" in err
    # ... and it says so once, not once per call.
    assert err.count("marking it unhealthy") == 1


def test_a_stray_error_never_opens_the_breaker(gmi_primary, fake, breaker):
    """One bad call in a healthy run must not divert the other 800."""
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    lm.forward(prompt="x")                 # diverted
    fake.gmi = _response("from-gmi")
    for _ in range(3):
        assert lm.forward(prompt="x").choices[0].message.content == "from-gmi"
    assert breaker.state == "closed"


def test_the_primary_comes_back_by_itself_after_the_cooldown(
    gmi_primary, fake, breaker, clock, capsys
):
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    _fail_primary(lm, fake, 3)
    fake.gmi = _response("from-gmi")       # outage over, nobody told us

    fake.calls.clear()
    assert lm.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL]

    clock.advance(61)
    fake.calls.clear()
    assert lm.forward(prompt="x").choices[0].message.content == "from-gmi"  # probe
    assert lm.forward(prompt="x").choices[0].message.content == "from-gmi"  # closed
    assert [c[0] for c in fake.calls] == [GMI_MODEL, GMI_MODEL]
    assert "probing it with one call" in capsys.readouterr().err


def test_the_breaker_is_shared_by_every_copy_of_the_lm(gmi_primary, fake, breaker):
    """dspy hands out lm.copy() deepcopies; provider health is not per-object
    (and a threading.Lock would not survive the deepcopy)."""
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    _fail_primary(lm, fake, 3)

    clone = lm.copy(temperature=0.7)
    fake.calls.clear()
    assert clone.forward(prompt="x").choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL]
    assert clone.breaker is breaker


def test_non_provider_errors_never_count_toward_the_breaker(gmi_primary, fake, breaker):
    """A program fault says nothing about the provider's health."""
    fake.gmi = litellm.ContextWindowExceededError(
        message="too long", llm_provider="openai", model=GMI_MODEL
    )
    lm = build_task_lm()
    for _ in range(4):
        with pytest.raises(litellm.ContextWindowExceededError):
            lm.forward(prompt="x")
    assert breaker.state == "closed"


def test_the_async_path_skips_an_unhealthy_primary_too(gmi_primary, fake, breaker):
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    _fail_primary(lm, fake, 3)
    fake.calls.clear()
    result = asyncio.run(lm.aforward(prompt="x"))
    assert result.choices[0].message.content == "from-deepinfra"
    assert [c[0] for c in fake.calls] == [DEEPINFRA_MODEL]


@pytest.mark.parametrize("value", ["0", "false", "off", "none"])
def test_lm_breaker_off_retries_the_dead_primary_on_every_call(
    gmi_primary, monkeypatch, fake, value
):
    """The pre-breaker behaviour, for diagnosing a provider rather than routing
    around it."""
    monkeypatch.setenv("LM_BREAKER", value)
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    assert lm.breaker is None
    _fail_primary(lm, fake, 5)
    assert [c[0] for c in fake.calls].count(GMI_MODEL) == 5


def test_the_breaker_is_inert_without_a_cover(keys, monkeypatch):
    """Skipping the primary is meaningless when there is nowhere to skip to."""
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = build_task_lm()
    assert lm.breaker_enabled is False and lm.breaker is None


def test_breaker_thresholds_come_from_the_environment(keys, monkeypatch):
    monkeypatch.setenv("LM_BREAKER_FAILURES", "7")
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "12.5")
    b = lm_provider.breaker_for("deepseek")
    assert b.threshold == 7 and b.cooldown == 12.5


def test_breaker_for_returns_one_instance_per_provider():
    assert lm_provider.breaker_for("gmi") is lm_provider.breaker_for("gmi")
    assert lm_provider.breaker_for("gmi") is not lm_provider.breaker_for("deepinfra")
