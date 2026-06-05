"""Enable DSPy -> OpenTelemetry auto-instrumentation.

Importing this module patches DSPy so every Predict / ChainOfThought / ReAct /
Retrieve / LM call emits an OTel span on the global TracerProvider that the
CodeEvolver orchestrator installs. Additive only -- it attaches to whatever
provider already exists and never creates or replaces one.
"""
from openinference.instrumentation.dspy import DSPyInstrumentor

_INSTRUMENTED = False


def setup_dspy_tracing() -> None:
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    DSPyInstrumentor().instrument()
    _INSTRUMENTED = True


setup_dspy_tracing()
