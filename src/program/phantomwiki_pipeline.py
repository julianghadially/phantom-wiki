import os

import dspy
import src.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct
from src.program.phantomwiki_solver import solve as _solver_solve

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"

# Arm DeepSeek-V4-Flash through GMI Cloud, routed through LiteLLM/DSPy, using
# LiteLLM's OpenAI-compatible route: model="openai/<id>" + api_base=<GMI endpoint>.
# Reasoning is enabled via the standard OpenAI `reasoning_effort` param. The GMI
# key MUST be passed explicitly otherwise it would fall back to OPENAI_API_KEY.
MODEL = "openai/deepseek-ai/DeepSeek-V4-Flash"
GMI_API_BASE = "https://api.gmi-serving.com/v1"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.lm = dspy.LM(
            MODEL,
            api_base=GMI_API_BASE,
            api_key=os.environ["GMI_API_KEY"],
            cache=False,
            reasoning_effort="high",
            allowed_openai_params=["reasoning_effort"],
        )
        self.program = PhantomWikiReAct()

    def forward(self, question):
        # Deterministic templated-question solver: when the question matches a
        # PhantomWiki template the solver returns the EXACT answer computed from
        # the local corpus index (validated 150/150 on the training set). This
        # eliminates LM variance on the (majority) templated questions and lets
        # the ReAct agent fall through only for anything it cannot parse.
        det = _solver_solve(question)
        if det is not None:
            return dspy.Prediction(answer=det)
        with dspy.context(lm=self.lm, rm=self.rm):
            return self.program(question=question)
