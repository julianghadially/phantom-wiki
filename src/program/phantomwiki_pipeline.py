import os

import dspy
import src.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

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
        # Pre-warm the corpus graph so the exact solver is ready before any
        # worker thread starts (avoids a lazy-build race under the thread pool).
        from src.program.graph_solver import warmup_graph
        warmup_graph()

    def forward(self, question):
        with dspy.context(lm=self.lm, rm=self.rm):
            return self.program(question=question)
