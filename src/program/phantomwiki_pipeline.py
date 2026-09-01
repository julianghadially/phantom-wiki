import dspy
import src.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.program.counting_rm import CountingRM
from src.program.lm_provider import build_task_lm
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        # DeepSeek-V4-Flash (reasoning_effort="high") on GMI Cloud, with a
        # per-call fallback to the same model on DeepInfra when GMI answers a
        # 4xx. Provider wiring lives in src/program/lm_provider.py.
        self.lm = build_task_lm(cache=False)
        self.program = PhantomWikiReAct()

    def forward(self, question):
        with dspy.context(lm=self.lm, rm=self.rm):
            return self.program(question=question)
