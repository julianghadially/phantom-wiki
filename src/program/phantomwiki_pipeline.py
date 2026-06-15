import dspy
import src.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        # gpt-5.4-nano is a reasoning model but runs with reasoning disabled by
        # default (reasoning_effort defaults to "none" -> 0 reasoning tokens).
        # DSPy also fails to auto-detect it as a reasoning model because the
        # version dot ("5.4") breaks its gpt-5 regex, so we configure it
        # explicitly. "low" is the only non-trivial effort this nano model
        # supports (it rejects "minimal"; "medium"/"high" are not honored).
        self.lm = dspy.LM("openai/gpt-5.4-nano", cache=False, reasoning_effort="low")
        self.program = PhantomWikiReAct()

    def forward(self, question):
        with dspy.context(lm=self.lm, rm=self.rm):
            return self.program(question=question)
