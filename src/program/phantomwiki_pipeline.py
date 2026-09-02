import dspy
import src.infra.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.infra.lm_provider import build_task_lm
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        # The benchmark's pinned task model. Provider wiring lives in
        # src/infra/lm_provider.py; $LM_PROVIDER and $LM_FALLBACK repoint the
        # provider / name the cover / disarm the fallback.
        self.lm = build_task_lm(cache=False)
        self.program = PhantomWikiReAct()

    def forward(self, question):
        with dspy.context(lm=self.lm, rm=self.rm):
            return self.program(question=question)
