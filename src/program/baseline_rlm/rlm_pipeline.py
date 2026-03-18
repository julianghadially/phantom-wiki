import dspy
from src.program.counting_rm import CountingRM
from src.program.rlm.rlm_module import PhantomWikiRLM

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class RLMPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiRLM()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
