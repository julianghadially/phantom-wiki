import dspy
from src.program.counting_rm import CountingRM
from src.program.baseline_rag.program_multihop_rag import PhantomWikiMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class BaselineRAGPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiMultiHop()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
