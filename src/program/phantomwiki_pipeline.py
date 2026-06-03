import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self, model_name = "openai/gpt-5-mini"):
        super().__init__()
        self.lm = dspy.LM(model_name, cache=False)
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()

    def forward(self, question):
        with dspy.context(rm=self.rm, lm=self.lm):
            return self.program(question=question)
