import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class FollowUpInvestigation(dspy.Signature):
    """You are given a question and a list of answers already found via one investigation pass.
    Explore alternative relationship chains and paths NOT yet investigated to find additional answers.
    Treat already_found as a non-exhaustive partial result — there may be more valid answers
    reachable via different paths, relationships, or entity traversals that were not explored before."""

    question: str = dspy.InputField()
    already_found: list[str] = dspy.InputField(desc="answers discovered so far; treat as partial/non-exhaustive")
    answer: list[str] = dspy.OutputField(desc="additional answers found via unexplored paths")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.retrieve = dspy.Retrieve(k=7)
        self.followup_react = dspy.ReAct(
            signature=FollowUpInvestigation,
            tools=[self._search_wiki],
            max_iters=25,
        )

    def _search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            result1 = self.program(question=question)
            result2 = self.followup_react(question=question, already_found=result1.answer)
            combined = list(dict.fromkeys(result1.answer + [a for a in result2.answer if a not in result1.answer]))
            return dspy.Prediction(answer=combined)
