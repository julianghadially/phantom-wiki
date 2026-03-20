import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActHighK(dspy.Module):
    """ReAct reasoning module with increased retrieval breadth (k=50) to improve recall for questions with many correct answers."""

    def __init__(self):
        self.retrieve = dspy.Retrieve(k=50)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)


class GenerateSearchStrategies(dspy.Signature):
    """Analyze the question and generate diverse, independent search strategies to find ALL valid answers. For questions asking about multiple entities, generate strategies that explore different starting points in the knowledge graph."""

    question: str = dspy.InputField()
    strategies: list[str] = dspy.OutputField(desc="list of 3 diverse search strategies/rephrased questions that together ensure exhaustive answer coverage")


class MergeAnswers(dspy.Signature):
    """Given a question and multiple answer sets discovered through different search strategies, merge and deduplicate all valid answers into a single comprehensive list."""

    question: str = dspy.InputField()
    candidate_answers: list[str] = dspy.InputField(desc="all candidate answers collected across multiple search strategies, may contain duplicates")
    answer: list[str] = dspy.OutputField(desc="deduplicated list of all valid, distinct answers to the question")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReActHighK()
        self.strategy_generator = dspy.ChainOfThought(GenerateSearchStrategies)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)

    def forward(self, question):
        strategies_result = self.strategy_generator(question=question)
        strategies = strategies_result.strategies

        all_answers = []
        with dspy.context(rm=self.rm):
            for s in strategies:
                result = self.program(question=s)
                all_answers.extend(result.answer)
            original_result = self.program(question=question)
            all_answers.extend(original_result.answer)

        merged = self.answer_merger(question=question, candidate_answers=all_answers)
        return dspy.Prediction(answer=merged.answer)
