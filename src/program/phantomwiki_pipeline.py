import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class GenerateSearchStrategies(dspy.Signature):
    """Analyze the question and generate diverse, independent search strategies to find ALL valid answers. For questions asking about multiple entities, generate strategies that explore different starting points in the knowledge graph."""

    question: str = dspy.InputField()
    strategies: list[str] = dspy.OutputField(desc="list of 3 diverse search strategies/rephrased questions that together ensure exhaustive answer coverage")


class MergeAnswers(dspy.Signature):
    """Given a question and multiple answer sets discovered through different search strategies, merge and deduplicate all valid answers into a single comprehensive list."""

    question: str = dspy.InputField()
    candidate_answers: list[str] = dspy.InputField(desc="all candidate answers collected across multiple search strategies, may contain duplicates")
    answer: list[str] = dspy.OutputField(desc="deduplicated list of all valid, distinct answers to the question")


class DecomposeToSteps(dspy.Signature):
    """Decompose a complex multi-hop question into an ordered list of simpler sub-questions that must be answered sequentially, where each step can build on answers from prior steps."""

    question: str = dspy.InputField()
    steps: list[str] = dspy.OutputField(desc="ordered list of simpler sub-questions; answers to earlier steps serve as context for later steps")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.strategy_generator = dspy.ChainOfThought(GenerateSearchStrategies)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)
        self.decomposer = dspy.ChainOfThought(DecomposeToSteps)

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

            decomposed = self.decomposer(question=question)
            steps = decomposed.steps
            context = ""
            for step in steps:
                if context:
                    contextualized_step = f"{step} (Prior answers: {context})"
                else:
                    contextualized_step = step
                step_result = self.program(question=contextualized_step)
                all_answers.extend(step_result.answer)
                context = ", ".join(step_result.answer) if step_result.answer else context

        merged = self.answer_merger(question=question, candidate_answers=all_answers)
        return dspy.Prediction(answer=merged.answer)
