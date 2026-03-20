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


class ExtractKnowledge(dspy.Signature):
    """After a ReAct reasoning run, extract structured entity-relationship facts discovered during the run. Focus on concrete facts about named entities, their attributes, and relationships that could help answer related sub-questions."""

    question: str = dspy.InputField()
    strategy: str = dspy.InputField()
    react_answer: list[str] = dspy.InputField()
    entity_facts: list[str] = dspy.OutputField(desc="structured list of discovered entity-relationship facts (e.g. 'Entity X has property Y', 'Entity A is related to Entity B via R')")


class EnrichQuestion(dspy.Signature):
    """Given a search strategy question and a set of already-discovered facts from prior reasoning runs, produce an enriched version of the question that incorporates the known facts as context. This helps subsequent ReAct runs build on intermediate results rather than starting from scratch."""

    question: str = dspy.InputField()
    accumulated_facts: list[str] = dspy.InputField(desc="structured entity-relationship facts discovered by previous ReAct runs")
    enriched_question: str = dspy.OutputField(desc="the original question augmented with known facts as context to guide the next ReAct run")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.strategy_generator = dspy.ChainOfThought(GenerateSearchStrategies)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)
        self.knowledge_extractor = dspy.ChainOfThought(ExtractKnowledge)
        self.question_enricher = dspy.ChainOfThought(EnrichQuestion)

    def forward(self, question):
        strategies_result = self.strategy_generator(question=question)
        strategies = strategies_result.strategies

        all_answers = []
        scratchpad: list[str] = []
        with dspy.context(rm=self.rm):
            for s in strategies:
                if scratchpad:
                    enriched = self.question_enricher(question=s, accumulated_facts=scratchpad)
                    query = enriched.enriched_question
                else:
                    query = s
                result = self.program(question=query)
                all_answers.extend(result.answer)
                extracted = self.knowledge_extractor(question=question, strategy=s, react_answer=result.answer)
                scratchpad.extend(extracted.entity_facts)

            if scratchpad:
                enriched_original = self.question_enricher(question=question, accumulated_facts=scratchpad)
                original_query = enriched_original.enriched_question
            else:
                original_query = question
            original_result = self.program(question=original_query)
            all_answers.extend(original_result.answer)

        merged = self.answer_merger(question=question, candidate_answers=all_answers)
        return dspy.Prediction(answer=merged.answer)
