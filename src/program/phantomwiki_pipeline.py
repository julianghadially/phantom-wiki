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


class DecomposeAggregationQuestion(dspy.Signature):
    """Analyze whether a question requires counting or aggregating a property independently for each of several qualifying entities (e.g., 'how many friends does each person born in 1990 have?' — one count per person). If so, decompose into an enumeration query and a per-entity count query template."""

    question: str = dspy.InputField()
    is_aggregation: bool = dspy.OutputField(desc="True if the question asks to compute a count or aggregate for EACH qualifying entity separately (indicated by words like 'each', 'every', 'for all', or when the answer is a set of counts — one per entity). False for simple lookup questions.")
    enumeration_query: str = dspy.OutputField(desc="If is_aggregation is True: a natural-language query that will retrieve ALL qualifying entities (e.g., 'Who was born in 1990?' or 'List all people whose occupation is interpreter.'). Empty string if is_aggregation is False.")
    count_query_template: str = dspy.OutputField(desc="If is_aggregation is True: a query template containing the literal placeholder {name} that, when filled in, asks for the aggregate value for that one entity (e.g., 'How many friends does {name} have?' or 'How many hobbies does {name} have?'). Empty string if is_aggregation is False.")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.strategy_generator = dspy.ChainOfThought(GenerateSearchStrategies)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)
        self.decomposer = dspy.ChainOfThought(DecomposeAggregationQuestion)

    def forward(self, question):
        decomp = self.decomposer(question=question)

        if decomp.is_aggregation and decomp.enumeration_query and decomp.count_query_template:
            # --- Aggregation path ---
            # Step 1: enumerate ALL qualifying entities
            with dspy.context(rm=self.rm):
                enum_result = self.program(question=decomp.enumeration_query)
            entities = enum_result.answer  # list of entity names

            # Step 2: for each entity, run the per-entity count query and collect results
            all_counts = []
            with dspy.context(rm=self.rm):
                for entity in entities[:25]:  # cap to limit total API calls
                    count_query = decomp.count_query_template.replace("{name}", entity)
                    count_result = self.program(question=count_query)
                    all_counts.extend(count_result.answer)

            # Step 3: merge / deduplicate
            merged = self.answer_merger(question=question, candidate_answers=all_counts)
            return dspy.Prediction(answer=merged.answer)

        else:
            # --- Standard multi-strategy path ---
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
