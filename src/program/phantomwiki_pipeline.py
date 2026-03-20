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


class ExtractEntitiesFromPassages(dspy.Signature):
    """Extract every entity name from the retrieved passages that is a valid answer to the question. Be completely exhaustive — include every person or entity name mentioned in any passage that satisfies the question criteria, even if you are uncertain."""

    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="retrieved passages from the knowledge base")
    entities: list[str] = dspy.OutputField(desc="exhaustive list of all entity/person names found in the passages that are valid answers to the question")


class GenerateExpansionQueries(dspy.Signature):
    """Given a question and the entities found so far, generate new diverse search queries designed to find additional entities that answer the question but have NOT yet been discovered. Target different regions of the knowledge graph."""

    question: str = dspy.InputField()
    found_so_far: list[str] = dspy.InputField(desc="entities already found — generate queries likely to surface DIFFERENT, not-yet-found entities")
    queries: list[str] = dspy.OutputField(desc="list of 5 diverse search queries targeting unexplored entity populations that answer the question")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.strategy_generator = dspy.ChainOfThought(GenerateSearchStrategies)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)
        self.entity_extractor = dspy.ChainOfThought(ExtractEntitiesFromPassages)
        self.query_expander = dspy.ChainOfThought(GenerateExpansionQueries)

    def forward(self, question):
        # Phase 1: Multi-strategy ReAct fan-out (existing approach)
        strategies_result = self.strategy_generator(question=question)
        strategies = strategies_result.strategies

        all_answers = []
        with dspy.context(rm=self.rm):
            for s in strategies:
                result = self.program(question=s)
                all_answers.extend(result.answer)
            original_result = self.program(question=question)
            all_answers.extend(original_result.answer)

        # Phase 2: Iterative broad-retrieval expansion to catch entities the ReAct
        # agents missed by stopping early. For each round, generate new queries
        # seeded by what has been found so far, retrieve passages with higher k
        # directly (bypassing ReAct's early-stopping heuristic), extract all
        # matching entities, and continue until no new entities are discovered.
        found = set(all_answers)
        broad_retriever = dspy.Retrieve(k=30)
        MAX_EXPANSION_ROUNDS = 3

        with dspy.context(rm=self.rm):
            for _ in range(MAX_EXPANSION_ROUNDS):
                expansion = self.query_expander(
                    question=question, found_so_far=list(found)
                )
                newly_found = set()
                for q in expansion.queries:
                    passages = broad_retriever(q)
                    passage_text = "\n\n".join(passages.passages)
                    extracted = self.entity_extractor(
                        question=question, passages=passage_text
                    )
                    newly_found.update(extracted.entities)
                novel = newly_found - found
                if not novel:
                    break
                found.update(novel)

        merged = self.answer_merger(question=question, candidate_answers=list(found))
        return dspy.Prediction(answer=merged.answer)
