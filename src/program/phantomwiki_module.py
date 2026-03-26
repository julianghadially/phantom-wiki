import dspy


class QuestionDecomposer(dspy.Signature):
    """Analyze a question and decompose it into a structured search plan.
    Identify the question type, the anchor entities to search for first, and
    a step-by-step plan for retrieving all relevant information."""

    question: str = dspy.InputField()
    question_type: str = dspy.OutputField(
        desc='one of "enumeration", "multi_hop_traversal", or "single_entity"'
    )
    anchor_entities: list[str] = dspy.OutputField(
        desc="the seed entities or attributes to search for first"
    )
    search_plan: str = dspy.OutputField(
        desc="step-by-step reasoning plan describing what intermediate entities to find and how"
    )


class AnswerGapFinder(dspy.Signature):
    """Review candidate answers gathered so far and determine whether any answers
    are missing. Identify additional queries that could surface missing answers,
    and produce the final deduplicated answer list."""

    question: str = dspy.InputField()
    candidate_answers: list[str] = dspy.InputField()
    exploration_summary: str = dspy.InputField()
    missing_searches: list[str] = dspy.OutputField(
        desc="additional queries to run if more answers likely exist; empty list if none needed"
    )
    final_answer: list[str] = dspy.OutputField(
        desc="the complete, deduplicated list of answers to the question"
    )


class PhantomWikiReAct(dspy.Module):
    """Stage 2 ReAct module: iteratively searches the PhantomWiki corpus using
    two complementary retrieval tools to accumulate candidate answers."""

    def __init__(self):
        self.retrieve_broad = dspy.Retrieve(k=15)
        self.retrieve_entity = dspy.Retrieve(k=5)
        self.react = dspy.ReAct(
            signature=(
                "question: str, search_plan: str, anchor_entities: list[str] "
                "-> candidate_answers: list[str], exploration_summary: str"
            ),
            tools=[self.search_wiki, self.search_entity],
            max_iters=40,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus broadly. Returns up to 15 relevant passages.
        Best for enumeration queries or broad topic exploration."""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def search_entity(self, entity_name: str) -> str:
        """Look up a specific entity by name in PhantomWiki. Returns up to 5 directly
        relevant passages. Best for direct entity lookup by name."""
        results = self.retrieve_entity(entity_name)
        return "\n\n".join(results.passages)

    def forward(self, question, search_plan, anchor_entities):
        result = self.react(
            question=question,
            search_plan=search_plan,
            anchor_entities=anchor_entities,
        )
        return dspy.Prediction(
            candidate_answers=result.candidate_answers,
            exploration_summary=result.exploration_summary,
        )
