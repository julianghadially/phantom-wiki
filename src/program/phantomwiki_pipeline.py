import dspy
from src.program.counting_rm import CountingRM

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class EnumerateAnchorEntities(dspy.Signature):
    """Exhaustively find ALL entities in PhantomWiki that match the filter criteria in the
    question. Issue many varied searches (different phrasings, name prefixes, related terms)
    to maximize coverage. Keep searching until no new entities appear."""

    question: str = dspy.InputField()
    anchor_entities: list[str] = dspy.OutputField(
        desc=(
            "Complete list of every entity that satisfies the question's filter criteria "
            "(e.g. every person whose hobby is auto racing). Aim for exhaustive coverage."
        )
    )


class ResolveRelationalHops(dspy.Signature):
    """Given a question and a set of anchor entities that satisfy the question's filter
    criteria, resolve any remaining relational hops (e.g. 'find the cousin of each anchor
    entity') to produce the final answers. If the anchor entities are already the final
    answers (no further hop is needed), return them directly."""

    question: str = dspy.InputField()
    anchor_entities: list[str] = dspy.InputField(
        desc="All entities found that match the question's filter criteria"
    )
    answer: list[str] = dspy.OutputField(
        desc=(
            "Final answers after resolving all relational hops. "
            "Include every valid answer; do not truncate."
        )
    )


class EntityEnumerator(dspy.Module):
    """Phase-1 module: exhaustively enumerates anchor entities via high-breadth retrieval."""

    def __init__(self):
        self.retrieve = dspy.Retrieve(k=20)
        self.react = dspy.ReAct(
            signature=EnumerateAnchorEntities,
            tools=[self.search_wiki],
            max_iters=40,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question: str):
        return self.react(question=question)


class HopResolver(dspy.Module):
    """Phase-2 module: resolves relational hops given the enumerated anchor entities."""

    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=ResolveRelationalHops,
            tools=[self.search_wiki],
            max_iters=20,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question: str, anchor_entities: list[str]):
        return self.react(question=question, anchor_entities=anchor_entities)


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.enumerator = EntityEnumerator()
        self.resolver = HopResolver()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Phase 1: exhaustively enumerate every entity matching the filter criteria
            enumeration = self.enumerator(question=question)
            anchor_entities = enumeration.anchor_entities

            # Phase 2: resolve relational hops (or pass through if no hop is needed)
            resolution = self.resolver(
                question=question, anchor_entities=anchor_entities
            )
            return dspy.Prediction(answer=resolution.answer)
