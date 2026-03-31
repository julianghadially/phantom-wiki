import dspy
import json
from concurrent.futures import ThreadPoolExecutor


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
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


class EntityEnumeratorReAct(dspy.Module):
    """Stage 1: Exhaustively identify ALL intermediate entities the question asks about."""

    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature="question -> all_target_entities: list[str], attribute_to_collect: str",
            tools=[self.search_wiki],
            max_iters=20,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return result


class AttributeExtractor(dspy.Signature):
    """Extract the specific attribute value for an entity from passages. Return only the bare value with no name prefix or sentence wrapping."""

    question: str = dspy.InputField()
    entity_name: str = dspy.InputField()
    passages: str = dspy.InputField()
    attribute_value: str = dspy.OutputField(
        desc="the exact attribute value only, no name prefix, no sentence"
    )


class ParallelAttributeFetcher(dspy.Module):
    """Stage 2: For each entity, fetch targeted passages and extract the bare attribute value."""

    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.extractor = dspy.ChainOfThought(AttributeExtractor)

    def fetch_single(self, question, entity_name):
        results = self.retrieve(entity_name)
        passages = "\n\n".join(results.passages)
        pred = self.extractor(
            question=question, entity_name=entity_name, passages=passages
        )
        return {"entity": entity_name, "value": pred.attribute_value}

    def forward(self, question, all_target_entities):
        entities = all_target_entities[:4]  # cap at 4 to avoid overload
        pairs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.fetch_single, question, entity)
                for entity in entities
            ]
            for future in futures:
                pairs.append(future.result())
        return pairs


class AnswerSynthesizer(dspy.Signature):
    """Deduplicate and normalize entity-attribute pairs into a final answer list. Return only bare values with no name prefixes or sentence wrapping."""

    question: str = dspy.InputField()
    entity_attribute_pairs: str = dspy.InputField(
        desc="JSON-formatted list of {entity, value} dicts"
    )
    answer: list[str] = dspy.OutputField(
        desc="list of bare values only, no name prefixes or sentences"
    )
