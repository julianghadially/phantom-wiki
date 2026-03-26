import dspy


def _extract_values(items: list[str]) -> list[str]:
    """Deterministically extract values from ReAct answer items.

    If any item contains ' — ', split all items on ' — ' and return the part
    after the separator (handling 'Name — Value' formatting). Otherwise return
    the items unchanged.
    """
    separator = " \u2014 "
    if any(separator in item for item in items):
        return [item.split(separator, 1)[1] if separator in item else item for item in items]
    return items


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.retrieve_broad = dspy.Retrieve(k=50)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki, self.search_wiki_broad],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_broad(self, query: str) -> str:
        """Search PhantomWiki for ALL entities matching a property such as an occupation or hobby — use this when you need to enumerate every person with a given attribute, returns up to 50 results"""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        answer = _extract_values(result.answer)
        return dspy.Prediction(answer=answer)
