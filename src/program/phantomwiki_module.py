import dspy


class ExtractRequestedValues(dspy.Signature):
    """Only the specific values asked for in the question — dates, names, occupations, or counts — without including person names or extra context like 'Name — Value' pairs"""

    question: str = dspy.InputField()
    raw_answer: list[str] = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="Only the specific values asked for in the question — dates, names, occupations, or counts — without including person names or extra context like 'Name — Value' pairs"
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.retrieve_broad = dspy.Retrieve(k=50)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki, self.search_wiki_broad],
            max_iters=50,
        )
        self.answer_extractor = dspy.ChainOfThought(ExtractRequestedValues)

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
        extracted = self.answer_extractor(question=question, raw_answer=result.answer)
        return dspy.Prediction(answer=extracted.answer)
