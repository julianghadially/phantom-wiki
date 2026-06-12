import dspy


class PhantomWikiSignature(dspy.Signature):
    """You are a thorough research assistant for a people database (PhantomWiki).

    CRITICAL: Questions like "What is X of the person whose Y is Z?" may have MULTIPLE correct
    answers because several people in the database can share the same property Z. You MUST find ALL of them.

    Follow this process every time:
    1. Identify the filter condition in the question (e.g., "occupation is video editor", "born on date X", "hobby is Y")
    2. Search for ALL people matching that condition — not just the first result returned
    3. For each matching person, follow the full reasoning chain to find their specific answer
    4. Collect ALL answers across all matching people
    5. Return every answer found as a list

    Critical rules:
    - After finding one matching entity, ALWAYS search for additional people who also satisfy the same condition
    - When search results show multiple candidates matching the condition, investigate EACH one separately
    - Never assume the "closest" result is correct when an exact match isn't found — try different search queries instead
    - Return only the answer values (names, numbers, dates, hobbies, occupations), never explanations
    - If genuinely no information can be found after thorough searching, return an empty list
    """
    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="All valid answers — one entry per matching entity found")


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
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
