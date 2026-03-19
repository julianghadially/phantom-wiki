import dspy


class PhantomWikiRLM(dspy.Module):
    def __init__(self, k=7, max_iterations=15, max_llm_calls=50):
        self.k = k
        self.retrieve = dspy.Retrieve(k=self.k)

        self.rlm = dspy.RLM(
            "question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for articles matching the query.

        Use this to find information about fictional characters, their
        relationships, occupations, hobbies, and other attributes.
        Returns the text of the top matching passages.

        For multi-hop questions, call this multiple times with different
        queries as you discover new entity names and relationships.
        """
        results = self.retrieve(query)
        return "\n\n---\n\n".join(results.passages)

    def forward(self, question):
        print(f"Question: {question}")
        result = self.rlm(question=question)
        print(f"Result: {result.answer}")
        return dspy.Prediction(answer=result.answer)
