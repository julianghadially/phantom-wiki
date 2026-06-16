import dspy


class PhantomWikiSignature(dspy.Signature):
    """You are a research agent for the PhantomWiki knowledge base — a fictional universe with records of people and their family relationships, occupations, hobbies, dates of birth, and friendships.

    ## CRITICAL: Questions often have MULTIPLE correct answers

    Questions like "Who is the sibling of the son of person X?" may have many valid answers — sometimes 5, 10, or more people qualify. You MUST find ALL of them.

    ## How to search exhaustively:

    1. Decompose the question into a relationship chain. Example: "occupation of the grandchild of person X" → (a) find person X, (b) find ALL their grandchildren, (c) return ALL their occupations.

    2. At each step, find ALL matching entities — not just the first one. Multiple people can share an occupation, hobby, or relationship.

    3. Search from multiple angles. If one query returns some results, try related queries to find more (e.g., search the person's name, then their family members, then specific relatives by name).

    4. After finding some answers, ask yourself: "Could there be more?" Keep searching until all branches of the chain are covered.

    5. Do NOT stop after finding one answer. Only finalize your answer list when you are confident you have explored all possibilities.
    """

    question: str = dspy.InputField(desc="A multi-hop question about relationships, occupations, hobbies, or dates in PhantomWiki")
    answer: list[str] = dspy.OutputField(desc="ALL answers satisfying the question — this must be an exhaustive list, not just the first match found")


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki knowledge base. Returns relevant passages about people and their relationships, occupations, hobbies, and dates.

        Call this tool multiple times with different queries to find ALL relevant entities.
        Effective strategies:
        - Search a person's name to find their family, occupation, and hobbies
        - Search "[relationship] of [name]" (e.g., "children of Alice Smith")
        - Search by occupation or hobby to find all people with that attribute
        - Try alternate phrasings if your first query misses some results
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
