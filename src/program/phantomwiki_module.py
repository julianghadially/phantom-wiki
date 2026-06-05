import dspy


class ExhaustiveInvestigationSignature(dspy.Signature):
    """You are investigating a question about fictional characters in PhantomWiki.

    CRITICAL: Many questions have MULTIPLE correct answers. Your job is to find ALL of them.

    INVESTIGATION STRATEGY:
    1. First, identify ALL key entities the question chain leads through (e.g., person -> grandparent -> grandparent's children).
    2. After finding an intermediate ancestor or group, EXPLICITLY search for EACH of their relatives/descendants by name.
    3. For "what is the occupation/hobby/trait of ALL X" questions: find EVERY member of the group first, then look up each one individually.
    4. When a retrieved passage lists multiple people (children, siblings, relatives), note ALL of them and search each one before concluding.
    5. Do NOT stop after finding 1-2 answers if the question asks for all members of a set.
    6. Exhaust all branches of the family tree relevant to the question before finalizing your answer.

    RELATIONSHIP VERIFICATION:
    - Only assert a family relationship (parent, sibling, child, cousin, etc.) if explicitly stated in retrieved text.
    - Do not assume X is Y's parent/child/sibling without direct textual evidence.
    - If a link in the chain is unclear, try multiple search queries to verify it.

    COUNTING & ENUMERATION QUESTIONS:
    - If the question asks "how many X does each of Y's [relatives] have?", return EACH count separately.
    - E.g., if one person has 2 and another has 5, return ["2", "5"] NOT ["7"].
    - If counting children/siblings for multiple people separately, list the count for each separately.

    COMPLETENESS CHECK:
    - Before finalizing, review the chain of entities you followed and ask: "Did I look up ALL members at each level?"
    - If the question involves a group (e.g., all grandchildren), verify you have searched for each group member individually.
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="Complete list of ALL correct answers. Be exhaustive—include every valid answer found, not just the first few. For counting questions, return each count as a separate string. Return [] only if the entity truly does not exist."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        self.react = dspy.ReAct(
            signature=ExhaustiveInvestigationSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for characters, families, hobbies, and occupations.
        Returns relevant passages.

        Search tips for best results:
        - Use specific person names (e.g., "John Smith family" or "Jane Doe occupation")
        - For family trees, search each relative separately (e.g., "Alice Brown children" then "Bob Brown children")
        - Try "X siblings", "X children", "X parents", or "X family" for relationship queries
        - For attributes like hobbies or occupation, search the person's full name directly
        - If the first query fails, try shorter or different phrasings
        """
        results = self.retrieve(query)
        return "\n\n---\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
