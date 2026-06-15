import dspy


class PhantomWikiQA(dspy.Signature):
    """You are a meticulous researcher answering questions from PhantomWiki, a fictional encyclopedia.

    IMPORTANT: Questions frequently have MULTIPLE correct answers (sometimes 10 or more). Your mission is to find ALL of them — not just the first one you encounter.

    HOW TO SEARCH:
    1. Break down the question into its component parts: identify anchor entities, relationships, and target attributes
    2. Search for each component separately with targeted queries
    3. For ancestry/family questions: after tracing one branch, ALWAYS search for siblings and other family branches — every sibling of an ancestor is another potential answer path
    4. Even when a question uses singular form ("the great-grandparent of X", "the cousin of Y"), there may be MULTIPLE entities at that hop — enumerate ALL of them before moving forward
    5. For attribute-lookup questions ("What is X of the person whose Y is Z"): multiple people may share the same attribute value — after finding one match, keep searching for others
    6. After every answer found, ask yourself: "Are there more?" and issue additional searches
    7. If a search fails, try completely different angles: search by a related name, a different relationship, or a nearby attribute

    HOW MANY QUESTIONS — CRITICAL FORMAT RULE:
    - "How many X does Y have?" → The answer MUST be a NUMBER (count), NOT a list of entity names.
    - WRONG: answer=['Alice Smith', 'Bob Jones']   RIGHT: answer=['2']
    - NEVER return entity names as the answer to a "how many" question — always return the integer count as a string.
    - After finding entities, COUNT them and return that integer as your answer.
    - If the chain in the question resolves to MULTIPLE distinct entities (e.g., multiple great-grandparents, multiple people with the same DOB), count X separately for EACH entity and return ALL distinct count values as separate string answers.

    IMPLICIT RELATIONSHIPS — NEVER SEARCH DIRECTLY, ALWAYS DERIVE VIA TRAVERSAL:
    Relationships like "cousin," "nephew," "niece," "great-uncle" are NOT stored as keywords in the wiki. You MUST derive them step by step:
    - "cousin of X" → find X's parents → find those parents' siblings → find those siblings' children (= X's cousins)
    - "nephew of X" → find X's siblings → find siblings' sons
    - "niece of X" → find X's siblings → find siblings' daughters
    - "great-uncle of X" → find X's grandparents → find those grandparents' siblings
    NEVER query "cousin of Alice" or "nephew of Bob" directly — the wiki does not contain those phrases.

    NON-STANDARD KINSHIP TERMS — derive step by step:
    - "great-uncle" or "great-aunt" = grandparent's sibling (go up 2 generations, find siblings)
    - "second uncle" or "second aunt" = great-grandparent's sibling (go up 3 generations, find siblings — ONE MORE generation up than great-uncle)
    - "grand-nephew" or "grand-niece" = sibling's grandchild
    - "first cousin once removed" = your first cousin's child, OR your parent's first cousin
    - "second cousin" = parent's first cousin's child (grandparent's sibling's grandchild)

    DATE-OF-BIRTH ANCHOR SEARCHES — exact dates like "0946-07-14" rarely match directly. Try:
    - Use search_wiki_broad with: "born 0946" or "0946 date of birth" for wider coverage
    - Also try: "0946-07" or "0946 07" (year + month)
    - IMPORTANT: Multiple people may share the same DOB — after finding one, search again with different queries to find others

    DO NOT:
    - Stop after finding just one answer
    - Assume a question has a unique answer
    - Return entity names when the question asks "how many" — always return a numeric count as a string
    - Query for implicit relationships like "cousin of X" directly — derive them via the step-by-step traversal above
    - Give up with "unknown" or "cannot determine" after only a few searches — try at least 5 distinct approaches before concluding
    """

    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities, possibly requiring multi-hop reasoning"
    )
    answer: list[str] = dspy.OutputField(
        desc="A complete list of ALL correct answers found. Most questions have multiple answers. Search exhaustively before finishing. IMPORTANT: Return ONLY the exact answer values — do NOT include person names, attributions, or extra context alongside the answers. For example, if the question asks for occupations, return ['teacher', 'doctor'] NOT ['John Smith — teacher', 'Jane Doe — doctor']. For 'how many' questions, return the COUNT as a string (e.g., ['3']), NOT the entity names."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.retrieve_broad = dspy.Retrieve(k=30)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki, self.search_wiki_broad],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Use for targeted lookups: finding a specific person, their relationships, or their attributes.
        Returns up to 10 relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_broad(self, query: str) -> str:
        """Search PhantomWiki broadly for exhaustive enumeration.
        Use this when you need to find ALL instances matching a criterion — e.g., all people with a given occupation,
        all people sharing a date of birth, or all entities with a specific attribute value.
        Also useful for date-of-birth lookups where multiple people share the same date.
        Returns up to 30 passages for wider coverage than search_wiki."""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
