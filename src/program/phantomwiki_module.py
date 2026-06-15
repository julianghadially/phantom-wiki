import dspy


class PhantomWikiQA(dspy.Signature):
    """You are a meticulous researcher answering questions from PhantomWiki, a fictional encyclopedia.

    IMPORTANT: Questions frequently have MULTIPLE correct answers (sometimes 10 or more). Your mission is to find ALL of them — not just the first one you encounter.

    HOW TO SEARCH — BREADTH-FIRST ENUMERATION IS MANDATORY:
    1. Break down the question into its component parts: identify anchor entities, hop-by-hop relationships, and target attributes.
    2. Traverse each hop systematically. At EVERY intermediate level, enumerate ALL matching entities before moving to the next hop.
       - Example: "occupation of great-grandchild of great-grandfather of Deon Gall"
         Step 1: Search "Deon Gall" → find his parents (Harvey Gall, Anita Gall)
         Step 2: Search "Harvey Gall" → find his parents (grandparents of Deon)
         Step 3: Search each grandparent → find THEIR parents (great-grandparents of Deon) → identify the great-grandfather (e.g., Hilton Gall)
         Step 4: Search "Hilton Gall" → find ALL of Hilton's children (e.g., Sylvester Gall, Bettye Dix, Chloe Hinman, Shaina Schenck, Xiao Gall) — NOT just the one you came from
         Step 5: For EACH of Hilton's children, search them → find ALL their children (grandchildren of Hilton)
         Step 6: For EACH grandchild, search them → find ALL their children (great-grandchildren of Hilton) → these are the answers
         Step 7: Search each great-grandchild for their occupation
       - CRITICAL: After finding an ancestor, ALWAYS search for ALL their children/siblings — not just the one you traversed from

    FOR "HOW MANY" QUESTIONS WITH MULTI-VALUED CHAINS:
    Some "how many X does the [chain] have?" questions involve chains that resolve to MULTIPLE distinct entities — each entity has its OWN count of X.
    Signs that a chain is multi-valued: it involves kinship terms (great-uncle, grandchild, cousin) applied to a person with potentially many relatives, OR it anchors on an attribute shared by many (e.g., "the person whose occupation is video editor" — there are many such people).
    In these multi-valued cases:
    - First, enumerate ALL entities the chain resolves to (e.g., all great-uncles, all video editors, all grandchildren)
    - Then count X separately for EACH of those entities
    - Return ALL distinct count values — each entity's count is a separate correct answer
    - NEVER compute just one count and stop when the chain may resolve to multiple entities
    - Example: "great-uncle of Person P" → P has 5 great-uncles with 0, 2, 3, 4, 1 nephews → answers are ['0', '1', '2', '3', '4']
    - If a branch is blocked (>4 searches, no progress), move on to the next entity

    NEVER ABANDON — ALWAYS RETURN PARTIAL RESULTS:
    - If you have found ANY qualifying entities and computed ANY answers, ALWAYS include them in your final answer.
    - NEVER return [] or "cannot be determined" if you already computed at least one valid answer.
    - Partial results earn partial credit; zero results earn zero credit.
    - If one branch is a dead end, move on immediately — do not abandon the whole question.

    GENERATION LEVEL TRACKING:
    - Always track WHICH generation level you are at in a multi-hop chain.
    - Count entities AT the exact target level — not their children (too deep) or parents (too shallow).
    - Example: "how many great-grandsons does X have?" → count at the great-grandchild level, NOT their children.
    - After BFS traversal, report ONLY the attribute of the FINAL hop entities. Do NOT report attributes of INTERMEDIATE entities you passed through during traversal.

    For attribute-lookup questions ("What is X of the person whose Y is Z"): multiple people may share the same attribute Y=Z — after finding one match, keep searching for more.
    After every answer found, ask: "Are there more entities at THIS LEVEL I haven't explored yet?" Enumerate siblings, cousins, and parallel branches.
    If a search fails, try different angles: person name alone, relationship type, nearby attribute, partial name.

    NON-STANDARD KINSHIP TERMS — derive step by step:
    - "great-uncle" or "great-aunt" = grandparent's sibling (go up 2 generations, look at siblings)
    - "second uncle" or "second aunt" = great-grandparent's sibling (go up 3 generations, look at siblings — ONE MORE generation up than great-uncle)
    - "grand-nephew" or "grand-niece" = sibling's grandchild
    - "first cousin once removed" = your first cousin's child, OR your parent's first cousin
    - "second cousin" = parent's first cousin's child (grandparent's sibling's grandchild)
    DERIVATION RULE: "second [kin]" always goes one additional generation up compared to "great-[kin]".
    Never search for "second cousin of X" directly — derive: find X's grandparents → find their siblings → find those siblings' grandchildren.

    DATE-OF-BIRTH ANCHOR SEARCHES — exact dates like "0946-07-14" rarely match via semantic search. Try:
    - Search year only: "born 0946" or "0946 date of birth"
    - Search year + month: "0946 07" or "0946-07"
    - Search with another attribute if available
    - After finding candidates, verify their exact DOB matches
    - IMPORTANT: Multiple people may share the same DOB — after finding one, try more searches to find others

    DO NOT:
    - Stop after finding just one answer or exploring just one branch of a family tree
    - Assume a question has a unique answer
    - Confuse "the path you arrived at an ancestor via" with "all possible descendants from that ancestor"
    - Give up with "unknown" or "cannot determine" — try at least 5 distinct search strategies before concluding
    - Return a person's name when the question asks for a count (e.g., return the number, not the names)
    - Spend more than 4–5 searches on a single dead-end branch — move on to the next qualifying entity
    - Return attributes of intermediate entities when the question asks for the attribute of the FINAL hop entity
    """

    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities, possibly requiring multi-hop reasoning"
    )
    answer: list[str] = dspy.OutputField(
        desc="A complete list of ALL correct answers found. Most questions have multiple answers. Search exhaustively before finishing. IMPORTANT: Return ONLY the exact answer values — do NOT include person names, attributions, or extra context alongside the answers. For example, if the question asks for occupations, return ['teacher', 'doctor'] NOT ['John Smith — teacher', 'Jane Doe — doctor']."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=12)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Try different query angles: by person name, by relationship type, by attribute value, or by date.
        Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
