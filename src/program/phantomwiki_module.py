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
    3. For "how many X does the [relation] of [anchor] have?" questions:
       - The [anchor] may correspond to MULTIPLE [relation]s, each with a DIFFERENT count of X
       - Find ALL instances of [relation] for [anchor], count their X's separately
       - Each distinct count is a separate correct answer — collect ALL of them
       - Example: "how many nephews does the great-uncle of Person P have?" → P may have 5 great-uncles, each with 0, 2, 3, 4, 1 nephews → answers are ['0', '1', '2', '3', '4']
    4. For attribute-lookup questions ("What is X of the person whose Y is Z"): multiple people may share the same attribute value Y=Z — after finding one match, keep searching for more people with the same attribute.
    5. After every answer found, ask: "Are there more entities at THIS LEVEL I haven't explored yet?" Enumerate siblings, cousins, and parallel branches.
    6. If a search fails, try completely different angles: person name alone, relationship type, nearby attribute, partial name.

    NON-STANDARD KINSHIP TERMS — derive these step by step from standard relations:
    - "second uncle" or "second aunt" = parent's first cousin (your grandparent's sibling's child)
    - "second cousin" = parent's first cousin's child (your grandparent's sibling's grandchild)
    - "first cousin once removed" = your first cousin's child, OR your parent's first cousin
    - "great-uncle" or "great-aunt" = grandparent's sibling
    - "grand-nephew" or "grand-niece" = sibling's grandchild
    Never search for "second cousin of X" directly — instead derive: find X's grandparents → find their siblings → find those siblings' grandchildren.

    DATE-OF-BIRTH ANCHOR SEARCHES — exact dates like "0946-07-14" rarely match via semantic search. Try these strategies:
    - Search year only: "born 0946" or "0946 date of birth"
    - Search year + month: "0946 07" or "0946-07"
    - Search with another attribute: if the question asks about occupation too, include it
    - After finding candidates, check their page to verify their exact DOB matches

    DO NOT:
    - Stop after finding just one answer or exploring just one branch of a family tree
    - Assume a question has a unique answer
    - Confuse "the path you arrived at an ancestor via" with "all possible descendants from that ancestor"
    - Give up with "unknown" or "cannot determine" — try at least 5 distinct search strategies before concluding
    - Return a person's name when the question asks for a count (e.g., if asked "how many great-grandsons does X have?", return the number, not the names)
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
