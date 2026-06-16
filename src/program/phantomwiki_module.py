import dspy


class PhantomWikiSignature(dspy.Signature):
    """You are a research agent for the PhantomWiki knowledge base — a fictional universe with records of people, family relationships, occupations, hobbies, dates of birth, and friendships.

    ## CRITICAL: Questions often have MULTIPLE correct answers

    Questions may have 1, 5, 10, or more valid answers. You MUST find ALL of them.

    ## Step 1: Decompose the question into a chain of lookups

    Example: "occupation of the grandchild of person X"
    → (a) find person X, (b) find ALL grandchildren of X (search each of their parents), (c) return ALL their occupations.

    DIRECTION RULE: Ancestor lookups (great-grandfather, grandparent, parent OF X) require going UP the family tree from X. Descendant lookups (great-grandchild, grandchild, child OF X) require going DOWN. Never confuse these directions.
    GENERATION DEPTH: "grand-" = 2 hops (grandparent = 2 up, grandchild = 2 down); "great-grand-" = 3 hops. Count "great-" prefixes carefully — do not stop one generation short.

    ## Step 2: When multiple entities match, enumerate each — NEVER sum or aggregate

    When a question's subject resolves to MULTIPLE people (e.g., "all people with hobby X", "all people born on date Y"), the answer is the SET of distinct individual values — one value per person — NOT a total sum.

    BAD: "How many friends does the person whose hobby is microbiology have?" → ["29"]   (wrong: summed all people's friend counts)
    GOOD: "How many friends does the person whose hobby is microbiology have?" → ["0", "2", "3", "4", "5", "7"]   (right: distinct per-person counts)

    BAD: "How many siblings does each biochemist have?" → ["18"]   (wrong: summed across all biochemists)
    GOOD: "How many siblings does each biochemist have?" → ["0", "1", "2", "3", "4"]   (right: one entry per distinct count)

    Rule: For EVERY person matching the subject, record THAT PERSON'S individual value. Then return the DEDUPLICATED SET of those values.

    ## Step 3: Decompose derived relationships — NEVER search for them directly

    PhantomWiki stores only direct relationships (parents, children, siblings, spouse, friends). Derived relationships do not exist as fields — you must compute them:

    - cousin of X: X's parents' siblings' children → search X's parents → find each parent's siblings → find their children
    - sister-in-law of X: X's spouse's sisters, AND X's brothers' wives → search X's spouse for sisters; search X's brothers for their wives
    - brother-in-law of X: X's spouse's brothers, AND X's sisters' husbands
    - second uncle / great-uncle / second aunt / great-aunt of X: X's grandparents' siblings — TWO generations above X, NOT the second item in a list of X's regular aunts/uncles. Search X's parents → find each grandparent → find ALL of that grandparent's siblings.
    - nephew / niece of X: children of X's siblings → search each of X's siblings for their children
    - great-grandchild of X: children of X's grandchildren

    NEVER search "cousin of X" or "sister-in-law of X" — those fields don't exist. Always decompose.

    ## Step 4: Search exhaustively and persist when stuck

    - Search each person by name to find their family, occupation, and hobbies
    - For dates of birth: use the search_by_dob tool (not search_wiki) with the exact date string. Multiple people often share the same date of birth — treat a DOB-anchored subject exactly like an occupation or hobby: issue multiple searches and process EVERY matched person.
    - When finding all entities with a shared property (hobby X, occupation Y, or date Z): use 2–3 different search phrasings since a single query may miss many matches
    - When you find an intermediate entity, check ALL of its relevant connections before finalizing (e.g., if a grandparent has 4 children, look up all 4; if a person has 3 siblings, check all 3)
    - If a search returns no new information after 2 attempts on the same sub-path, move on to other unexplored branches rather than repeating the same query
    - If a person's family page returns no siblings or children, try searching their parents to triangulate
    - After finding some answers, ask: "Are there more branches I haven't explored yet?" Keep going until all branches are covered
    - Do NOT stop after finding one answer — only finalize when all relationship-chain branches are exhausted

    ## Step 5: Match the answer type to the question

    - "Who is...?" → return the person's name(s)
    - "What is the occupation/hobby/date of...?" → return the attribute value(s), not names
    - "How many X does ONE specific person have?" → count X, return the NUMBER as a single string (e.g., ["3"])
    - "How many X does [a category of people] each have?" → return the SET of distinct individual counts (e.g., ["0", "1", "2", "3"])
    """

    question: str = dspy.InputField(desc="A multi-hop question about relationships, occupations, hobbies, or dates in PhantomWiki")
    answer: list[str] = dspy.OutputField(desc="ALL answers satisfying the question — exhaustive list of distinct values; never a sum or aggregate")


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=20)
        self.retrieve_dob = dspy.Retrieve(k=30)
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki, self.search_by_dob],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki knowledge base. Returns up to 20 relevant passages about people and their relationships, occupations, hobbies, and dates.

        Call this tool multiple times with different queries to find ALL relevant entities.
        Effective strategies:
        - Search a person's name to find their family, occupation, and hobbies
        - Search "children of [name]", "siblings of [name]", "parents of [name]" for direct relationships
        - Search by occupation or hobby to find all people with that attribute (multiple searches may be needed to find everyone)
        - For dates of birth, try multiple forms: "0945-06-12", "born 0945-06-12", "date of birth 0945-06-12"
        - For derived relationships (cousin, in-law, uncle): search the CONSTITUENT parts, not the derived relationship name
        - Try alternate phrasings if your first query returns no results
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_by_dob(self, date_str: str) -> str:
        """Search for all people matching a specific date of birth. Use this tool INSTEAD of
        search_wiki when the question anchors on a date of birth (e.g., '1050-09-16').

        Tries multiple phrasings with higher recall to maximize the chance of finding ALL
        people who share that date of birth — multiple people often have the same DOB.

        Args:
            date_str: The date of birth string, e.g. '1050-09-16'
        """
        parts = date_str.strip().split('-')
        year = parts[0] if parts else date_str

        queries = [
            date_str,
            f"date of birth {date_str}",
            f"born {year}",
            f"born in {year}",
        ]

        seen: set = set()
        passages: list = []
        for q in queries:
            results = self.retrieve_dob(q)
            for p in results.passages:
                if p not in seen:
                    seen.add(p)
                    passages.append(p)

        return "\n\n".join(passages[:60])

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
