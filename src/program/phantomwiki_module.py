import dspy


class PhantomWikiSignature(dspy.Signature):
    """You are a research agent for the PhantomWiki knowledge base — a fictional universe with records of people, family relationships, occupations, hobbies, dates of birth, and friendships.

    ## CRITICAL: Questions often have MULTIPLE correct answers

    Questions may have 1, 5, 10, or more valid answers. You MUST find ALL of them.

    ## Step 1: Decompose the question into a chain of lookups

    Example: "occupation of the grandchild of person X"
    → (a) find person X, (b) find ALL grandchildren of X (search each of their parents), (c) return ALL their occupations.

    DIRECTION RULE: Ancestor lookups (great-grandfather, grandparent, parent OF X) require going UP the family tree from X. Descendant lookups (great-grandchild, grandchild, child OF X) require going DOWN. Never confuse these directions before searching.

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
    - second uncle / great-uncle of X: X's grandparents' siblings → search X's parents → find each grandparent → find grandparents' siblings
    - nephew / niece of X: children of X's siblings → search each of X's siblings for their children
    - great-grandchild of X: children of X's grandchildren

    NEVER search "cousin of X" or "sister-in-law of X" — those fields don't exist. Always decompose.

    ## Step 4: Discover ALL matching entities using multiple searches

    When finding all entities that share a property (same hobby, occupation, or date of birth):
    - Make AT LEAST 2–3 searches with different phrasings before moving to the next chain step
    - Example for "hobby is microbiology": search "hobby microbiology", then "microbiology hobbyist", then "microbiology" — each may return different people
    - A single search may miss many matches — PhantomWiki has many people per property
    - Only proceed to chain traversal after making multiple independent discovery searches

    ## Step 5: For each entity found, check ALL its connections — never stop partway

    Once you find an intermediate entity (e.g., a grandparent, sibling, or cousin):
    - Explicitly find ALL their relevant connections (all children, all siblings, all parents) as needed by the question
    - Do NOT stop after finding 1–2 connections when more likely exist
    - If a person has 3 children listed, you must look up all 3 before finalizing
    - Before finalizing answers, explicitly ask yourself: "Have I explored ALL branches of every intermediate entity? Are there siblings, children, or parents I haven't checked yet?"

    ## Step 6: Anti-loop guard — move on after 2 failed searches

    If a search returns no useful new information about a specific entity or sub-path:
    - Try at most 2 different phrasings for that same entity or relationship
    - After 2 failures on the same sub-path, mark it as "data unavailable" and continue to remaining unexplored branches
    - Never repeat the same search query — always rephrase or move on

    ## Step 7: Search from multiple angles and persist

    - Search each person by name to find their family, occupation, and hobbies
    - Search "children of [name]", "siblings of [name]", "parents of [name]" for direct relationships
    - Search by occupation or hobby to find all people with that attribute (multiple searches may be needed)
    - For dates of birth (e.g., 0945-06-12): try multiple forms — "0945-06-12", "born 0945-06-12", "date of birth 0945-06-12"
    - For derived relationships (cousin, in-law, uncle): search the CONSTITUENT parts, not the derived relationship name
    - Try alternate phrasings if your first query returns no results
    - After finding some answers, ask: "Are there more branches I haven't explored yet?" Keep going until all branches are covered
    - Do NOT stop after finding one answer — only finalize when all relationship-chain branches are exhausted

    ## Step 8: Match the answer type to the question

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
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki knowledge base. Returns up to 20 relevant passages about people and their relationships, occupations, hobbies, and dates.

        Call this tool multiple times with different queries to find ALL relevant entities.
        Effective strategies:
        - Search a person's name to find their family, occupation, and hobbies
        - Search "children of [name]", "siblings of [name]", "parents of [name]" for direct relationships
        - Search by occupation or hobby to find all people with that attribute (use 2-3 different phrasings to find everyone)
        - For dates of birth, try multiple forms: "0945-06-12", "born 0945-06-12", "date of birth 0945-06-12"
        - For derived relationships (cousin, in-law, uncle): search the CONSTITUENT parts, not the derived relationship name
        - Try alternate phrasings if your first query returns no results
        - After 2 failed searches for the same entity, move on to other branches
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
