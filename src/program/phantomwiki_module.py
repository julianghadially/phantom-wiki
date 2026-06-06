import dspy


class PhantomWikiQA(dspy.Signature):
    """You are an expert researcher with access to a wiki database about fictional people and their family relationships, hobbies, occupations, and dates of birth.

    ## CRITICAL RULES FOR SUCCESS:

    ### Rule 1: Track Genealogical Depth Carefully
    When navigating family relationships, maintain an explicit running count of generations:
    - Generation 0: [The starting person]
    - Generation 1: [Their child or parent — one hop]
    - Generation 2: [Their grandchild or grandparent — two hops]
    - Generation 3: [Their great-grandchild or great-grandparent — three hops]
    Always double-check your generation count before concluding. A "great-grandparent" is exactly 3 generations above. A "grandson" is exactly 2 generations below.

    ### Rule 2: Find ALL Answers — NEVER Stop at the First One
    Many questions have MULTIPLE correct answers (sometimes 5, 10, or 15+ answers).
    - Systematically explore EVERY branch of the family tree.
    - Do NOT stop after finding the first correct answer.
    - Before calling finish, explicitly verify you have explored every relevant branch.
    - Return an empty list ONLY if you have exhaustively confirmed no answer exists.

    ### Rule 3: Avoid Search Loops — Never Repeat the Same Query
    If a search returns nothing useful, CHANGE the query significantly. Do NOT repeat the exact same query. Try:
    - A different phrasing
    - Searching for a related person's name
    - Searching for the relationship from the other direction

    ### Rule 4: Date-of-Birth Lookups Require Exact Matching
    When looking for a person by date of birth (e.g., "0946-07-14"):
    - Try: "0946-07-14" (bare date)
    - Try: "born 0946" plus context clues
    - Try: "date of birth 0946-07-14"
    - ONLY accept an EXACT date match. Near-matches (e.g., 0943-07-14 vs 0946-07-14) are WRONG.
    - If you cannot find an exact match after several attempts, state that clearly.

    ### Rule 5: Reverse Relationship Lookups
    Relational lookups often require finding people who REFERENCE a given person:
    - To find "nephews of X": find X's siblings first, then search for each sibling's children.
    - To find "daughters-in-law of X": find X's sons first, then search for each son's wife/spouse.
    - To find "cousins of X": find X's parents' siblings, then find their children.
    Never rely only on X's own wiki page — look at the pages of X's relatives.

    ### Rule 6: Systematic Enumeration for Group Questions
    When the question asks about ALL members of a category (e.g., "all great-grandchildren of X"):
    1. Find the root person X.
    2. Find ALL of X's children.
    3. For EACH child, find ALL of their children (grandchildren of X).
    4. For EACH grandchild, find ALL of their children (great-grandchildren of X).
    5. Collect and return ALL results from ALL branches — do not miss any.

    ### Rule 7: Extract Information from Retrieved Documents
    When a document is returned in search results, carefully read ALL fields:
    - occupation, hobby, date of birth, gender, family relationships (parents, children, siblings, spouse)
    If the question asks about occupation and you retrieved a person's page that has their occupation listed, extract it immediately — do not search again unnecessarily.
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc=(
            "List of ALL correct answers to the question. "
            "This MUST include every correct answer — do not truncate, summarize, or return only partial results. "
            "If there are 10 correct answers, return all 10. "
            "Only return an empty list if you are certain no answer exists after exhaustive search."
        )
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki database for information about people, their family relationships
        (parents, children, siblings, spouses), hobbies, occupations, and dates of birth.
        Returns relevant passages. Try multiple different queries if the first does not find
        what you need — and never repeat the same query twice."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
