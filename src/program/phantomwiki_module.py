import dspy


class AnchorEnumerationQA(dspy.Signature):
    """Search PhantomWiki to find ALL people matching the anchor description in a question.

    Your ONLY goal is exhaustive entity enumeration — do NOT try to answer the question.

    ### ABSOLUTE PROHIBITION — VIOLATION TERMINATES THE TASK:
    You MUST NOT perform ANY of the following actions:
    - Traverse family trees of found people (no searching for their children, parents, siblings, cousins, grandparents, great-grandchildren, etc.)
    - Look up relatives, friends, or any person who is NOT directly stated as having the target attribute
    - Compute any family relationship counts or distances
    - Resolve the downstream question (e.g., do NOT look up cousins of found people just because the question asks about cousins)
    - Include in your anchor_entities list ANY person who does not DIRECTLY have the stated occupation, hobby, or date of birth

    If you find yourself searching for "cousins of X", "children of Y", "grandchildren of Z", "friends of W", or ANY relationship of a found person, you have violated this prohibition — STOP and return only the direct attribute matches found so far.

    EXAMPLE OF CORRECT BEHAVIOR: Question asks "How many cousins does the great-grandchild of the video editor have?"
    - CORRECT: Search for all video editors. Return their names. Done.
    - WRONG: Search for video editors, then search for Randell Kaufmann's great-grandchildren, then search for their cousins.

    ### How to Find All Anchors:

    **Attribute-based anchors** (occupation, hobby, date of birth):
    - Make 3-5 searches with DIFFERENT phrasings to find ALL people with that attribute
    - Example for "occupation video editor":
      1. Search "video editor occupation"
      2. Search "occupation video editor wiki"
      3. Search "video editor person family"
    - Example for "hobby microbiology":
      1. Search "microbiology hobby"
      2. Search "hobby microbiology wiki"
      3. Search "microbiology person"
    - Example for "date of birth 0946-07-14":
      1. Search "0946-07-14"
      2. Search "born 0946-07-14"
      3. Search "date of birth 0946-07"
      4. Search "0946 07 14 birth"
    - Extract EVERY person name from results who has that exact attribute
    - Continue searching with NEW phrasings until 2 consecutive searches yield no new names

    **Name-based anchors** (e.g., "John Smith", "the person named X"):
    - Search for the person directly, return just their name

    ### Stopping Criterion:
    - Stop when 2 consecutive searches with DIFFERENT phrasings return no new entity names
    - Aim to find at least 5-8 distinct entities for occupation/hobby anchors (there are often 20-40+)
    """
    question: str = dspy.InputField()
    anchor_entities: list[str] = dspy.OutputField(
        desc="ALL person names found that DIRECTLY match the anchor description. "
             "For attribute-based anchors (occupation/hobby/DOB), include EVERY person found "
             "with that exact attribute. Expect 5-40+ names. "
             "For name-based anchors, include only the named person. "
             "CRITICAL: Include ONLY people who directly have the stated attribute — "
             "do NOT include their relatives, friends, cousins, or any intermediate traversal results. "
             "Search multiple times with different phrasings to be exhaustive."
    )


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

    ### Rule 8: "How Many" Questions Require NUMERIC COUNTS — Not Names or Sentences
    When the question is phrased "How many X does Y have?", your answer MUST be bare numeric
    count values (e.g., "3", "7", "0") — NOT names, NOT sentences, NOT descriptions.

    **CRITICAL — Multi-Anchor Case:**
    If a "[ANCHOR SEARCH COMPLETE: ...]" block appears in the question, it means MULTIPLE
    people match the anchor description. You MUST:
    1. Process EACH listed anchor entity independently
    2. Compute the required count for EACH entity (searching as needed)
    3. Return ALL counts as separate items in your answer list
    Example: if 4 video editors have 2, 5, 0, and 3 great-grandchildren respectively,
    return ["2", "5", "0", "3"] — NOT just one of these values.

    If no anchor list is provided but the anchor is attribute-based (occupation/hobby/DOB),
    search for ALL people with that attribute yourself — there are usually many.

    Format: ONLY bare numbers in the answer list, e.g., ["0", "2", "5", "3"].
    NEVER output sentences like "Billy Childs has 0 children."

    ### Rule 9: Determine Answer Type From Question Wording
    - "Who is ..." / "What is the name of ..." → answer is person names (e.g., ["John Smith", "Jane Doe"])
    - "What is the occupation/hobby/date of birth of ..." → answer is the attribute value (e.g., ["teacher", "fishing"])
    - "How many ..." → answer is numeric counts (e.g., ["3", "0", "7"]) — NOT names
    - "What is the date of birth of ..." → answer is date strings (e.g., ["0954-10-08"])

    ### Rule 10: Closed-World Assumption — Count Only What Is Documented
    This wiki follows a CLOSED-WORLD assumption: if a person's parents, children,
    siblings, or other relatives are NOT documented in any retrieved wiki page,
    they DO NOT EXIST in this world.
    - If a person's parents are not listed, they have 0 parents (do not assume 2)
    - If a parent has no documented parents, they have 0 grandparents
    - Count only what is EXPLICITLY documented; never infer from real-world biology
    - "No record found" means the count is 0 for missing relationships

    ### Rule 11: Multi-Referent Chains — Compute Per Entity, NEVER Aggregate
    Many questions pass through MULTIPLE intermediate entities (e.g., "How many cousins does
    the friend of X have?" — X may have several friends, each with their own cousin count).
    When traversal yields multiple entities at any step, you MUST:
    1. Enumerate ALL entities at that step — do NOT pick just one representative
    2. Compute the answer INDEPENDENTLY for EACH entity using separate searches
    3. Return the COMPLETE SET of distinct per-entity values as separate answer items
    CRITICAL: Do NOT sum counts across entities. If 3 entities have [0, 1, 2] results
    respectively, return ["0", "1", "2"] — NOT "3" (a sum) and NOT just "2" (one value).
    This applies even when no anchor list was provided: whenever your traversal discovers
    multiple entities, treat each independently and collect ALL their answers.

    ### Rule 12: Second-Degree Relationships — Exact Definitions
    "Second aunt" / "second uncle" = a sibling of one of your GRANDPARENTS (not your parent).
    - Search ALL FOUR grandparents (paternal grandfather, paternal grandmother, maternal
      grandfather, maternal grandmother) and find EACH grandparent's siblings
    - Strict gender filter: second uncle = MALE grandparent sibling only;
      second aunt = FEMALE grandparent sibling only. Never count opposite-gender siblings.
    "Second cousin" = your grandparent's sibling's grandchild (NOT parent's sibling's child).
    - Full path: you → parent → grandparent → grandparent's sibling → their child → their grandchild
    - First cousin (parent's sibling's child) is a DIFFERENT relationship — do not confuse them.

    **IMPORTANT SCOPE CAVEAT**: The above definitions, the 'search all 4 grandparents' instruction,
    and the strict gender-filter rules apply ONLY to questions that explicitly use the terms
    'second aunt', 'second uncle', or 'second cousin'. For ALL other relationship types —
    including first cousin, first cousin once removed, aunt, uncle, great-aunt, great-uncle —
    do NOT apply these rules. For non-second-degree questions, apply gender filtering only to the
    FINAL answer set (not to intermediate ancestors or relatives in the traversal path).
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc=(
            "List of ALL correct answers to the question. "
            "CRITICAL: Match the answer type to the question: "
            "(1) For 'Who/What name' questions → list person names. "
            "(2) For 'What occupation/hobby/date' questions → list attribute values. "
            "(3) For 'How many' questions → list NUMERIC COUNTS only (e.g., ['3', '0', '7']), NEVER names or sentences. "
            "Include every correct answer value — do not truncate. "
            "Only return an empty list if you have exhaustively confirmed no answer exists."
        )
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        self.anchor_retrieve = dspy.Retrieve(k=15)  # wider retrieval for anchor enumeration
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki],
            max_iters=65,
        )
        self.anchor_finder = dspy.ReAct(
            signature=AnchorEnumerationQA,
            tools=[self.search_wiki_wide],
            max_iters=12,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki database for information about people, their family relationships
        (parents, children, siblings, spouses), hobbies, occupations, and dates of birth.
        Returns relevant passages. Try multiple different queries if the first does not find
        what you need — and never repeat the same query twice."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_wide(self, query: str) -> str:
        """Search the PhantomWiki database broadly to find all matching entities.
        Use this to enumerate ALL people with a given occupation, hobby, or date of birth.
        Try multiple different query phrasings to find as many matches as possible."""
        results = self.anchor_retrieve(query)
        return "\n\n".join(results.passages)

    # Phrases that indicate attribute-based anchors (requiring exhaustive enumeration)
    _ATTRIBUTE_ANCHOR_PHRASES = [
        "whose occupation is", "whose hobby is", "whose date of birth is",
        "with occupation", "with hobby", "born on", "date of birth",
        "occupation is", "hobby is",
    ]

    def _is_attribute_anchored(self, question: str) -> bool:
        """Return True if the question uses an attribute-based anchor (occupation/hobby/DOB)
        rather than a name-based anchor. Attribute-anchored questions benefit from exhaustive
        enumeration because multiple entities may share the same attribute value."""
        q_lower = question.lower()
        return any(phrase in q_lower for phrase in self._ATTRIBUTE_ANCHOR_PHRASES)

    def forward(self, question):
        # Phase 1: Exhaustive anchor enumeration — only for attribute-based anchors
        # (occupation, hobby, DOB). Skip for name-based questions to avoid misdirection.
        anchors = []
        if self._is_attribute_anchored(question):
            try:
                anchor_result = self.anchor_finder(question=question)
                anchors = list(getattr(anchor_result, 'anchor_entities', []) or [])
                anchors = [a.strip() for a in anchors if a and a.strip()]
            except Exception:
                anchors = []

        # Cap anchor list to avoid overloading the main agent
        anchors = anchors[:20]

        # Phase 2: Augment question with discovered anchor list
        if len(anchors) > 1:
            anchors_str = ", ".join(anchors)
            augmented_question = (
                f"{question}\n\n"
                f"[PRE-IDENTIFIED ANCHOR ENTITIES: {anchors_str}. "
                f"These are people matching the anchor description — they are NOT the answer. "
                f"Process EACH one and compute the required value. "
                f"Return ALL results in your answer list.]"
            )
        elif len(anchors) == 1:
            augmented_question = (
                f"{question}\n\n"
                f"[ANCHOR ENTITY CONFIRMED: {anchors[0]}]"
            )
        else:
            augmented_question = question

        # Phase 3: Main ReAct with augmented context
        result = self.react(question=augmented_question)
        return dspy.Prediction(answer=result.answer or [])
