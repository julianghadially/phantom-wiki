import re as _re

import dspy

# Matches a date like 0945-06-12 anywhere in a search query string
_DOB_PATTERN = _re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')

# Matches "the person whose (hobby|occupation) is VALUE" — property-anchor questions
# These admit multiple people with the same property, requiring multi-entity enumeration
_PROPERTY_ANCHOR = _re.compile(
    r'\bthe\s+persons?\s+whose\s+(hobby|occupation)\s+is\s+(.+?)(?=\s+have\b|\s+has\b|\?|\s*$)',
    _re.IGNORECASE
)


class PhantomWikiSignature(dspy.Signature):
    """You are a research agent for the PhantomWiki knowledge base — a fictional universe with records of people, family relationships, occupations, hobbies, dates of birth, and friendships.

    ## CRITICAL: Questions often have MULTIPLE correct answers

    Questions may have 1, 5, 10, or more valid answers. You MUST find ALL of them.

    ## Step 1: Decompose the question into a chain of lookups

    Example: "occupation of the grandchild of person X"
    → (a) find person X, (b) find ALL grandchildren of X (search each of their parents), (c) return ALL their occupations.

    DIRECTION RULE: Ancestor lookups (great-grandfather, grandparent, parent OF X) require going UP the family tree from X. Descendant lookups (great-grandchild, grandchild, child OF X) require going DOWN. Never confuse these directions.
    GENERATION DEPTH: "grand-" = 2 hops; "great-grand-" = 3 hops; "great-great-grand-" = 4 hops. Example: "great-grandparent of X" → (1) find X's parents (1 up), (2) find THEIR parents = X's grandparents (2 up), (3) find THEIR parents = X's great-grandparents (3 up). Do not stop at step 2 for a great-grandparent question.
    ANCESTOR BRANCH: After identifying the target ancestor (e.g., the great-grandfather "Hilton Gall"), immediately search for that ancestor by name to find ALL their children — you have only seen the one branch leading back to X; there are other branches you have not yet visited.

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
    - For dates of birth (e.g., 0945-06-12): try multiple forms — "0945-06-12", "born 0945", "date of birth 0945-06-12"
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
        self.retrieve_k30 = dspy.Retrieve(k=30)  # higher-k retriever for DOB queries
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=75,  # Increased from 50
        )
        # Shorter sub-ReAct for per-entity sub-questions in multi-entity mode
        self.sub_react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=20,
        )

    def _extract_anchor_property(self, question: str):
        """Return (prop_type, prop_value) if question has a property-anchor pattern, else None."""
        m = _PROPERTY_ANCHOR.search(question)
        if m:
            return m.group(1).lower(), m.group(2).strip()
        return None

    def _pre_retrieve_anchor(self, prop_type: str, prop_value: str) -> list:
        """Pre-retrieve passages for anchor property using multiple phrasings."""
        seen: set = set()
        passages: list = []
        for phrasing in [
            prop_value,
            f"{prop_type} {prop_value}",
            f"{prop_value} {prop_type}",
        ]:
            for p in self.retrieve(phrasing).passages:
                if p not in seen:
                    seen.add(p)
                    passages.append(p)
        return passages[:30]

    def _extract_entity_names_from_passages(self, passages: list, prop_value: str) -> list:
        """Extract person names from ColBERT passages.

        Passage format: "Name Surname: # Name Surname  ## Family ..."
        Entity name is everything before the first colon.
        Only include passages that actually contain the property value.
        """
        names: list = []
        seen: set = set()
        for passage in passages:
            if prop_value.lower() not in passage.lower():
                continue
            colon_idx = passage.find(':')
            if colon_idx <= 0:
                continue
            name = passage[:colon_idx].strip()
            parts = name.split()
            if (len(parts) >= 2 and
                    all(p[0].isupper() for p in parts if p) and
                    name not in seen):
                names.append(name)
                seen.add(name)
        return names[:12]  # Cap at 12 entities to control compute

    def _make_entity_question(self, question: str, prop_type: str, prop_value: str, entity_name: str) -> str:
        """Replace 'the person whose PROP is VALUE' with entity_name in the question."""
        lower_q = question.lower()
        for anchor in [
            f"the person whose {prop_type} is {prop_value}",
            f"the persons whose {prop_type} is {prop_value}",
        ]:
            idx = lower_q.find(anchor.lower())
            if idx >= 0:
                return question[:idx] + entity_name + question[idx + len(anchor):]
        return question  # Fallback: return unchanged

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki knowledge base. Returns relevant passages about people and their relationships, occupations, hobbies, and dates.

        Call this tool multiple times with different queries to find ALL relevant entities.
        Effective strategies:
        - Search a person's name to find their family, occupation, and hobbies
        - Search "children of [name]", "siblings of [name]", "parents of [name]" for direct relationships
        - Search by occupation or hobby to find all people with that attribute (multiple searches may be needed to find everyone)
        - For dates of birth, try multiple forms: "0945-06-12", "born 0945-06-12", "date of birth 0945-06-12"
        - For derived relationships (cousin, in-law, uncle): search the CONSTITUENT parts, not the derived relationship name
        - Try alternate phrasings if your first query returns no results
        """
        # Transparent DOB intercept: when the query contains a date (YYYY-MM-DD),
        # issue multiple phrasings at higher k and filter to exact date matches.
        dob_match = _DOB_PATTERN.search(query)
        if dob_match:
            year, month, day = dob_match.group(1), dob_match.group(2), dob_match.group(3)
            full_date = f"{year}-{month}-{day}"
            # Search with (a) exact date, (b) year+month (broader), (c) year-only (broadest)
            # at k=30 each, then filter to only passages containing the exact date string.
            seen: set = set()
            all_passages: list = []
            for phrasing in [full_date, f"{year}-{month}", f"born {year}"]:
                for passage in self.retrieve_k30(phrasing).passages:
                    if passage not in seen:
                        seen.add(passage)
                        all_passages.append(passage)
            exact_matches = [p for p in all_passages if full_date in p]
            if exact_matches:
                return "\n\n".join(exact_matches[:30])
            # Fallback: return original query results at k=20
            return "\n\n".join(self.retrieve(query).passages)

        return "\n\n".join(self.retrieve(query).passages)

    def forward(self, question):
        # Detect property anchor (e.g., "the person whose hobby is die-cast toy")
        anchor = self._extract_anchor_property(question)

        if anchor:
            prop_type, prop_value = anchor
            # Pre-retrieve all passages matching the anchor property
            passages = self._pre_retrieve_anchor(prop_type, prop_value)
            # Extract entity names from the passage headers
            entities = self._extract_entity_names_from_passages(passages, prop_value)

            if len(entities) >= 2:
                # Multi-entity mode: run a focused sub-question for each entity and union results
                all_answers: list = []
                for entity_name in entities:
                    entity_q = self._make_entity_question(question, prop_type, prop_value, entity_name)
                    try:
                        sub_result = self.sub_react(question=entity_q)
                        if sub_result.answer:
                            for ans in sub_result.answer:
                                if ans and ans.strip():
                                    all_answers.append(ans.strip())
                    except Exception:
                        pass  # Skip failed sub-questions gracefully

                if all_answers:
                    # Deduplicate while preserving first-occurrence order
                    seen_lower: set = set()
                    final_answers: list = []
                    for ans in all_answers:
                        normalized = ans.strip().lower()
                        if normalized not in seen_lower:
                            final_answers.append(ans.strip())
                            seen_lower.add(normalized)
                    return dspy.Prediction(answer=final_answers)

        # Default path: main ReAct (also handles single-entity anchor questions)
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
