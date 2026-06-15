import threading
import dspy


class PhantomWikiQA(dspy.Signature):
    """You are a meticulous researcher answering questions from PhantomWiki, a fictional encyclopedia.

    IMPORTANT: Questions frequently have MULTIPLE correct answers (sometimes 10 or more). Your mission is to find ALL of them — not just the first one you encounter.

    ══════════════════════════════════════════════════════════
    WORKSPACE TOOLS — MANDATORY USAGE
    ══════════════════════════════════════════════════════════
    You have two workspace tools: note_finding() and recall_findings().

    RULE 1 — RECORD INTERMEDIATE ENTITIES: Whenever you discover a set of intermediate entities
    (e.g., all hobbyists matching a criterion, all grandchildren, all friends of a person,
    all great-grandparents), you MUST immediately call note_finding() with a clear label and
    the COMPLETE list of entities found. Do not proceed to the next step until you have noted them.

    RULE 2 — ENUMERATE BEFORE COMPUTING: For "how many X does [the Y of Z] have?" questions,
    you MUST first find ALL entities that satisfy Y=Z (there may be 5-10 matching people),
    note them ALL via note_finding(), and ONLY THEN compute X for each one separately.

    RULE 3 — CHECK BEFORE FINISHING: Before producing your final answer, ALWAYS call
    recall_findings() to verify that EVERY noted entity has been fully explored.
    If any remain unexplored, continue searching before finalizing.

    ══════════════════════════════════════════════════════════
    STEP-BY-STEP PROCESS FOR "HOW MANY" QUESTIONS
    ══════════════════════════════════════════════════════════
    Question form: "How many X does the [relation chain] of [anchor] have?"

    Step 1: Resolve the relation chain to get all intermediate entities.
            Example — "the great-grandparent of Dwight" → find ALL 4 great-grandparents.
    Step 2: note_finding("intermediate entities", "entity1, entity2, entity3, ...")
    Step 3: For EACH entity, search for their friends/siblings/children/etc. and note those too.
    Step 4: For EACH final entity, count X (e.g., number of friends).
            note_finding("counts", "entity1: N1, entity2: N2, ...")
    Step 5: recall_findings() — ensure every entity from Step 2 has a count in Step 4.
    Step 6: Return ALL distinct counts as the answer list.

    Example: "How many friends does the friend of the great-grandparent of Dwight Lazarus have?"
    → Dwight has 4 great-grandparents → each may have friends → each friend has a friend count.
    → Expected answer: ['1', '2', '3', '4', '5', '6', '7', '8', '9'] — one count per path.
    → You MUST explore every great-grandparent's friends, not just one.

    ══════════════════════════════════════════════════════════
    GENERAL SEARCH STRATEGY
    ══════════════════════════════════════════════════════════
    1. Break down the question into component parts: anchor entities, relationships, target attributes
    2. Search for each component with targeted queries
    3. For ancestry/family questions: after tracing one branch, ALWAYS search for siblings and
       other family branches — every sibling of an ancestor creates another valid answer path
    4. For attribute-lookup questions ("the person whose hobby is X"): multiple people may share
       the same hobby/occupation/date — after finding one match, keep searching for all others
    5. For date-of-birth queries: search by year only ('born 0858') if exact date fails; also try
       searching by a known family member's name instead of the date directly
    6. After every answer found, ask yourself: "Are there more?" and issue additional searches
    7. Try at least 5 distinct query angles before concluding a search has failed

    DO NOT:
    - Pick just one intermediate entity and ignore the rest — enumerate ALL of them
    - Stop after finding just one answer
    - Assume a question has a unique answer
    - Return 'unknown' or 'cannot determine' without trying at least 5 different query approaches
    - Forget to explore sibling branches (each sibling creates a new valid path)
    - Collapse multiple distinct counts into a single number
    """

    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities, possibly requiring multi-hop reasoning"
    )
    answer: list[str] = dspy.OutputField(
        desc=(
            "A complete list of ALL correct answers found. Most questions have multiple answers. "
            "Search exhaustively before finishing. "
            "For 'how many' questions: return ALL distinct counts found across ALL matching entities — "
            "e.g., if 5 people match the anchor and their counts are 0, 1, 1, 3, 4, return ['0', '1', '3', '4']. "
            "IMPORTANT: Return ONLY the exact answer values — do NOT include person names, attributions, "
            "or extra context alongside the answers. For example, if the question asks for occupations, "
            "return ['teacher', 'doctor'] NOT ['John Smith — teacher', 'Jane Doe — doctor']."
        )
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self._local = threading.local()
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki, self.note_finding, self.recall_findings],
            max_iters=50,
        )

    def _get_notes(self):
        """Get the thread-local notes list, initializing if needed."""
        if not hasattr(self._local, "notes"):
            self._local.notes = []
        return self._local.notes

    def note_finding(self, label: str, content: str) -> str:
        """Record an important intermediate finding to your workspace so you don't forget to explore it.
        Use this whenever you find a set of entities at an intermediate reasoning step.
        label: short description of what these entities are (e.g., 'great-grandparents of Dwight', 'die-cast toy hobbyists', 'friends of Demetria')
        content: the complete list of entities or counts found (e.g., 'Lauretta Westerman, Troy Westerman, Marlena Lazarus, Spencer Lazarus')
        Returns a confirmation with the total number of entries noted so far."""
        notes = self._get_notes()
        notes.append(f"[{label}]: {content}")
        return f"Noted ({len(notes)} total entries). Remember to explore ALL entities listed here."

    def recall_findings(self) -> str:
        """Read all intermediate findings recorded so far. Call this before finalizing your answer
        to verify that every noted entity has been explored. If any remain unexplored, continue searching.
        Returns all workspace entries."""
        notes = self._get_notes()
        if not notes:
            return "Workspace is empty — you have not noted any intermediate findings yet."
        return "=== WORKSPACE CONTENTS ===\n" + "\n".join(notes) + "\n=== END WORKSPACE ==="

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Try different query angles: by person name, by relationship type, by attribute value, or by date.
        For date-of-birth searches, try year-only queries (e.g., 'born 0858') if exact date strings fail.
        Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        # Reset workspace for each new question (thread-safe via thread-local storage)
        self._local.notes = []
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
