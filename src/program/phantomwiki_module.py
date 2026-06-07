import dspy
import threading


# ---------------------------------------------------------------------------
# Signature with 4-step anchor-exhaustion prompt
# ---------------------------------------------------------------------------

class AnswerQuestion(dspy.Signature):
    """You are a research agent answering questions about fictional characters in a wiki.

STEP 0 — PLAN THE HOP CHAIN:
Before any searching, parse the question into an explicit hop chain.
Count the relational hops by counting "of the" phrases and possessives.
Write: [ANCHOR] → [hop-1-relation] → [hop-2-relation] → ... → [ANSWER TYPE]

Examples:
• "sister-in-law of the friend of the friend of X" → X →[friend]→[friend]→[sister-in-law]→ ENTITY (3 hops: apply sister-in-law at hop 3, NOT hop 1)
• "great-grandchild of the great-grandfather of Y" → Y →[great-grandfather UP 3 gen]→[great-grandchild DOWN 3 gen]→ ENTITY
• "How many cousins does the daughter of Z have?" → Z →[daughter]→[cousins]→ COUNT (2 hops)
• "What is the occupation of the person whose DOB is X?" → DOB=X →[find all people]→[occupation]→ ATTRIBUTE

Save this plan in your notes as 'hop_plan'. Do NOT start searching until you have written the hop plan.

STEP 1 — CLASSIFY ANCHOR AND ANSWER TYPE:
Anchor type:
• Named-person anchor (e.g., "of Forest Benner", "of Demetria Woodland"): EXACTLY ONE person exists with this name. Search by name directly. Do NOT try to find multiple matching entities.
• Attribute-value anchor (e.g., "whose date of birth is 0945-06-12", "whose hobby is painting", "whose occupation is X"): MANY people may share this attribute. Use search_wiki_deep and try AT LEAST 5 different query phrasings (e.g., "date of birth 0945-06-12", "born 0945-06-12", "DOB 0945-06-12", "0945-06-12 person", "born June 12 year 0945") to find ALL matching entities. NEVER settle for a nearest-date approximation, NEVER strip date components to broaden the query.

Answer type:
• "How many X does Y have?" → COUNT: your answer must be NUMBER(s), never entity names.
• "Who is/are the X of Y?" → ENTITY: return all matching entity NAMES.
• "What is the X of Y?" → ATTRIBUTE: return all matching attribute VALUES.

STEP 2 — TRAVERSE HOP BY HOP:
Follow your hop_plan exactly — execute one hop at a time.
• Save each hop's results in notes: 'hop_1_results', 'hop_2_results', 'hop_3_results', etc.
• Before each search, check your notes to confirm WHICH HOP you are currently executing.
• Exhaust all entities at the current hop before moving to the next.

⚠️ CRITICAL RULE: Do NOT apply the FINAL relation until you have FULLY completed ALL intermediate hops and recorded them in notes. If your plan says "A → B → C → ANSWER", then you must have 'hop_1_results' (B entities) and 'hop_2_results' (C entities) in notes BEFORE computing the answer. Applying the final relation one hop too early is the #1 mistake — verify your hop count against your hop_plan before finishing.

STEP 3 — APPLY FINAL RELATION TO ALL PREVIOUS-HOP ENTITIES:
Apply the FINAL relation to EVERY entity from the previous hop (all entities in your last intermediate results note).
• COUNT question: compute the count for EACH entity separately → return the SET of all distinct counts as strings. NEVER sum across entities. Example: 3 entities with counts 2, 3, 3 → answer is ['2', '3'].
• ENTITY question: collect all matching names from EVERY entity → return the full union (no duplicates).
• ATTRIBUTE question: collect all matching attribute values from EVERY entity → return the full union.

STEP 4 — VERIFY AND FINISH:
Read your 'hop_plan' note. Verify:
1. You executed all hops in the plan
2. The final relation was applied at the LAST hop (not an earlier one)
3. The answer type matches the question format
Only call finish() after verifying."""

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="ALL correct answers — complete set. For COUNT questions: numbers only (e.g., ['3', '7']), never names. For WHO/WHAT questions: all matching names or values."
    )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        # Thread-local storage for notes (evaluation uses multiple threads)
        self._local = threading.local()

        # k=10 for broader document coverage per search
        self.retrieve = dspy.Retrieve(k=10)

        self.react = dspy.ReAct(
            signature=AnswerQuestion,
            tools=[self.search_wiki, self.search_wiki_deep, self.take_notes, self.read_notes],
            max_iters=50,
        )

    # ------------------------------------------------------------------
    # Thread-safe notes property
    # ------------------------------------------------------------------

    @property
    def _notes(self):
        if not hasattr(self._local, "notes"):
            self._local.notes = {}
        return self._local.notes

    @_notes.setter
    def _notes(self, value):
        self._local.notes = value

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages about the queried topic.
        Tips: Search by person name for full articles, or by attribute for entity discovery."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_deep(self, query: str) -> str:
        """Deep search using 30 results instead of 10. Use this ONLY for attribute-value anchor enumeration (e.g., finding ALL people whose date of birth is X, whose hobby is Y, whose occupation is Z). Returns more results to maximize entity discovery for multi-entity attribute anchors."""
        retrieve_deep = dspy.Retrieve(k=30)
        results = retrieve_deep(query)
        return "\n\n".join(results.passages)

    def take_notes(self, key: str, note: str) -> str:
        """Save a finding or plan to your notes workspace.
        key: short identifier (e.g., 'anchor_entities', 'results_per_anchor', 'todo')
        note: what you found or plan to investigate next"""
        self._notes[key] = note
        return f"Saved note '{key}'. You now have {len(self._notes)} note(s) total."

    def read_notes(self, key: str = "all") -> str:
        """Read from your notes workspace.
        key: 'all' to see all notes, or a specific key to read one note"""
        if key == "all":
            if not self._notes:
                return "No notes saved yet."
            return "\n".join(f"[{k}]: {v}" for k, v in self._notes.items())
        return self._notes.get(key, f"No note found with key '{key}'.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, question):
        self._notes = {}  # Reset notes for each new question
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
