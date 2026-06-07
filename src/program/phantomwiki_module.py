import dspy
import threading


# ---------------------------------------------------------------------------
# Signature with 4-step anchor-exhaustion prompt
# ---------------------------------------------------------------------------

class AnswerQuestion(dspy.Signature):
    """You are a research agent answering questions about fictional characters in a wiki.

STEP 0 — PLAN THE HOP CHAIN:
Parse the question into an explicit hop chain: [ANCHOR] → [hop-1] → [hop-2] → ... → [ANSWER TYPE]
Count hops carefully: "great-grandchild of great-grandfather of X" = X → great-grandfather(up 3) → great-grandchild(down 3).
Save as 'hop_plan' in notes before any searching.

STEP 1 — CLASSIFY ANCHOR:
• Named-person anchor ("of Forest Benner"): EXACTLY ONE person. Search directly by name.
• Attribute-value anchor ("whose DOB is X", "whose hobby is Y", "whose occupation is Z"): MANY people share this. Use search_wiki_deep with AT LEAST 5 query phrasings. Save ALL matching entities as 'anchor_entities'.
  ⚠️ For attribute-value anchors: NEVER declare complete after 1-2 entities — expect 5-15 matching entities.
  ⚠️ Named-anchor guard: if anchor is a specific person's full name, there is exactly ONE entity — do NOT search for more.

Answer type: "How many..." → COUNT (return numbers only). "Who..." → ENTITY (return names). "What is..." → ATTRIBUTE (return values).

STEP 2 — EXHAUSTIVE HOP TRAVERSAL:
For EACH hop in your plan, process EVERY entity — do NOT stop at the first one found.

Repeat this loop for each hop level:
  For EACH entity in current hop's entity list (read from your notes):
    a. State: "Processing entity [N] of [total]: [name]. Hop [K] of [M]."
    b. Search for that entity's [hop-K relation]
    c. Use append_notes('hop_K_results', '[entity] → [results]')  ← ALWAYS use append_notes here, NOT take_notes
  ⚠️ Complete ALL entities before advancing to the next hop.

⚠️ CRITICAL: Do NOT apply the FINAL relation until ALL intermediate hops are complete for ALL entities.
⚠️ CRITICAL: Applying the final relation one hop early (hop 2 instead of hop 3) is the most common mistake — re-read hop_plan before each search.

STEP 3 — APPLY FINAL RELATION TO ALL:
For EACH entity in your last intermediate hop note (process ALL of them, one by one):
  a. State: "Applying final relation to entity [name]."
  b. Search for the final relation.
  c. append_notes('final_results', '[entity]: [result]')

Then compile the answer:
• COUNT: collect distinct count values from EVERY processed entity → return as SET of strings (e.g., ['0','2','3']). NEVER return just one count if multiple entities exist. COUNT means per-individual count — never sum across entities.
• ENTITY: collect all names from every entity → return full union (no duplicates).
• ATTRIBUTE: collect all values from every entity → return full union.

⚠️ SINGULAR PHRASING RULE: Questions using "Who is the X?" or "What is the X?" may have MULTIPLE valid answers. NEVER reduce your answer to 1 entity because the question uses "the" or singular phrasing. If your notes contain 5 female cousins, your answer MUST contain all 5. Grammatical number in the question does NOT determine cardinality.

STEP 4 — COMPLETENESS CHECK:
Before calling finish(), read all notes and verify:
1. Did you process EVERY entity at each hop? (Not just 1-2)
2. For attribute-value anchor: did you make 5+ query phrasings? Find 5+ anchor entities?
3. For COUNT: does your answer include values from ALL processed entities, not just one?
4. Was the final relation applied at the LAST hop only?
Only call finish() after confirming completeness."""

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
            tools=[self.search_wiki, self.search_wiki_deep, self.take_notes, self.append_notes, self.read_notes],
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

    def append_notes(self, key: str, note: str) -> str:
        """Append new findings to an existing note. Use this when accumulating results across multiple entities (e.g., building up hop_1_results for many anchors).
        key: the note key to append to (e.g., 'hop_1_results', 'final_results')
        note: the new information to add (will be added on a new line below any existing content)"""
        existing = self._notes.get(key, "")
        if existing:
            self._notes[key] = existing + "\n" + note
        else:
            self._notes[key] = note
        return f"Appended to note '{key}'."

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
