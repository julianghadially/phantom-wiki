import dspy
import threading


# ---------------------------------------------------------------------------
# Signature with 4-step anchor-exhaustion prompt
# ---------------------------------------------------------------------------

class AnswerQuestion(dspy.Signature):
    """You are a research agent answering questions about fictional characters in a wiki.

STEP 1 — CLASSIFY THE QUESTION:
- "How many X does Y have?" → COUNT question: your answer must be NUMBER(s), never entity names.
- "Who is/are the X of Y?" → ENTITY question: return all matching entity NAMES.
- "What is the X of Y?" → ATTRIBUTE question: return all matching attribute VALUES.

STEP 2 — EXHAUST THE ANCHOR:
The anchor is the entity/condition used to find the subject (e.g., "the person whose date of birth is X", "the grandchild of John").
Multiple people may share the same date of birth, occupation, or relation. You MUST find ALL of them.
- Try different search phrasings to find every entity matching the anchor.
- Record ALL anchor entities in your notes before proceeding.
- Do NOT stop at the first match.

STEP 3 — PROCESS EACH ANCHOR ENTITY SEPARATELY:
For EACH anchor entity found in Step 2, traverse the required relation(s) independently.
- COUNT question: count the target for each anchor entity individually. Return the SET of distinct counts. Example: two anchors with counts 3 and 7 → answer is ['3', '7']. NEVER sum across anchors.
- ENTITY/ATTRIBUTE question: collect all results from every anchor entity. Return the union.

STEP 4 — VERIFY AND FINISH:
Use read_notes to review all found anchor entities and results.
Only call finish() once you have processed ALL anchor entities and exhausted ALL relation paths."""

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
            tools=[self.search_wiki, self.take_notes, self.read_notes],
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
