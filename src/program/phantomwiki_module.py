import dspy
from typing import List


class ExhaustiveMultiHopQA(dspy.Signature):
    """Answer questions about fictional characters by searching the PhantomWiki corpus.

    CRITICAL: Most questions have MANY correct answers, not just one. The question
    involves a category (e.g., "structural engineer", "die-cast toy" hobbyist, a birth
    date) that matches MULTIPLE people, and you must find ALL of them.

    STRATEGY — follow this every time:
    1. Identify the "anchor" (occupation, hobby, birth date, name) in the question.
    2. Search for EVERY person who matches that anchor. Issue multiple search queries
       with different phrasings (e.g., "structural engineer", "occupation structural
       engineer", "hobby die-cast toy") until no new people appear.
    3. For EACH matching person, follow ALL relationship chains described in the
       question (spouse → great-grandchild, etc.) to collect every valid answer.
    4. Keep searching until TWO consecutive searches return zero new entities.
       Do NOT stop after finding the first answer.

    RULES:
    - Never respond with "Cannot be determined" — instead keep trying alternate
      search queries (e.g., search just the birth year, or a name fragment).
    - When multiple people share an attribute, process ALL of them; never ask for
      clarification about which one.
    - For "How many …" questions, return ONLY the count as a plain integer string
      (e.g., ["3"]), never as "Name — 3" or "3 (Alice, Bob)".
    - For questions asking for a person or attribute, list every distinct answer found.

    WORKSPACE: Use add_to_workspace() to record entities and relationships as you
    find them (e.g., "birdwatchers found: Alice Smith, Bob Jones"). Use get_workspace()
    to review accumulated findings and avoid re-searching already-covered entities.
    Recording progress lets you process ALL anchor entities exhaustively without
    losing track across many search steps.
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc=(
            "ALL answers found after exhaustive search. "
            "For 'how many' questions: plain number strings only, e.g. ['3'] not ['Alice — 3']. "
            "For entity/attribute questions: every distinct matching value."
        )
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self._workspace: List[str] = []
        self.react = dspy.ReAct(
            signature=ExhaustiveMultiHopQA,
            tools=[self.search_wiki, self.add_to_workspace, self.get_workspace],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def add_to_workspace(self, note: str) -> str:
        """Save a note to your persistent workspace to track found entities and relationships.
        Use this after each discovery, e.g.: 'Anchor entities (birdwatcher): Alice Smith, Bob Jones'
        or 'Alice Smith grandchildren: Carol, Dave'. Helps avoid forgetting intermediate findings."""
        self._workspace.append(note)
        return f"Saved. Workspace has {len(self._workspace)} note(s)."

    def get_workspace(self) -> str:
        """Retrieve all notes from your persistent workspace. Use this to review what entities
        and relationships you have already found, so you can pick up where you left off and
        ensure you process every anchor entity before finalizing your answer."""
        if not self._workspace:
            return "Workspace is empty — no notes recorded yet."
        entries = "\n".join(f"[{i + 1}] {note}" for i, note in enumerate(self._workspace))
        return f"=== WORKSPACE ({len(self._workspace)} notes) ===\n{entries}"

    def forward(self, question):
        self._workspace = []  # Reset workspace for each new question
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
