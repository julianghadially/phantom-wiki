import dspy


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.notes: list[str] = []
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki, self.write_note, self.read_notes],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def write_note(self, note: str) -> str:
        """Write a persistent note to the scratchpad. Use this to record discovered entities,
        intermediate facts, or a to-do list of items still needing processing.
        E.g., 'Brothers of X: Carmine, Damon, Sal — need occupations for all three.'"""
        self.notes.append(note)
        return f"Note saved. Total notes: {len(self.notes)}"

    def read_notes(self) -> str:
        """Read all notes from the scratchpad. Use this to recall previously found entities
        or check which items still need to be processed before finishing."""
        if not self.notes:
            return "No notes yet."
        return "\n".join(f"[{i+1}] {n}" for i, n in enumerate(self.notes))

    def forward(self, question):
        self.notes = []
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
