import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class AnswerExpansionQuery(dspy.Signature):
    """Given a question and the partial answers already found, decide whether more answers
    likely exist and — if so — craft a targeted follow-up question to find them.

    Expansion IS appropriate when:
    - The answer set looks like a small subset of a potentially larger set
      (e.g., only 1-2 people found for a relationship that commonly fans out)
    - The answer contains failure phrases such as "Unknown", "Cannot be determined",
      "no person found", or similar — a differently-phrased search may succeed
    - The question asks for ALL members of a category (occupation, hobby, birth date,
      relationship type) and the initial pass may have missed some

    Expansion is NOT appropriate when:
    - The question is a counting question ("How many …") — the count is already computed
    - The answers look fully exhaustive and cross-checked
    - The question asks for a single unique person and one clear answer was found

    The follow-up question MUST:
    - Explicitly name every already-found answer (so the agent does not repeat them)
    - Ask specifically for additional answers not yet found
    - Rephrase the original retrieval angle when the initial answers were all failures
      (e.g., search by birth-year only instead of full ISO date, or approach via the
      relationship chain from the other direction)
    """

    question: str = dspy.InputField(desc="The original question")
    partial_answers: str = dspy.InputField(
        desc="String representation of the answers already found (may be incomplete or failed)"
    )
    needs_expansion: bool = dspy.OutputField(
        desc="True if additional answers likely exist beyond what was already found"
    )
    followup_question: str = dspy.OutputField(
        desc=(
            "A targeted follow-up question to find ADDITIONAL answers beyond the partial set. "
            "Name the already-found answers explicitly so the agent searches for others. "
            "If the initial answers were all failures, rephrase the query to approach the "
            "information from a different angle (e.g., search by year only, search by "
            "relationship in the opposite direction, etc.)."
        )
    )


class AnswerMerger(dspy.Signature):
    """Merge two answer lists for the same question into one comprehensive, deduplicated answer.

    Rules:
    - Remove exact duplicates (case-insensitive).
    - For entity or attribute questions, take the union of all distinct valid values.
    - Discard failure phrases ("Unknown", "Cannot be determined", "no person found", etc.)
      unless BOTH lists consist entirely of failure phrases.
    - Do NOT sum counts from the two lists; if the question is a counting question and
      one list has a plain integer, prefer the larger integer.
    """

    question: str = dspy.InputField()
    answers_a: str = dspy.InputField(desc="First answer list (string repr of list[str])")
    answers_b: str = dspy.InputField(
        desc="Second answer list from the expansion pass (string repr of list[str])"
    )
    answer: list[str] = dspy.OutputField(
        desc="Merged, comprehensive, deduplicated final answer list"
    )


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        # Primary reasoning chain
        self.program = PhantomWikiReAct()
        # Independent expansion chain (separate weights / trajectory)
        self.program_expand = PhantomWikiReAct()
        # Lightweight modules for the expansion decision and merge
        self.expander = dspy.ChainOfThought(AnswerExpansionQuery)
        self.merger = dspy.ChainOfThought(AnswerMerger)

    def forward(self, question):
        # ── Phase 1: primary multi-hop ReAct chain ──────────────────────────
        with dspy.context(rm=self.rm):
            result = self.program(question=question)

        initial_answers = (
            result.answer if isinstance(result.answer, list) else [result.answer]
        )

        # ── Phase 2: decide whether the answer set looks incomplete ──────────
        expansion = self.expander(
            question=question,
            partial_answers=str(initial_answers),
        )

        if not expansion.needs_expansion:
            return result

        # ── Phase 3: targeted expansion pass ────────────────────────────────
        # Run an independent ReAct chain that starts from a different angle,
        # explicitly aware of what was already found so it searches for the rest.
        with dspy.context(rm=self.rm):
            result2 = self.program_expand(question=expansion.followup_question)

        extra_answers = (
            result2.answer if isinstance(result2.answer, list) else [result2.answer]
        )

        # ── Phase 4: merge both answer sets into one comprehensive response ──
        merged = self.merger(
            question=question,
            answers_a=str(initial_answers),
            answers_b=str(extra_answers),
        )

        return dspy.Prediction(answer=merged.answer)
