import re

import dspy


class CompletenessCheck(dspy.Signature):
    """Assess whether the current set of answers fully satisfies the question, or if additional answers are likely missing."""

    question: str = dspy.InputField()
    current_answers: list[str] = dspy.InputField(
        desc="answers found so far for the question"
    )
    appears_complete: bool = dspy.OutputField(
        desc="True if the current answers appear to fully answer the question; False if additional answers are likely missing"
    )
    follow_up_hint: str = dspy.OutputField(
        desc="a reformulated search hint to find missing entities if the answer appears incomplete; empty string if complete"
    )


# Pre-compiled regex for _normalize_answers:
# Matches "Firstname Lastname: value" (at least two capitalized words before colon)
_RE_PERSON_NAME_PREFIX = re.compile(
    r"^[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+:\s+(.+)$"
)
# Matches "relationship-word: Name" (one or more lowercase/hyphenated words before colon)
_RE_RELATIONSHIP_PREFIX = re.compile(
    r"^[a-z][a-z-]*(?:\s+[a-z][a-z-]*)*:\s+(.+)$"
)


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=50,
        )
        self.completeness_check = dspy.ChainOfThought(CompletenessCheck)

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    @staticmethod
    def _normalize_answers(answers: list[str]) -> list[str]:
        """Strip format contamination from answers.

        Handles:
        1. 'Name — Value' em-dash patterns → keep right side
        2. 'Firstname Lastname: value' colon patterns (person name before colon) → keep right side
        3. 'relationship-word: Name' colon patterns (lowercase/hyphenated prefix) → keep right side
        """
        normalized = []
        for item in answers:
            item = item.strip()
            # 1. Strip em-dash "Name — Value" pattern
            if " — " in item:
                item = item.split(" — ", 1)[1].strip()
            # 2. Strip "Firstname Lastname: value" person-name colon pattern
            m = _RE_PERSON_NAME_PREFIX.match(item)
            if m:
                item = m.group(1).strip()
            else:
                # 3. Strip "relationship-word: Name" lowercase prefix pattern
                m2 = _RE_RELATIONSHIP_PREFIX.match(item)
                if m2:
                    item = m2.group(1).strip()
            normalized.append(item)
        return normalized

    def forward(self, question):
        # Pass 1: initial ReAct search
        pass1_result = self.react(question=question)
        pass1_answers = self._normalize_answers(pass1_result.answer)

        # Check completeness to decide whether a second pass is needed
        completeness = self.completeness_check(
            question=question,
            current_answers=pass1_answers,
        )

        if completeness.appears_complete:
            return dspy.Prediction(answer=pass1_answers)

        # Pass 2: targeted follow-up embedding already-found answers and the hint
        follow_up_q = (
            f"{question}\n"
            f"[Already found: {pass1_answers}. "
            f"Hint for missing answers: {completeness.follow_up_hint}. "
            f"Search exhaustively for the remaining answers.]"
        )
        pass2_result = self.react(question=follow_up_q)
        pass2_answers = self._normalize_answers(pass2_result.answer)

        # Merge Pass 1 + Pass 2 with order-preserving deduplication
        merged = list(dict.fromkeys(pass1_answers + pass2_answers))
        return dspy.Prediction(answer=merged)
