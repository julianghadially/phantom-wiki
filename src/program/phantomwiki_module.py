import dspy


class AnchorEntityExtraction(dspy.Signature):
    """Identify the anchor attribute being queried in the question and extract all entity names from the passages that match that anchor attribute/value."""

    question: str = dspy.InputField()
    search_passages: str = dspy.InputField(
        desc="passages retrieved from broad PhantomWiki search"
    )
    anchor_attribute: str = dspy.OutputField(
        desc="the attribute and value being queried, e.g. 'occupation: farm manager'"
    )
    matching_entities: list[str] = dspy.OutputField(
        desc="all entity names in the passages that match the anchor attribute/value from the question"
    )


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


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.retrieve_broad = dspy.Retrieve(k=25)
        self.anchor_extractor = dspy.ChainOfThought(AnchorEntityExtraction)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=50,
        )
        self.check_completeness = dspy.ChainOfThought(CompletenessCheck)

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    @staticmethod
    def _normalize_answers(answers: list[str]) -> list[str]:
        """Strip 'Name — Value' format contamination by keeping only the right-hand side."""
        normalized = []
        for item in answers:
            if " — " in item:
                normalized.append(item.split(" — ", 1)[1].strip())
            else:
                normalized.append(item)
        return normalized

    def forward(self, question):
        # Pre-enumeration step: broad retrieval + anchor entity extraction
        broad_results = self.retrieve_broad(question)
        passages_str = "\n\n".join(broad_results.passages)
        anchor_result = self.anchor_extractor(
            question=question, search_passages=passages_str
        )
        matching_entities = anchor_result.matching_entities or []

        if matching_entities:
            entities = matching_entities
            enhanced_q = (
                f"{question}\n[Important: Found {len(entities)} entities matching the "
                f"anchor attribute: {entities}. You MUST trace the complete chain for "
                f"EACH of these entities and report a separate answer for each one.]"
            )
        else:
            enhanced_q = question

        # Pass 1: initial ReAct search using enhanced question
        pass1_result = self.react(question=enhanced_q)
        pass1_answers = self._normalize_answers(pass1_result.answer)

        # Completeness check: gate second pass to avoid regression on single-answer questions
        completeness = self.check_completeness(
            question=question, current_answers=pass1_answers
        )

        if completeness.appears_complete:
            return dspy.Prediction(answer=pass1_answers)

        # Pass 2: targeted follow-up search for missing answers
        anchor_context = (
            f" The anchor entities identified are: {matching_entities}."
            if matching_entities
            else ""
        )
        follow_up_q = (
            f"{question}\n[Context: Already found these answers: {pass1_answers}."
            f"{anchor_context} "
            f"Now find ANY additional answers not yet listed above, by trying different search angles.]"
        )
        pass2_result = self.react(question=follow_up_q)
        pass2_answers = self._normalize_answers(pass2_result.answer)

        # Merge: preserve order, deduplicate
        merged = list(dict.fromkeys(pass1_answers + pass2_answers))
        return dspy.Prediction(answer=merged)
