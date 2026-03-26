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


class EntityFrontierExtractor(dspy.Signature):
    """From the reasoning trace and partial answers, identify intermediate entities in the relationship chain that were found but not fully explored, and generate targeted search queries to find their relatives (children, siblings, etc.)."""

    question: str = dspy.InputField()
    current_answers: list[str] = dspy.InputField(desc="answers found so far")
    pass_reasoning: str = dspy.InputField(
        desc="reasoning trace describing which entities were found and which relationships were traversed"
    )
    frontier_entities: list[str] = dspy.OutputField(
        desc="specific named entities found in the chain that still need their relatives explored (e.g., grandchildren who haven't had their own children searched yet)"
    )
    targeted_search_hints: list[str] = dspy.OutputField(
        desc="concrete search queries like 'son of [name]', 'children of [name]', '[name] family' to find next-level entities"
    )
    needs_more_exploration: bool = dspy.OutputField(
        desc="True if the answer set is clearly incomplete and specific entities remain unexplored"
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=50,
        )
        self.frontier_extractor = dspy.ChainOfThought(EntityFrontierExtractor)

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
        # Pass 1: initial ReAct search
        pass1_result = self.react(question=question)
        pass1_answers = self._normalize_answers(pass1_result.answer)

        # Extract frontier entities from Pass 1 to guide targeted follow-up
        pass1_reasoning = (
            pass1_result.get("reasoning", "")
            if hasattr(pass1_result, "get")
            else getattr(pass1_result, "reasoning", "")
        )
        frontier1 = self.frontier_extractor(
            question=question,
            current_answers=pass1_answers,
            pass_reasoning=pass1_reasoning,
        )

        if not frontier1.needs_more_exploration:
            return dspy.Prediction(answer=pass1_answers)

        # Pass 2: targeted search using frontier entities and concrete search hints
        follow_up_q = (
            f"{question}\n"
            f"[Already found: {pass1_answers}. "
            f"Key intermediate entities to explore further: {frontier1.frontier_entities}. "
            f"Suggested searches: {frontier1.targeted_search_hints}. "
            f"Search exhaustively for relatives of each frontier entity.]"
        )
        pass2_result = self.react(question=follow_up_q)
        pass2_answers = self._normalize_answers(pass2_result.answer)

        # Merge Pass 1 + Pass 2 with deduplication
        merged_12 = list(dict.fromkeys(pass1_answers + pass2_answers))

        # Extract frontier entities from Pass 2 for potential Pass 3
        pass2_reasoning = (
            pass2_result.get("reasoning", "")
            if hasattr(pass2_result, "get")
            else getattr(pass2_result, "reasoning", "")
        )
        frontier2 = self.frontier_extractor(
            question=question,
            current_answers=merged_12,
            pass_reasoning=pass2_reasoning,
        )

        if not frontier2.needs_more_exploration:
            return dspy.Prediction(answer=merged_12)

        # Pass 3: final targeted search (max 3 passes total)
        follow_up_q3 = (
            f"{question}\n"
            f"[Already found: {merged_12}. "
            f"Key intermediate entities to explore further: {frontier2.frontier_entities}. "
            f"Suggested searches: {frontier2.targeted_search_hints}. "
            f"Search exhaustively for relatives of each frontier entity.]"
        )
        pass3_result = self.react(question=follow_up_q3)
        pass3_answers = self._normalize_answers(pass3_result.answer)

        # Merge all three passes with deduplication
        merged_all = list(dict.fromkeys(merged_12 + pass3_answers))
        return dspy.Prediction(answer=merged_all)
