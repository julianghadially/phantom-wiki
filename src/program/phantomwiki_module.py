import dspy

MAX_EXPANSION_PASSES = 4


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


class SearchAngleGenerator(dspy.Signature):
    """Given the original question, the answers found so far, and the search strategies already tried, generate a new search angle and a reformulated question that targets a different slice of the answer space not yet covered."""

    question: str = dspy.InputField(
        desc="the original question to answer"
    )
    found_answers: list[str] = dspy.InputField(
        desc="answers discovered in previous search passes"
    )
    strategies_used: list[str] = dspy.InputField(
        desc="descriptions of search strategies already attempted, to avoid repetition"
    )
    search_angle: str = dspy.OutputField(
        desc="a short description of the new search angle to explore (e.g., 'search by reverse relationship', 'look up by attribute type X', 'enumerate subset Y')"
    )
    reformulated_question: str = dspy.OutputField(
        desc="a reformulated version of the original question that targets the new search angle and instructs the agent to find entities not yet in found_answers"
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature="question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=50,
        )
        self.check_completeness = dspy.ChainOfThought(CompletenessCheck)
        self.generate_angle = dspy.ChainOfThought(SearchAngleGenerator)

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

        # Completeness check: gate expansion loop to avoid regression on single-answer questions
        completeness = self.check_completeness(
            question=question, current_answers=pass1_answers
        )

        if completeness.appears_complete:
            return dspy.Prediction(answer=pass1_answers)

        # Multi-pass iterative expansion loop with convergence detection
        all_answers = list(pass1_answers)
        strategies_used = ["initial broad search"]

        for _ in range(MAX_EXPANSION_PASSES):
            # Generate a new search angle targeting a different slice of the answer space
            angle_result = self.generate_angle(
                question=question,
                found_answers=all_answers,
                strategies_used=strategies_used,
            )
            reformulated_question = angle_result.reformulated_question
            strategies_used.append(angle_result.search_angle)

            # Run a targeted ReAct pass with the reformulated question
            pass_result = self.react(question=reformulated_question)
            new_answers = self._normalize_answers(pass_result.answer)

            # Convergence detection: stop if no new entities are discovered
            newly_found = set(new_answers) - set(all_answers)
            if not newly_found:
                break

            # Merge new answers (order-preserving deduplication)
            all_answers = list(dict.fromkeys(all_answers + new_answers))

        return dspy.Prediction(answer=all_answers)
