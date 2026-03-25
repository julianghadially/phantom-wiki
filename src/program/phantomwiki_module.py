import time
import dspy


class QuestionDecomposerSignature(dspy.Signature):
    """Decompose a question into diverse search queries and metadata to guide multi-hop retrieval."""

    question: str = dspy.InputField()
    seed_queries: list[str] = dspy.OutputField(
        desc="2-3 diverse query phrasings targeting the root entity in the question, so if one phrasing times out, the others can be tried"
    )
    is_multi_answer: bool = dspy.OutputField(
        desc="True if the question is likely to have multiple valid answer chains (i.e., many people share the described attribute)"
    )
    answer_format_hint: str = dspy.OutputField(
        desc="Describes the expected output format, e.g., 'a plain hobby name with no person name', 'an integer count', 'a full person name'"
    )


class AnswerValidatorSignature(dspy.Signature):
    """Validate and clean the raw answer list: strip accidentally included context artifacts and ensure completeness given the multi-answer flag."""

    question: str = dspy.InputField()
    raw_answer: list[str] = dspy.InputField(desc="The raw answer list from the ReAct module")
    answer_format_hint: str = dspy.InputField(
        desc="Expected output format hint, e.g., 'a plain hobby name with no person name'"
    )
    answer: list[str] = dspy.OutputField(
        desc="Corrected and de-duplicated answer list. Strip any accidentally included context like 'Person — attribute' to just 'attribute'. If is_multi_answer suggests more chains should have been explored, note missing answers may exist."
    )


class QuestionDecomposer(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(QuestionDecomposerSignature)

    def forward(self, question):
        return self.cot(question=question)


class AnswerValidator(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(AnswerValidatorSignature)

    def forward(self, question, raw_answer, answer_format_hint):
        return self.cot(
            question=question,
            raw_answer=raw_answer,
            answer_format_hint=answer_format_hint,
        )


class PhantomWikiReActSignature(dspy.Signature):
    """Answer the question by iteratively searching the PhantomWiki corpus. Use is_multi_answer to know whether to keep searching for additional answer chains, and answer_format_hint to format the final answer correctly."""

    question: str = dspy.InputField()
    is_multi_answer: bool = dspy.InputField(
        desc="True if the question likely has multiple valid answer chains; keep searching until all chains are exhausted"
    )
    answer_format_hint: str = dspy.InputField(
        desc="Expected output format for each answer item, e.g., 'a plain hobby name with no person name'"
    )
    answer: list[str] = dspy.OutputField(
        desc="List of answer strings in the format described by answer_format_hint"
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature=PhantomWikiReActSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        max_attempts = 4
        last_exception = None
        for attempt in range(max_attempts):
            try:
                results = self.retrieve(query)
                if results and results.passages:
                    return "\n\n".join(results.passages)
                # Empty results — sleep and retry
                if attempt < max_attempts - 1:
                    time.sleep(2)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    time.sleep(2)
        error_detail = str(last_exception) if last_exception else "empty results"
        return f"No results found for query: {query} (reason: {error_detail})"

    def forward(self, question, is_multi_answer=False, answer_format_hint=""):
        result = self.react(
            question=question,
            is_multi_answer=is_multi_answer,
            answer_format_hint=answer_format_hint,
        )
        return dspy.Prediction(answer=result.answer)
