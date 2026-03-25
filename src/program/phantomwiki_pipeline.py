import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct, QuestionDecomposer, AnswerValidator

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.decomposer = QuestionDecomposer()
        self.program = PhantomWikiReAct()
        self.validator = AnswerValidator()

    def forward(self, question):
        # Stage 1: Decompose question into seed queries and answer metadata
        decomposed = self.decomposer(question=question)
        seed_queries = decomposed.seed_queries
        is_multi_answer = decomposed.is_multi_answer
        answer_format_hint = decomposed.answer_format_hint

        # Use the first seed query as the primary question for retrieval-oriented reasoning
        primary_query = seed_queries[0] if seed_queries else question

        # Stage 2: Run ReAct with retry-resilient retrieval and multi-answer awareness
        with dspy.context(rm=self.rm):
            raw_result = self.program(
                question=primary_query,
                is_multi_answer=is_multi_answer,
                answer_format_hint=answer_format_hint,
            )

        # Stage 3: Validate, clean, and de-duplicate the answer list
        validated = self.validator(
            question=question,
            raw_answer=raw_result.answer,
            answer_format_hint=answer_format_hint,
        )

        return dspy.Prediction(answer=validated.answer)
