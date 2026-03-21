import re
import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class DecomposeQuestion(dspy.Signature):
    """Decompose a complex multi-hop question into an ordered sequence of simpler single-hop sub-questions. Each sub-question should be answerable independently, and later steps may reference 'the result from step N' as a placeholder for the answer obtained at step N."""

    question: str = dspy.InputField()
    sub_questions: list[str] = dspy.OutputField(desc="ordered list of 1–5 simpler single-hop sub-questions; later steps may reference 'the result from step N' as a placeholder for the answer from step N")


class ResolveWithContext(dspy.Signature):
    """Given a sub-question and previously resolved entities from prior steps, answer the sub-question using the provided context entities as grounding. Focus on finding the specific entities that answer this single-hop query."""

    sub_question: str = dspy.InputField()
    context_entities: list[str] = dspy.InputField(desc="answers/entities resolved from previous steps, used as context to answer the current sub-question")
    original_question: str = dspy.InputField()
    answers: list[str] = dspy.OutputField(desc="list of entities or values that answer the sub-question")


class MergeAnswers(dspy.Signature):
    """Given a question and multiple answer sets discovered through different search strategies, merge and deduplicate all valid answers into a single comprehensive list."""

    question: str = dspy.InputField()
    candidate_answers: list[str] = dspy.InputField(desc="all candidate answers collected across multiple search strategies, may contain duplicates")
    answer: list[str] = dspy.OutputField(desc="deduplicated list of all valid, distinct answers to the question")


class ExtractKnowledge(dspy.Signature):
    """After a ReAct reasoning run, extract structured entity-relationship facts discovered during the run. Focus on concrete facts about named entities, their attributes, and relationships that could help answer related sub-questions."""

    question: str = dspy.InputField()
    strategy: str = dspy.InputField()
    react_answer: list[str] = dspy.InputField()
    entity_facts: list[str] = dspy.OutputField(desc="structured list of discovered entity-relationship facts (e.g. 'Entity X has property Y', 'Entity A is related to Entity B via R')")


class EnrichQuestion(dspy.Signature):
    """Given a search strategy question and a set of already-discovered facts from prior reasoning runs, produce an enriched version of the question that incorporates the known facts as context. This helps subsequent ReAct runs build on intermediate results rather than starting from scratch."""

    question: str = dspy.InputField()
    accumulated_facts: list[str] = dspy.InputField(desc="structured entity-relationship facts discovered by previous ReAct runs")
    enriched_question: str = dspy.OutputField(desc="the original question augmented with known facts as context to guide the next ReAct run")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.decomposer = dspy.ChainOfThought(DecomposeQuestion)
        self.contextual_resolver = dspy.ChainOfThought(ResolveWithContext)
        self.answer_merger = dspy.ChainOfThought(MergeAnswers)
        self.knowledge_extractor = dspy.ChainOfThought(ExtractKnowledge)
        self.question_enricher = dspy.ChainOfThought(EnrichQuestion)

    def _substitute_placeholders(self, sub_question: str, context_entities: list[str]) -> str:
        """Replace 'the result from step N' placeholders with actual resolved entities."""
        def replace_match(m):
            step_num = int(m.group(1))
            idx = step_num - 1
            if 0 <= idx < len(context_entities):
                return context_entities[idx]
            return m.group(0)

        return re.sub(r'the result from step (\d+)', replace_match, sub_question, flags=re.IGNORECASE)

    def forward(self, question):
        # Step 1: Decompose the question into ordered sub-questions
        decomposed = self.decomposer(question=question)
        sub_questions = decomposed.sub_questions

        all_answers = []
        context_entities: list[str] = []

        # Step 2: Resolve each sub-question sequentially
        with dspy.context(rm=self.rm):
            for i, sub_q in enumerate(sub_questions):
                # Substitute any "result from step N" placeholders
                enriched_sub_q = self._substitute_placeholders(sub_q, context_entities)

                # Run the ReAct program on this sub-question
                result = self.program(question=enriched_sub_q)
                step_answers = result.answer if result.answer else []

                all_answers.extend(step_answers)

                # Accumulate answers as context for next steps
                # Use the most relevant answer(s) as context entities
                if step_answers:
                    context_entities.extend(step_answers)

            # Step 3: Run a final pass on the original question enriched with all discovered entities
            if context_entities:
                enriched_original = self.question_enricher(
                    question=question,
                    accumulated_facts=context_entities
                )
                final_query = enriched_original.enriched_question
            else:
                final_query = question

            final_result = self.program(question=final_query)
            all_answers.extend(final_result.answer)

        # Step 4: Merge and deduplicate all collected answers
        merged = self.answer_merger(question=question, candidate_answers=all_answers)
        return dspy.Prediction(answer=merged.answer)
