import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class AnswerCompletenessChecker(dspy.Signature):
    """Analyze whether the initial answers are likely complete for the given question.
    Many questions in PhantomWiki have MULTIPLE valid answers (e.g., all siblings, all people with a given hobby, all unique count values).
    If the initial answers appear incomplete, generate up to 4 targeted search queries that would retrieve the missing answers.
    Return an empty list if the answers appear complete.
    For aggregation questions like 'how many X does each person with property Y have?', you must search for ALL people with property Y, not just the first found.
    For multi-branch relationship questions (e.g., 'who are all female first cousins once removed'), you must explore ALL relationship paths, not just one.
    """
    question: str = dspy.InputField()
    initial_answers: str = dspy.InputField(desc="Initial answers found so far")
    followup_queries: list[str] = dspy.OutputField(desc="Targeted search queries to find missing answers (empty list if answers are complete, max 4 queries)")


class AnswerSynthesizer(dspy.Signature):
    """Synthesize a comprehensive final answer from all investigation results.
    Extract ONLY the requested values — not person names paired with values.
    For count/number questions, return unique numeric count values as strings (e.g. ['0', '1', '2', '3']).
    For name questions, return all valid matching names.
    For attribute questions (occupation, hobby, etc.), return all found attribute values.
    Remove error strings like 'Cannot be determined', 'not found', 'No person found'.
    Deduplicate the final answer list.
    """
    question: str = dspy.InputField()
    initial_answers: str = dspy.InputField()
    supplemental_results: str = dspy.InputField(desc="Additional search results from targeted follow-up queries")
    answer: list[str] = dspy.OutputField(desc="Complete deduplicated list of all valid answers")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.completeness_checker = dspy.ChainOfThought(AnswerCompletenessChecker)
        self.answer_synthesizer = dspy.ChainOfThought(AnswerSynthesizer)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Pass 1: Initial ReAct investigation
            initial_result = self.program(question=question)
            initial_answers_str = str(initial_result.answer)

            # Pass 2: Completeness check — identify missing answer branches
            completeness = self.completeness_checker(
                question=question,
                initial_answers=initial_answers_str,
            )
            followup_queries = completeness.followup_queries[:4]  # cap at 4

            # Pass 3: Targeted searches for missing answers (direct search, not ReAct)
            supplemental_parts = []
            if followup_queries:
                for query in followup_queries:
                    try:
                        results = self.rm(query)
                        if hasattr(results, 'passages'):
                            supplemental_parts.append(f"Query '{query}':\n" + "\n\n".join(results.passages))
                        else:
                            supplemental_parts.append(f"Query '{query}':\n" + str(results))
                    except Exception:
                        pass

            # Pass 4: Synthesis — combine all findings with format normalization
            if supplemental_parts:
                final = self.answer_synthesizer(
                    question=question,
                    initial_answers=initial_answers_str,
                    supplemental_results="\n\n---\n\n".join(supplemental_parts),
                )
                return dspy.Prediction(answer=final.answer)
            else:
                # No follow-up needed — still normalize format
                final = self.answer_synthesizer(
                    question=question,
                    initial_answers=initial_answers_str,
                    supplemental_results="(none)",
                )
                return dspy.Prediction(answer=final.answer)
