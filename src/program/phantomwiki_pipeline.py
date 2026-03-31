import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class FollowUpInvestigation(dspy.Signature):
    """You are given a question and a list of answers already found via one investigation pass.
    Explore alternative relationship chains and paths NOT yet investigated to find additional answers.
    Treat already_found as a non-exhaustive partial result — there may be more valid answers
    reachable via different paths, relationships, or entity traversals that were not explored before."""

    question: str = dspy.InputField()
    already_found: list[str] = dspy.InputField(desc="answers discovered so far; treat as partial/non-exhaustive")
    answer: list[str] = dspy.OutputField(desc="additional answers found via unexplored paths")


class AnswerNormalizerSignature(dspy.Signature):
    """Normalize a list of candidate answers to the correct output format.

    Rules (apply in order):
    1. If answers contain 'Name: count' pairs (e.g. 'Alice Smith: 3', 'Bob Jones: 0'), extract ONLY the
       numeric count part from each entry, then return the UNIQUE numeric values as plain strings (e.g. ['0', '3']).
    2. Remove any entries that are error/uncertainty messages containing phrases like 'Cannot be determined',
       'cannot determine', 'No person found', 'not found', 'not present', 'please confirm', 'Near match', etc.
    3. For answers that are already clean atomic values (plain names, plain numbers, plain hobby/occupation strings
       without explanatory text), keep them unchanged.
    4. IMPORTANT: If after applying all rules the list would be empty, return the original raw_answers unchanged.
    """
    question: str = dspy.InputField()
    raw_answers: list[str] = dspy.InputField(desc="raw combined answer list from ReAct passes, may contain format artifacts")
    normalized_answers: list[str] = dspy.OutputField(desc="cleaned answer list with format artifacts and error strings removed")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.retrieve = dspy.Retrieve(k=7)
        self.followup_react = dspy.ReAct(
            signature=FollowUpInvestigation,
            tools=[self._search_wiki],
            max_iters=25,
        )
        self.answer_normalizer = dspy.ChainOfThought(AnswerNormalizerSignature)

    def _search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            result1 = self.program(question=question)
            result2 = self.followup_react(question=question, already_found=result1.answer)
            combined = list(dict.fromkeys(result1.answer + [a for a in result2.answer if a not in result1.answer]))
            normalized = self.answer_normalizer(question=question, raw_answers=combined)
            final = normalized.normalized_answers if normalized.normalized_answers else combined
            return dspy.Prediction(answer=final)
