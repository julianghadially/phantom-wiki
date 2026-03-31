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


class PathDecomposerSignature(dspy.Signature):
    """Given a question and a list of partial answers found so far, identify the specific entity/relationship
    paths that have NOT yet been investigated. Return 1–4 concrete investigation targets that are most likely
    to yield additional answers to the question. Each path should be a specific, actionable description such as
    'count friends of Darwin Laughlin', 'explore paternal grandparent of Sam Highsmith', or
    'find occupation of Alice Turner's sibling'. Focus on paths directly relevant to answering the question
    that the partial answers suggest were skipped or incomplete."""

    question: str = dspy.InputField()
    partial_answers: list[str] = dspy.InputField(desc="answers already discovered; used to infer which paths were explored")
    unexplored_paths: list[str] = dspy.OutputField(desc="1–4 specific entity/relationship paths to investigate next")


class TargetedInvestigation(dspy.Signature):
    """You are given a question and a specific investigation target (an entity/relationship path to follow).
    Focus exclusively on exploring that target path to find answers relevant to the overall question.
    Use available search tools to look up the specific entities and relationships described in the
    investigation target. Return all answers discovered along this path."""

    question: str = dspy.InputField()
    investigation_target: str = dspy.InputField(desc="the specific entity/relationship path to investigate")
    answer: list[str] = dspy.OutputField(desc="answers discovered by following the specified investigation target path")


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
        self.path_decomposer = dspy.ChainOfThought(PathDecomposerSignature)
        self.micro_react = dspy.ReAct(
            signature=TargetedInvestigation,
            tools=[self._search_wiki],
            max_iters=8,
        )
        self.answer_normalizer = dspy.ChainOfThought(AnswerNormalizerSignature)

    def _search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Pass 1: primary ReAct investigation
            result1 = self.program(question=question)

            # Pass 1.5: PathDecomposer + Sequential Micro-Investigation
            # Identify unexplored entity/relationship paths from partial answers
            decomposed = self.path_decomposer(question=question, partial_answers=result1.answer)
            unexplored_paths = decomposed.unexplored_paths[:4]  # cap at 4 paths

            # Run targeted micro-investigations for each unexplored path
            micro_answers = []
            for path in unexplored_paths:
                micro_result = self.micro_react(question=question, investigation_target=path)
                micro_answers.extend(micro_result.answer)

            # Deduplicate micro_answers and merge with Pass 1 answers
            pass1_set = set(result1.answer)
            unique_micro = list(dict.fromkeys(a for a in micro_answers if a not in pass1_set))
            enriched_answers = result1.answer + unique_micro

            # Pass 2: FollowUpInvestigation enriched with micro-investigation results
            result2 = self.followup_react(question=question, already_found=enriched_answers)

            # Combine all answers with deduplication
            enriched_set = set(enriched_answers)
            combined = enriched_answers + [a for a in result2.answer if a not in enriched_set]
            combined = list(dict.fromkeys(combined))

            # Post-processing: normalize answer format
            normalized = self.answer_normalizer(question=question, raw_answers=combined)
            final = normalized.normalized_answers if normalized.normalized_answers else combined
            return dspy.Prediction(answer=final)
