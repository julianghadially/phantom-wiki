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


class SeedEnumerationSignature(dspy.Signature):
    """You are given a question and a list of answers/entities already found.
    Exhaustively enumerate ALL intermediate entities relevant to answering this question
    (e.g., all friends of X, all people with occupation Y, all people with hobby Z).
    Be exhaustive — list every entity you can find, not just the first one."""

    question: str = dspy.InputField()
    already_found: list[str] = dspy.InputField()
    seed_entities: list[str] = dspy.OutputField(desc="ALL intermediate entities relevant to answering this question (e.g., all friends of X, all people with occupation Y, all people with hobby Z) — be exhaustive, list every one you can find")


class MultiSeedAnswerExtractorSignature(dspy.Signature):
    """Extract additional answers from retrieved passages for multiple seed entities.
    Focus on finding new atomic answer values not already in the already_found list."""

    question: str = dspy.InputField()
    retrieved_passages: str = dspy.InputField(desc="passages retrieved for each seed entity")
    already_found: list[str] = dspy.InputField()
    additional_answers: list[str] = dspy.OutputField(desc="new atomic answer values (names, numbers, occupations) not already in already_found — no explanatory text, no 'Name: value' pairs")


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
        self.retrieve_k10 = dspy.Retrieve(k=10)
        self.seed_enumerator = dspy.ReAct(
            SeedEnumerationSignature,
            tools=[self._search_wiki_k10],
            max_iters=20,
        )
        self.multi_seed_extractor = dspy.ChainOfThought(MultiSeedAnswerExtractorSignature)

    def _search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def _search_wiki_k10(self, query: str) -> str:
        """Search the PhantomWiki corpus with higher recall (k=10). Returns relevant passages."""
        results = self.retrieve_k10(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            result1 = self.program(question=question)
            result2 = self.followup_react(question=question, already_found=result1.answer)
            combined = list(dict.fromkeys(result1.answer + [a for a in result2.answer if a not in result1.answer]))

            # Pass 3: Seed Entity Fan-Out — exhaustively enumerate intermediate entities
            try:
                seed_result = self.seed_enumerator(question=question, already_found=combined)
                seed_entities = seed_result.seed_entities[:6]  # cap at 6 to avoid overload

                all_passages_parts = []
                for seed_entity in seed_entities:
                    passages = self.retrieve_k10(seed_entity + " " + question[:60])
                    all_passages_parts.append("\n\n".join(passages.passages))

                all_passages = "\n\n---\n\n".join(all_passages_parts)
                all_passages = all_passages[:4000]  # truncate to ~4000 chars

                extracted = self.multi_seed_extractor(
                    question=question,
                    retrieved_passages=all_passages,
                    already_found=combined,
                )
                combined = list(dict.fromkeys(combined + [a for a in extracted.additional_answers if a not in combined]))
            except Exception:
                pass  # fall back to combined from passes 1 & 2 if fan-out fails

            normalized = self.answer_normalizer(question=question, raw_answers=combined)
            final = normalized.normalized_answers if normalized.normalized_answers else combined
            return dspy.Prediction(answer=final)
