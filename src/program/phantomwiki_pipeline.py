import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import (
    AnswerGapFinder,
    PhantomWikiReAct,
    QuestionDecomposer,
)

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    """Three-stage pipeline for exhaustive answer accumulation on PhantomWiki.

    Stage 1 – QuestionDecomposer: classifies the question and produces anchor
              entities plus a structured search plan.
    Stage 2 – PhantomWikiReAct (ReAct): iteratively retrieves evidence using
              two tools (broad search and direct entity lookup) and outputs
              candidate answers with an exploration summary.
    Stage 3 – AnswerGapFinder: inspects candidates for gaps, issues up to 5
              extra retrieval queries, and finalises the answer list.
    """

    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))

        # Stage 1: question decomposition
        self.decomposer = dspy.ChainOfThought(QuestionDecomposer)

        # Stage 2: iterative ReAct with two search tools
        self.program = PhantomWikiReAct()

        # Stage 3: answer gap analysis and finalisation
        self.gap_finder = dspy.ChainOfThought(AnswerGapFinder)

        # Extra retriever used during gap-filling loop
        self.retrieve_gap = dspy.Retrieve(k=15)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # ------------------------------------------------------------------
            # Stage 1 – Question Decomposer
            # ------------------------------------------------------------------
            decomposed = self.decomposer(question=question)
            anchor_entities = decomposed.anchor_entities
            search_plan = decomposed.search_plan

            # ------------------------------------------------------------------
            # Stage 2 – Iterative ReAct (exhaustive candidate accumulation)
            # ------------------------------------------------------------------
            stage2 = self.program(
                question=question,
                search_plan=search_plan,
                anchor_entities=anchor_entities,
            )
            candidate_answers = list(stage2.candidate_answers)
            exploration_summary = stage2.exploration_summary

            # ------------------------------------------------------------------
            # Stage 3 – Answer Gap Finder (first pass)
            # ------------------------------------------------------------------
            gap = self.gap_finder(
                question=question,
                candidate_answers=candidate_answers,
                exploration_summary=exploration_summary,
            )
            missing_searches = gap.missing_searches or []
            final_answer = gap.final_answer

            # ------------------------------------------------------------------
            # Gap-filling loop: execute up to 5 additional retrieval queries,
            # append newly discovered unique entities, then re-run gap finder.
            # ------------------------------------------------------------------
            if missing_searches:
                extra_queries = missing_searches[:5]
                new_entities: list[str] = []

                for query in extra_queries:
                    results = self.retrieve_gap(query)
                    # Treat the query itself as a candidate entity (gap finder
                    # typically surfaces entity names as missing searches), and
                    # record a snippet of retrieved evidence for re-evaluation.
                    if query not in candidate_answers and query not in new_entities:
                        new_entities.append(query)
                    # Also harvest the leading token of each retrieved passage
                    # as a lightweight entity hint.
                    for passage in results.passages:
                        first_line = passage.split("\n")[0].strip()
                        if (
                            first_line
                            and first_line not in candidate_answers
                            and first_line not in new_entities
                        ):
                            new_entities.append(first_line)

                if new_entities:
                    candidate_answers = candidate_answers + new_entities
                    exploration_summary = (
                        exploration_summary
                        + "\nGap-filling searches surfaced: "
                        + ", ".join(new_entities[:10])
                    )

                # Re-run gap finder once more with augmented candidate set.
                gap = self.gap_finder(
                    question=question,
                    candidate_answers=candidate_answers,
                    exploration_summary=exploration_summary,
                )
                final_answer = gap.final_answer

            return dspy.Prediction(answer=final_answer)
