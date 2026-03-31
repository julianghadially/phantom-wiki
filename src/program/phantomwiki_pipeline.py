import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class GapAnalyzer(dspy.Signature):
    """Identify specific named individuals or entities that were mentioned in the trajectory as intermediate nodes (e.g., great-grandchildren, cousins, friends, siblings) but whose final attribute (occupation, count, DOB, etc.) was NOT retrieved and included in initial_answers. These are the gaps that need targeted follow-up."""

    question: str = dspy.InputField()
    initial_answers: list[str] = dspy.InputField()
    trajectory_summary: str = dspy.InputField(
        desc="The agent's thought/observation log from pass 1, truncated to ~2000 chars"
    )
    pending_entities: list[str] = dspy.OutputField(
        desc="Specific named entities encountered during investigation but whose final attribute/count was NOT yet looked up to produce an answer"
    )
    investigation_complete: bool = dspy.OutputField(
        desc="Whether the initial investigation fully answered the question"
    )


class EntityTargetedFocusSignature(dspy.Signature):
    """Perform a focused investigation on a specific entity to find the answer to the question."""

    question: str = dspy.InputField()
    target_entity: str = dspy.InputField(desc="The specific entity to investigate")
    context_from_prior_search: str = dspy.InputField(
        desc="Context gathered from previous search passes"
    )
    partial_answers: list[str] = dspy.OutputField(
        desc="Answers found for this specific entity"
    )


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.retrieve = dspy.Retrieve(k=7)
        self.program = PhantomWikiReAct()
        self.gap_analyzer = dspy.ChainOfThought(GapAnalyzer)
        self.entity_react = dspy.ReAct(
            EntityTargetedFocusSignature,
            tools=[self.search_wiki],
            max_iters=12,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Pass 1: initial ReAct — call inner react directly to access trajectory fields
            react_pred = self.program.react(question=question)
            initial_answers = (
                react_pred.answer
                if isinstance(react_pred.answer, list)
                else [react_pred.answer]
            )

            # Extract trajectory summary by joining thought/observation fields
            trajectory_parts = []
            for key, val in react_pred.items():
                if key != "answer" and val:
                    trajectory_parts.append(f"{key}: {str(val)[:200]}")
            summary = "\n".join(trajectory_parts)[:2000]

            # Gap analysis: identify entities whose attributes were NOT yet retrieved
            gap_result = self.gap_analyzer(
                question=question,
                initial_answers=initial_answers,
                trajectory_summary=summary,
            )
            pending_entities = (
                gap_result.pending_entities
                if isinstance(gap_result.pending_entities, list)
                else []
            )
            investigation_complete = gap_result.investigation_complete

            # If investigation is incomplete, run targeted entity follow-up passes
            all_answers = list(initial_answers)
            if not investigation_complete and pending_entities:
                for entity in pending_entities[:4]:
                    try:
                        entity_result = self.entity_react(
                            question=question,
                            target_entity=entity,
                            context_from_prior_search=summary,
                        )
                        partial = (
                            entity_result.partial_answers
                            if isinstance(entity_result.partial_answers, list)
                            else []
                        )
                        all_answers.extend(partial)
                    except Exception:
                        continue

            # Deduplicate preserving insertion order
            seen = set()
            deduped = []
            for ans in all_answers:
                norm = str(ans).lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    deduped.append(ans)

            return dspy.Prediction(answer=deduped)
