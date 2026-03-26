import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class IdentifyMissingEntityQueries(dspy.Signature):
    """You are analyzing whether a multi-hop question has been fully answered.
    Many questions have MULTIPLE correct answers because there are multiple valid entity chains.
    For example, 'occupation of the brother of the grandson of X' requires checking ALL grandsons
    and ALL their brothers — not just one path. Similarly, 'how many sisters does the person
    whose hobby is Y have' requires finding ALL people with hobby Y.

    Identify if the initial answer likely missed valid answer paths, and generate targeted search
    queries for each unchecked entity or entity group."""
    question: str = dspy.InputField()
    initial_answer: list[str] = dspy.InputField()
    reasoning: str = dspy.InputField(desc="The reasoning trace from the initial ReAct pass")

    needs_more_search: bool = dspy.OutputField(desc="True if there are likely additional correct answers not yet found")
    additional_queries: list[str] = dspy.OutputField(desc="Specific search queries for unchecked entities (e.g. each sibling/cousin/person-with-hobby not yet individually looked up). Max 10 queries.")
    target_attribute: str = dspy.OutputField(desc="The final attribute to extract from each search result (e.g. 'occupation', 'number of sisters', 'hobby')")


class ExtractTargetAttribute(dspy.Signature):
    """Extract a specific attribute value from wiki search passages to answer a question.
    Return only the raw value (e.g. 'toxicologist', '3') not a full sentence."""
    question: str = dspy.InputField()
    target_attribute: str = dspy.InputField()
    passages: str = dspy.InputField()
    existing_answers: list[str] = dspy.InputField(desc="Answers already found — only return NEW values not in this list")

    new_answers: list[str] = dspy.OutputField(desc="New attribute values found in the passages not already in existing_answers. Empty list if nothing new.")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.gap_finder = dspy.ChainOfThought(IdentifyMissingEntityQueries)
        self.attr_extractor = dspy.ChainOfThought(ExtractTargetAttribute)
        self.completion_retrieve = dspy.Retrieve(k=7)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Initial ReAct pass
            initial = self.program(question=question)
            all_answers = list(initial.answer) if isinstance(initial.answer, list) else [initial.answer]
            reasoning = getattr(initial, 'reasoning', '') or ''

            # Targeted completion loop: find unchecked entity paths
            try:
                gap = self.gap_finder(
                    question=question,
                    initial_answer=all_answers,
                    reasoning=reasoning,
                )
                if gap.needs_more_search and gap.additional_queries:
                    for query in gap.additional_queries[:10]:
                        try:
                            results = self.completion_retrieve(query)
                            passages = "\n\n".join(results.passages)
                            extraction = self.attr_extractor(
                                question=question,
                                target_attribute=gap.target_attribute,
                                passages=passages,
                                existing_answers=all_answers,
                            )
                            if extraction.new_answers:
                                all_answers.extend(extraction.new_answers)
                        except Exception:
                            continue
            except Exception:
                pass

            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for a in all_answers:
                key = str(a).strip().lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(a)

            return dspy.Prediction(answer=deduped)
