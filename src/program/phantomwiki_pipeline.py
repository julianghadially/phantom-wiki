import dspy
import json
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import (
    PhantomWikiReAct,
    EntityEnumeratorReAct,
    ParallelAttributeFetcher,
    AnswerSynthesizer,
)

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.fallback_react = PhantomWikiReAct()
        self.entity_enumerator = EntityEnumeratorReAct()
        self.attribute_fetcher = ParallelAttributeFetcher()
        self.answer_synthesizer = dspy.ChainOfThought(AnswerSynthesizer)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Stage 1 — EntityEnumeratorReAct: identify ALL target entities
            enum_result = self.entity_enumerator(question=question)
            all_target_entities = enum_result.all_target_entities

            # Fallback: if Stage 1 returns no entities, use the original ReAct agent
            if not all_target_entities:
                return self.fallback_react(question=question)

            # Stage 2 — ParallelAttributeFetcher: fetch bare attribute value per entity
            entity_attribute_pairs = self.attribute_fetcher(
                question=question,
                all_target_entities=all_target_entities,
            )

            # Stage 3 — AnswerSynthesizer: deduplicate and normalize into final answer list
            pairs_json = json.dumps(entity_attribute_pairs)
            synthesis_result = self.answer_synthesizer(
                question=question,
                entity_attribute_pairs=pairs_json,
            )

            return dspy.Prediction(answer=synthesis_result.answer)
