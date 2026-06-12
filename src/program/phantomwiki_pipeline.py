import re

import dspy
import src.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import (
    PhantomWikiEntityAgent,
    PhantomWikiReAct,
    _HOW_MANY_RE,
    _PROPER_NAME_RE,
    _SKIP_NAME_PARTS,
)

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"

# Detects a proper person name (two+ consecutive capitalized words) in the question
_NAMED_ENTITY_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
        self.program = PhantomWikiReAct()          # main agent (named-entity questions + fallback)
        self.entity_agent = PhantomWikiEntityAgent()  # per-entity agent (filter-condition questions)
        self.discovery_retrieve = dspy.Retrieve(k=10)  # wider k for entity discovery

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _has_named_entity(self, question: str) -> bool:
        """True if the question directly names a person (proper First Last name)."""
        return bool(_NAMED_ENTITY_RE.search(question))

    def _generate_discovery_queries(self, question: str) -> list:
        """Generate up to 5 diverse search queries for entity discovery from a filter question."""
        queries = [question]

        # Extract YYYY-MM-DD dates and add date-specific queries
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', question)
        if date_match:
            date = date_match.group(1)
            queries.append(date)
            queries.append(f"born {date}")
            queries.append(f"date of birth {date}")

        # Extract attribute filter phrases like "whose hobby is X" or "whose occupation is Y"
        # Use lookahead to stop before trailing verbs like "have", "has", "do", "does"
        attr_match = re.search(
            r'\bwhose\s+(\w+(?:\s+(?:of\s+)?\w+){0,2})\s+is\s+(.+?)(?=\s+(?:have|has|do|does|did|can|will|would|should|could)\b|\?|$)',
            question, re.I
        )
        if attr_match:
            attr = attr_match.group(1).strip()
            value = attr_match.group(2).strip().rstrip('?').strip()
            # Add "attr value" shorthand query
            candidate = f"{attr} {value}"
            if candidate not in queries:
                queries.append(candidate)
            # Also add just the value as a standalone query for better coverage
            if value not in queries and len(value) > 3:
                queries.append(value)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:5]

    def _extract_entity_names(self, passages: list) -> list:
        """Extract person names from passages in relevance order.

        The first proper name in each passage is typically the article subject
        (the entity the passage is about). Returns primary names first (one per
        passage, in passage-relevance order), then secondary names. This ensures
        the most ColBERT-relevant entities are chosen, not alphabetically-first ones.
        """
        seen = set()
        primary_names = []    # First name from each passage (likely article subject)
        secondary_names = []  # Other names mentioned in passages

        for passage in passages:
            first_in_passage = True
            for m in _PROPER_NAME_RE.finditer(passage):
                name = m.group(1)
                parts = name.split()
                # Skip false positives: common words that look capitalised
                if any(p in _SKIP_NAME_PARTS for p in parts):
                    continue
                # Skip very short parts (single letters like "A Smith")
                if any(len(p) < 2 for p in parts):
                    continue
                if name not in seen:
                    seen.add(name)
                    if first_in_passage:
                        primary_names.append(name)
                        first_in_passage = False
                    else:
                        secondary_names.append(name)

        # Primary names (article subjects, most ColBERT-relevant) come first
        return primary_names + secondary_names

    def _reflexive_shortcut(self, question: str) -> list | None:
        """Return [Y] if question is 'What is X of the person whose X is Y?' (self-referential).

        These reflexive questions have the answer embedded in the question itself.
        E.g.: 'What is the date of birth of the person whose date of birth is 0954-03-04?'
        → the answer is '0954-03-04'.
        """
        m = re.match(
            r'what\s+is\s+(?:the\s+)?(.+?)\s+of\s+the\s+person\s+whose\s+(.+?)\s+is\s+(.+?)\??$',
            question.strip(), re.I
        )
        if not m:
            return None
        asked = re.sub(r'^the\s+', '', m.group(1).strip().lower())
        cond = re.sub(r'^the\s+', '', m.group(2).strip().lower())
        val = m.group(3).strip().rstrip('?').strip()
        # Only return the shortcut if asked attribute == filter attribute
        if asked == cond:
            return [val]
        return None

    def _postprocess(self, question: str, answers: list, apply_pattern_g: bool = True) -> dspy.Prediction:
        """Deduplicate answers and optionally apply Pattern G for 'how many' questions."""
        # Dedup while preserving order
        seen: set = set()
        deduped: list = []
        for a in answers:
            val = str(a).strip()
            if val and val not in seen:
                seen.add(val)
                deduped.append(val)

        # Pattern G: for "how many" questions, convert entity-name lists to counts
        if apply_pattern_g and _HOW_MANY_RE.search(question) and deduped:
            numeric = [a for a in deduped if a.lstrip('-').isdigit()]
            non_numeric = [a for a in deduped if not a.lstrip('-').isdigit()]
            if non_numeric and not numeric:
                # Agent returned entity names instead of a count — count them
                deduped = [str(len(non_numeric))]
            elif numeric:
                # Already have numeric answers — discard any stray names
                deduped = numeric

        return dspy.Prediction(answer=deduped)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def forward(self, question: str) -> dspy.Prediction:
        with dspy.context(lm=self.lm, rm=self.rm):

            # --- Reflexive question shortcut ---
            # "What is X of person whose X is Y?" → answer is Y, no LLM needed
            reflexive = self._reflexive_shortcut(question)
            if reflexive is not None:
                return dspy.Prediction(answer=reflexive)

            # --- Path A: named-entity questions ---
            # The question directly names a person → use the main single-stage agent
            if self._has_named_entity(question):
                result = self.program(question=question)
                return self._postprocess(question, result.answer or [], apply_pattern_g=True)

            # --- Path B: filter-condition questions ---
            # No named entity in the question → use programmatic entity discovery + per-entity agents

            # Step 1: run multiple discovery searches
            queries = self._generate_discovery_queries(question)
            all_passages = []
            for q in queries:
                try:
                    results = self.discovery_retrieve(q)
                    all_passages.extend(results.passages)
                except Exception:
                    pass

            # Step 2: extract candidate entity names from passages
            candidates = self._extract_entity_names(all_passages)

            if not candidates:
                # No candidates found — fall back to main agent
                result = self.program(question=question)
                return self._postprocess(question, result.answer or [], apply_pattern_g=True)

            # Step 3: run per-entity agents (breadth cap = 6)
            candidates = candidates[:6]
            all_answers = []
            for entity in candidates:
                try:
                    entity_result = self.entity_agent(question=question, entity=entity)
                    entity_answers = list(entity_result.answer or [])
                    # Mini-Pattern G: for "how many" questions, convert entity names to count per entity
                    if entity_answers and _HOW_MANY_RE.search(question):
                        numeric = [a for a in entity_answers if str(a).lstrip('-').isdigit()]
                        non_numeric = [a for a in entity_answers if not str(a).lstrip('-').isdigit()]
                        if non_numeric and not numeric:
                            # This entity agent returned names instead of count — count them
                            entity_answers = [str(len(non_numeric))]
                        elif numeric:
                            entity_answers = numeric
                    all_answers.extend(entity_answers)
                except Exception:
                    pass

            if not all_answers:
                # Per-entity agents returned nothing — fall back to main agent
                result = self.program(question=question)
                return self._postprocess(question, result.answer or [], apply_pattern_g=True)

            # Per-entity agents produced results — aggregate and postprocess
            # Note: Pattern G is intentionally disabled here because each per-entity agent
            # is already instructed to return numeric counts for "how many" questions.
            # Applying Pattern G globally would miscount across multiple entities.
            return self._postprocess(question, all_answers, apply_pattern_g=False)
