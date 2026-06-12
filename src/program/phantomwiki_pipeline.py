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
        """Generate up to 4 diverse search queries for entity discovery from a filter question."""
        queries = [question]

        # Extract YYYY-MM-DD dates and add date-specific queries
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', question)
        if date_match:
            date = date_match.group(1)
            queries.append(date)
            queries.append(f"born {date}")
            queries.append(f"date of birth {date}")

        # Extract attribute filter phrases like "whose hobby is X" or "whose occupation is Y"
        attr_match = re.search(
            r'\bwhose\s+(\w+(?:\s+\w+){0,2})\s+is\s+([^?]+?)(?:\?|$)', question, re.I
        )
        if attr_match:
            attr = attr_match.group(1).strip()
            value = attr_match.group(2).strip().rstrip('?').strip()
            candidate = f"{attr} {value}"
            if candidate not in queries:
                queries.append(candidate)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:4]

    def _extract_entity_names(self, passages: list) -> list:
        """Extract person names from a list of passage strings via regex. Returns sorted deduped list."""
        names = set()
        for passage in passages:
            for m in _PROPER_NAME_RE.finditer(passage):
                name = m.group(1)
                parts = name.split()
                # Skip false positives: common words that look capitalised
                if any(p in _SKIP_NAME_PARTS for p in parts):
                    continue
                # Skip very short parts (single letters like "A Smith")
                if any(len(p) < 2 for p in parts):
                    continue
                names.add(name)
        return sorted(names)

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

            # Step 3: run per-entity agents (breadth cap = 4 per client guidance)
            candidates = candidates[:4]
            all_answers = []
            for entity in candidates:
                try:
                    entity_result = self.entity_agent(question=question, entity=entity)
                    all_answers.extend(entity_result.answer or [])
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
