import contextvars
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy


# ── Anchor Expansion ─────────────────────────────────────────────────────────

def _detect_named_person_anchor(question: str) -> str | None:
    """
    Detect if a question directly names a specific person as the anchor entity
    (as opposed to a property-based lookup). Returns the person's name or None.

    Handles AnchorExpander policy-refusal bugs where the model refuses to return
    a named person when the question also contains relational chains
    (e.g., "friend of X", "wife of X", "sister-in-law of the friend of X").
    """
    q_lower = question.lower()
    # Skip property-based questions — anchor is a property value, not a named person
    if re.search(r'\bwhose\s+\w+(?:\s+\w+)?\s+is\b', q_lower):
        return None
    if re.search(r'\bborn\s+on\s+\d|\bdate\s+of\s+birth\b|\bwith\s+(?:dob|date\s+of\s+birth)\b', q_lower):
        return None
    # Look for proper names: Title-Cased words preceded by "of"
    # The LAST such occurrence is typically the starting (anchor) entity
    matches = re.findall(r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?=[^a-zA-Z]|$)', question)
    if matches:
        return matches[-1]
    return None


class AnchorExpansionSig(dspy.Signature):
    """Find ALL people in the wiki that match the anchor property in the question.

    YOUR ONLY GOAL: enumerate every person who has the specific property value
    mentioned in the question. NOTHING ELSE.

    WHAT IS AN ANCHOR PROPERTY:
    - "the person whose occupation is X" -> find ALL people with occupation X
    - "the person whose hobby is X"      -> find ALL people with hobby X
    - "the person whose date of birth is X" -> find ALL people born on date X
    - If no property lookup (question directly names a person): return just that person's name

    CRITICAL RULES:
    1. ONLY search for people who HAVE the property. Search queries must be about
       the property value ONLY. Examples:
         "occupation financial controller"
         "hobby microbiology"
         "date of birth 1050-09-16" / "born on 1050-09-16"
    2. DO NOT search for family members, relatives, siblings, parents, children, or
       ANY relationship. Do NOT traverse family trees. Do NOT answer the full question.
    3. STOP RULE: Once you have found people matching the property, call finish IMMEDIATELY.
       Your output must ONLY contain names of people who DIRECTLY POSSESS the anchor property.
       Do NOT follow 'of the X of Y' relationship chains. Do NOT traverse family trees.
       Do NOT compute or research anything beyond the identity of property holders.
       The moment you have a list of names, output them and stop.
    4. In this wiki, each property value is typically shared by 5-15+ people.
       Run 3-5 searches with DIFFERENT PHRASINGS of the SAME property lookup to find them all.
    5. If you cannot find anyone after 4 searches, return an empty list.
    6. NAMED PERSON CASE: If the question directly names a specific person (e.g. "Forest Benner",
       "Deon Gall"), immediately return that person's name as the single anchor entity.
       Do NOT search. Do NOT traverse their family tree. Just return their name.
    """
    question: str = dspy.InputField(
        desc="Question containing a property-based entity lookup"
    )
    anchor_entities: list[str] = dspy.OutputField(
        desc="ALL person names matching the anchor property. Only people with the property — no family members."
    )


class DOBNameExtractor(dspy.Signature):
    """Extract full names of people born on a specific date from wiki passages.

    Carefully read all passages. Identify every person whose date of birth
    exactly matches the target date (format: YYYY-MM-DD). Do NOT include
    people with similar but different dates. Only confirmed exact matches.
    """
    passages: str = dspy.InputField(
        desc="Wiki passages potentially containing birthday/date of birth information for multiple people"
    )
    target_date: str = dspy.InputField(
        desc="The exact birth date to find, in YYYY-MM-DD format (e.g., '0918-01-17')"
    )
    names: list[str] = dspy.OutputField(
        desc="Full names of people whose date of birth exactly matches target_date. Return [] if none found."
    )


class AnchorExpanderModule(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        self.react = dspy.ReAct(
            signature=AnchorExpansionSig,
            tools=[self.search_wiki],
            max_iters=15,
        )
        self.dob_extractor = dspy.Predict(DOBNameExtractor)

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 15 relevant passages.
        Extract every person name that matches the property you are searching for."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def _dob_fallback(self, existing_entities: list, full_date: str, year: str) -> list:
        """
        Programmatic DOB year-only fallback search.
        Issues year-only ColBERT queries to find additional people with the given
        birth date, beyond what the main ReAct found (which uses full-date queries
        within its 15-iteration budget).

        Called when anchor_entities < 5 AND a DOB pattern is in the question.
        """
        existing_lower = {e.strip().lower() for e in existing_entities}
        combined = list(existing_entities)

        queries = [
            f"born {year}",
            f"date of birth {year}",
            f"birthday {year}",
        ]

        all_passages = []
        seen_p: set = set()
        for query in queries:
            try:
                result = self.retrieve(query)
                for p in result.passages:
                    if p not in seen_p:
                        seen_p.add(p)
                        all_passages.append(p)
            except Exception:
                pass

        if not all_passages:
            return combined

        try:
            extracted = self.dob_extractor(
                passages="\n\n---\n\n".join(all_passages[:20]),
                target_date=full_date,
            )
            for name in (extracted.names or []):
                name_clean = name.strip()
                if name_clean and name_clean.lower() not in existing_lower:
                    existing_lower.add(name_clean.lower())
                    combined.append(name_clean)
        except Exception:
            pass

        return combined

    def forward(self, question):
        # Pre-check: if question directly names a person, return them without invoking
        # the ReAct (avoids policy-refusal bug for chains like "friend of X", "wife of X")
        named_person = _detect_named_person_anchor(question)
        if named_person:
            return dspy.Prediction(anchor_entities=[named_person])

        # Main ReAct: find property-based anchor entities
        try:
            result = self.react(question=question)
            anchor_entities = result.anchor_entities or []
        except Exception:
            anchor_entities = []

        # DOB year-only fallback: if few anchors found, issue additional year-only
        # ColBERT queries beyond the main ReAct's full-date search budget
        if len(anchor_entities) < 5:
            dob_match = re.search(r'\b(\d{3,4})-(\d{2})-(\d{2})\b', question)
            if dob_match:
                full_date = dob_match.group(0)
                year = dob_match.group(1)
                anchor_entities = self._dob_fallback(anchor_entities, full_date, year)

        return dspy.Prediction(anchor_entities=anchor_entities)


# ── Per-Entity Processor ──────────────────────────────────────────────────────

class SingleAnchorQA(dspy.Signature):
    """Answer a question by starting from ONE specific anchor entity.

    You are processing ONE anchor entity in a multi-entity question.
    Follow the relationship chain described in the question FROM THIS SINGLE ENTITY ONLY.

    EXAMPLES:
    - Question: "How many brothers-in-law does the person whose hobby is die-cast toy have?"
      Anchor: "Refugio Crum"
      → Search for Refugio Crum, find their spouse, count spouse's brothers, return that number.

    - Question: "What is the occupation of the grandchild of the person whose DOB is 0918-01-17?"
      Anchor: "Loraine Moritz"
      → Search for Loraine Moritz's children, then their children (grandchildren), find occupation.

    COUNTING vs NAMING — CRITICAL:
    - "How many X does [anchor] have?" → return exactly ONE NUMBER like "3", "0", "7"
      NEVER return person names for a counting question.
    - "Who is the X of [anchor]?" → return person NAME(s)
    - "What is the occupation/hobby/DOB of [anchor]?" → return that value

    SEARCH STRATEGY:
    1. Search for the anchor entity by name to get their full profile
    2. Follow the chain step-by-step, searching each intermediate entity
    3. Read ALL returned passages carefully — extract all relevant names/values

    If you cannot find a verifiable answer for this anchor entity, return an empty list.
    """
    question: str = dspy.InputField(desc="The original question")
    anchor_entity: str = dspy.InputField(desc="The specific anchor entity to process")
    answer: list[str] = dspy.OutputField(
        desc="Answer(s) found for this anchor entity. ONE number for counting questions ('how many'). ALL relevant names/values for other question types."
    )


class PerEntityProcessor(dspy.Module):
    """Process a single anchor entity and return the answer contribution."""
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=SingleAnchorQA,
            tools=[self.search_wiki],
            max_iters=28,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 10 relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question: str, anchor_entity: str) -> dspy.Prediction:
        try:
            result = self.react(question=question, anchor_entity=anchor_entity)
            return dspy.Prediction(answer=result.answer or [])
        except Exception:
            return dspy.Prediction(answer=[])


# ── Main Q&A ──────────────────────────────────────────────────────────────────

class PhantomWikiQA(dspy.Signature):
    """Answer questions about a fictional wiki universe. Provide accurate and complete answers.

    You are given ANCHOR ENTITIES — the complete list of all people matching the
    question's starting property. Process EVERY anchor entity and collect ALL answers.

    HOW TO ANSWER:

    1. PROCESS ALL ANCHOR ENTITIES: Work through each entity in anchor_entities_context.
       Each anchor entity may give a DIFFERENT answer — do not skip any.
       For each anchor entity: search the wiki, follow the relationship chain, note the answer.
       Keep a running list: "Processed: [entity] -> answer: [X]"

    2. COUNTING vs NAMING — CRITICAL DISTINCTION:
       - "How many X does Y have?" -> return a NUMBER (e.g. "3", "0", "7")
         NEVER return person names for a counting question.
       - "Who is the X of Y?" -> return person NAME(s).
       - "What is the occupation / hobby / DOB of Y?" -> return that value.

    3. READ ALL SEARCH RESULTS: Read every passage returned. Multiple passages may
       contain different matching entities — extract all of them.

    4. AGGREGATE ALL ANSWERS: Collect one answer per anchor entity you process.
       If 5 anchor entities each have a different count, return ALL 5 counts.
       Do NOT discard answers — include every distinct value found.

    5. LEVEL-INVENTORY TRAVERSAL: At EACH hop in a relationship chain, inventory
       ALL entities at that level before proceeding to the next hop. Do NOT stop
       after the first match — explore EVERY branch.
       Example approach: "Level 0: [starting entity] → Level 1: [ALL children/
       parents/siblings found] → Level 2: [ALL entities at next hop] → ..."
       Check every branch. Missing one branch means missing valid answers.

    6. EXTENDED FAMILY DEFINITIONS (apply precisely):
       - uncle/aunt = parent's SIBLING (one generation up from parent)
       - second uncle/second aunt = grandparent's SIBLING (sibling of parent's parent)
       - cousin = parent's sibling's child (children of uncles/aunts)
       - second cousin = grandparent's sibling's grandchild
       - GENDER is critical: 'uncle'/'brother'/'grandson'/'son' = MALE ONLY.
         'aunt'/'sister'/'granddaughter'/'daughter' = FEMALE ONLY.
         Apply gender filter when counting or identifying relatives.

    7. VERIFIED ANSWERS ONLY: Only include answers explicitly supported by search results.
       Do not guess. If you cannot verify an answer for an anchor entity, skip that entity.
    """
    question: str = dspy.InputField(
        desc="The question to answer about the wiki universe"
    )
    anchor_entities_context: str = dspy.InputField(
        desc="Pre-identified anchor entities: all people matching the question's starting property"
    )
    answer: list[str] = dspy.OutputField(
        desc="All answers found. Numbers for counting questions, names for 'who' questions."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.anchor_expander = AnchorExpanderModule()
        self.entity_processor = PerEntityProcessor()
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki],
            max_iters=35,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 10 relevant passages.

        Read ALL returned passages carefully — multiple passages may contain
        different people or entities matching the search condition.
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def _parallel_process(self, question: str, anchor_entities: list, max_parallel: int = 4) -> list:
        """Process anchor entities in parallel batches of max_parallel.

        Uses a fresh contextvars.copy_context() per submitted task to correctly
        propagate DSPy's LM/RM context to each worker thread independently.
        IMPORTANT: Each task needs its OWN copy — CPython's Context.run() sets
        an object-level ctx_entered flag (not per-thread), so sharing one Context
        across concurrent threads causes all but the first to silently fail.
        """
        all_answers = []

        for i in range(0, len(anchor_entities), max_parallel):
            batch = anchor_entities[i:i + max_parallel]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        contextvars.copy_context().run,
                        self.entity_processor,
                        question=question,
                        anchor_entity=entity,
                    ): entity
                    for entity in batch
                }
                for future in as_completed(futures):
                    try:
                        pred = future.result()
                        all_answers.extend(pred.answer or [])
                    except Exception:
                        pass

        return all_answers

    def forward(self, question):
        # Phase 1: expand anchor entities
        try:
            anchor_result = self.anchor_expander(question=question)
            anchor_entities = anchor_result.anchor_entities or []
        except Exception:
            anchor_entities = []

        # Phase 2: choose processing strategy based on anchor count
        if len(anchor_entities) > 1:
            # Multiple anchors: use parallel per-entity processing (max 4 at a time)
            all_answers = self._parallel_process(question, anchor_entities, max_parallel=4)
        else:
            # Single anchor or no anchor: use original full ReAct (better for complex chains)
            if anchor_entities:
                anchor_context = (
                    f"Found 1 anchor entity: {anchor_entities}. "
                    "Process it to collect ALL answers. Note: there may be multiple "
                    "intermediate entities in the chain — explore ALL branches thoroughly."
                )
            else:
                anchor_context = (
                    "No anchor entities pre-identified. Determine them during your search."
                )
            result = self.react(
                question=question,
                anchor_entities_context=anchor_context,
            )
            all_answers = result.answer or []

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in all_answers:
            key = a.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        return dspy.Prediction(answer=deduped)
