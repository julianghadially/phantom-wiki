import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy


# ── Anchor Expansion ─────────────────────────────────────────────────────────

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
    3. Stop and return as soon as you have listed all people with the property.
       Do NOT continue researching after finding the people.
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


class AnchorExpanderModule(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        self.react = dspy.ReAct(
            signature=AnchorExpansionSig,
            tools=[self.search_wiki],
            max_iters=15,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 15 relevant passages.
        Extract every person name that matches the property you are searching for."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        try:
            return self.react(question=question)
        except Exception:
            return dspy.Prediction(anchor_entities=[])


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
        desc="ALL answers found for this anchor entity. Include EVERY name, number, or value found across all branches. If multiple intermediate entities each contribute answers, include ALL of them — do NOT collapse to one answer."
    )


class PerEntityProcessor(dspy.Module):
    """Process a single anchor entity and return the answer contribution."""
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=SingleAnchorQA,
            tools=[self.search_wiki],
            max_iters=15,
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

    5. FAMILY TREE TRAVERSAL: For relationship questions, find ALL intermediate entities
       at each hop and explore every branch systematically. Do not stop after the first match.

    6. VERIFIED ANSWERS ONLY: Only include answers explicitly supported by search results.
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

        Uses contextvars.copy_context() to propagate DSPy's LM/RM context to threads.
        """
        ctx = contextvars.copy_context()
        all_answers = []

        for i in range(0, len(anchor_entities), max_parallel):
            batch = anchor_entities[i:i + max_parallel]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        ctx.run,
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
