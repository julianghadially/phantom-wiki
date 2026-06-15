import dspy
import json
import os
import re

_date_index_cache = None

def _load_date_index():
    global _date_index_cache
    if _date_index_cache is None:
        index_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../output/depth_10_size_1000000/date_passages.json'
            )
        )
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                _date_index_cache = json.load(f)
        else:
            _date_index_cache = {}
    return _date_index_cache


def _classify_question(question: str) -> str:
    """Deterministically classify question type from surface form.
    Never calls an LLM — based purely on lexical patterns.
    Returns: 'entity', 'attribute', 'count_answer', or 'count_pivot'.
    """
    q = question.strip().lower()

    if q.startswith('how many'):
        # Find 'does', 'did', or 'do' and inspect what follows
        m = re.search(r'\b(?:does|did|do)\b', q)
        if m:
            anchor = q[m.end():].strip()
            # If anchor starts with 'the ', it is a derived chain → count_pivot
            # Otherwise it is a proper noun (named person) → count_answer
            if anchor.startswith('the '):
                return 'count_pivot'
        return 'count_answer'

    elif q.startswith(('what is', 'what are', 'what was', 'what were')):
        return 'attribute'

    else:
        # 'who is', 'who are', 'list', etc.
        return 'entity'


class EntityFinderSig(dspy.Signature):
    """Phase 1: Find ALL terminal entities in the traversal chain of the question.
    YOUR ONLY JOB IS TO FIND PERSON NAMES. Do NOT compute attributes. Do NOT answer the question yet.

    Instructions by question_type:

    'entity': Find ALL entities that ARE the direct answer.
        e.g., "Who is the great-grandson of X?" → find ALL great-grandsons of X.

    'attribute': Find ALL entities whose ATTRIBUTE the question asks about.
        e.g., "What is the hobby of the great-uncle of X?" → find the great-uncle entity.
        e.g., "What is the occupation of the grandchild of the great-grandfather of Y?" → find ALL such great-grandchildren.

    'count_answer': Find ALL entities being counted.
        e.g., "How many great-grandsons does Y have?" → find ALL great-grandsons of Y.
        forward() will count them; you just need to enumerate them exhaustively.

    'count_pivot': Find ALL PIVOT entities (the SUBJECTS of the counting, NOT the things counted).
        e.g., "How many X does the cousin of Z have?" → find ALL cousins of Z (NOT their X).
        e.g., "How many X does the person whose hobby is H have?" → find ALL persons with hobby H.
        e.g., "How many X does the great-grandchild of the person whose occupation is V have?" →
              find ALL great-grandchildren of ALL persons with occupation V.
        Do NOT find or count X — that is done in Phase 2 using your output.

    KINSHIP DEPTH — CRITICAL:
    - "grandparent of X" = X's parents' PARENTS (go UP TWO generations from X, not one).
    - "great-grandparent of X" = THREE generations up from X.
    - "great-uncle/aunt of X" = X's grandparents' siblings (go up TWO gen, find siblings).
    - "second uncle/aunt of X" = X's great-grandparents' siblings (go up THREE gen, find siblings).
    - "first cousin once removed" = first cousin's child OR parent's first cousin.
    - "second cousin" = grandparent's sibling's grandchild.
    - NEVER confuse "grandparent" (2 gen up) with "parent" (1 gen up). If the question says
      "grandparent of Y", you must go to Y's parents first and THEN to those parents' parents.

    IMPLICIT RELATIONSHIPS — DERIVE VIA TRAVERSAL, NEVER QUERY DIRECTLY:
    - "cousin of X" → X's parents → parents' siblings → siblings' children
    - "nephew/niece of X" → X's siblings → siblings' sons/daughters
    - "great-uncle of X" → X's grandparents → grandparents' siblings
    - Never search "cousin of Alice" or "nephew of Bob" — those phrases are not in the wiki.

    BILATERAL BRANCH SEARCH — MANDATORY:
    - For every ancestor entity found, ALSO search for the other lineage branch.
    - If you find a maternal grandparent, ALSO find the paternal grandparent.
    - For "cousin of X": search BOTH X's mother's siblings AND X's father's siblings.
    - For "great-uncle of X": search BOTH maternal grandparents' siblings AND paternal grandparents' siblings.
    - Only after exhausting BOTH branches should you move forward.

    OCCUPATION / HOBBY ANCHORS — MULTIPLE PEOPLE MAY MATCH:
    - "the person whose occupation is X" does NOT mean there is only one such person.
    - Issue at least 3 varied search queries (e.g., "occupation X", "works as X", "job X").
    - Also use search_wiki_broad for broader coverage.
    - Continue until you are confident you have found ALL matching persons.

    DATE-OF-BIRTH ANCHOR:
    - ALWAYS use search_by_date_exact("YYYY-MM-DD") for DOB-anchored questions.
    - This returns EVERY person with that exact DOB. Read ALL returned articles.
    - Tautological case: if asked for DOB of person whose DOB is X, return [X] directly.

    COMPLETENESS:
    - After finding entities on one branch, ALWAYS check whether there are more.
    - For each ancestor entity: verify whether they have OTHER children not yet explored.
    - For attribute anchors (occupation/hobby): verify whether other people share the attribute.
    - Do NOT call finish() until you have confirmed no remaining branches or matches exist.
    - Return ONLY person names. Do NOT include attribute values, counts, or "unknown".
    """
    question: str = dspy.InputField()
    question_type: str = dspy.InputField(
        desc="One of: entity, attribute, count_answer, count_pivot"
    )
    target_entities: list[str] = dspy.OutputField(
        desc="Complete list of all terminal entities found. Return ONLY person names, never attributes."
    )


class AttributeComputerSig(dspy.Signature):
    """Phase 2 (attribute questions): Look up attributes for explicitly named entities.

    Phase 1 has already identified all target entities. Your job is ONLY to look up their attributes.
    Do NOT re-traverse the kinship chain — it is already done and your target_entities are the result.

    TASK: For EACH entity in target_entities:
    1. Search for that entity BY NAME directly (e.g., search_wiki("Werner Corrigan")).
    2. Extract the attribute type requested in the original question.
    3. Add the attribute value(s) to your answer list.

    RULES:
    - Return ONLY attribute values (e.g., 'fishkeeping', 'marine scientist') — NOT entity names.
    - Combine attribute values from ALL target entities into one list.
    - Process every entity in target_entities — do not skip any.
    - If an entity has multiple values for the requested attribute (e.g., multiple hobbies), include all.
    - Deduplicate: if two entities share the same attribute value, include it only once.
    """
    question: str = dspy.InputField(desc="Original question — tells you which attribute type to extract")
    target_entities: list[str] = dspy.InputField(desc="The entities to look up attributes for")
    answer: list[str] = dspy.OutputField(
        desc="All attribute values (not entity names) across all target entities, deduplicated"
    )


class CountComputerSig(dspy.Signature):
    """Phase 2 (count questions): Count how many X the given pivot entity has.

    The original question asks 'How many X does [PIVOT] have?'. Your pivot_entity is the PIVOT.
    Count X for THIS SPECIFIC ENTITY ONLY.

    TASK:
    1. Use the ORIGINAL QUESTION to determine what X is (the thing being counted).
    2. Search for pivot_entity by name.
    3. Count X for pivot_entity using traversal if needed (e.g., cousins = parents' siblings' children).
    4. Return a SINGLE integer string.

    RULES:
    - DO NOT re-traverse the chain that led to pivot_entity; focus only on counting X for it.
    - For implicit relationships (cousin, nephew, great-uncle): derive via step-by-step traversal.
    - If you find 0 instances of X, return '0'.
    - Return only a single integer string (e.g., '3', '0', '12') — NOT entity names, NOT ranges.
    """
    question: str = dspy.InputField(desc="Original question — tells you what X to count")
    pivot_entity: str = dspy.InputField(desc="The specific entity to count X for")
    count: str = dspy.OutputField(desc="Count of X for this entity as a single integer string (e.g., '3')")


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.retrieve_broad = dspy.Retrieve(k=50)

        # Phase 1: Find all terminal entities via multi-hop traversal
        self.entity_finder = dspy.ReAct(
            signature=EntityFinderSig,
            tools=[self.search_wiki, self.search_wiki_broad, self.search_by_date_exact],
            max_iters=40,
        )

        # Phase 2a: Look up attributes for explicitly named entities (bulk, single call)
        self.attribute_computer = dspy.ReAct(
            signature=AttributeComputerSig,
            tools=[self.search_wiki, self.search_wiki_broad],
            max_iters=35,
        )

        # Phase 2b: Count X for a single pivot entity
        self.count_computer = dspy.ReAct(
            signature=CountComputerSig,
            tools=[self.search_wiki, self.search_wiki_broad],
            max_iters=15,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Use for targeted lookups: finding a specific person, their relationships, or their attributes.
        Returns up to 10 relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_broad(self, query: str) -> str:
        """Search PhantomWiki broadly for exhaustive enumeration.
        Use this when you need to find ALL instances matching a criterion — e.g., all people with a given occupation,
        all people sharing a date of birth, or all entities with a specific attribute value.
        Also useful for date-of-birth lookups where multiple people share the same date.
        Returns up to 50 passages for wider coverage than search_wiki."""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def search_by_date_exact(self, date_str: str) -> str:
        """Search PhantomWiki for ALL entities with exactly this date of birth.
        Uses a pre-built exact-match index — guaranteed to return EVERY person in the wiki
        born on this exact date. Far more complete than semantic search (which misses most matches).
        date_str should be in format YYYY-MM-DD (e.g., '0946-07-14').
        ALWAYS use this tool for questions involving 'the person whose date of birth is X'."""
        index = _load_date_index()
        passages = index.get(date_str, [])
        if not passages:
            return f"No entries found in exact index for date: {date_str}. Try search_wiki_broad with 'born {date_str.split('-')[0]}'."
        header = f"EXACT MATCH: Found {len(passages)} people born on {date_str}:\n\n"
        return header + "\n\n".join(passages)

    def forward(self, question):
        question_type = _classify_question(question)

        # Phase 1: Find all terminal entities via traversal
        phase1 = self.entity_finder(question=question, question_type=question_type)
        entities = phase1.target_entities or []

        if question_type == 'entity':
            return dspy.Prediction(answer=entities)

        elif question_type == 'count_answer':
            return dspy.Prediction(answer=[str(len(entities))])

        elif question_type == 'attribute':
            if not entities:
                return dspy.Prediction(answer=[])
            phase2 = self.attribute_computer(
                question=question,
                target_entities=entities
            )
            return dspy.Prediction(answer=phase2.answer or [])

        elif question_type == 'count_pivot':
            if not entities:
                return dspy.Prediction(answer=[])
            counts = []
            # Call count_computer once per pivot entity — guarantees per-entity counting
            # without the 'enumerate-then-collapse' failure mode
            for entity in entities[:20]:  # cap at 20 to prevent runaway
                try:
                    phase2 = self.count_computer(question=question, pivot_entity=entity)
                    c = phase2.count if phase2.count else '0'
                    # Normalize: keep only leading digits
                    c_stripped = c.strip()
                    if c_stripped.isdigit():
                        counts.append(c_stripped)
                    else:
                        import re as _re
                        m = _re.match(r'^(\d+)', c_stripped)
                        counts.append(m.group(1) if m else '0')
                except Exception:
                    counts.append('0')
            return dspy.Prediction(answer=counts)

        else:
            return dspy.Prediction(answer=[])
