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


_hobby_index_cache = None

def _load_hobby_index():
    global _hobby_index_cache
    if _hobby_index_cache is None:
        index_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../output/depth_10_size_1000000/hobby_names.json'
            )
        )
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                _hobby_index_cache = json.load(f)
        else:
            _hobby_index_cache = {}
    return _hobby_index_cache


_occupation_index_cache = None

def _load_occupation_index():
    global _occupation_index_cache
    if _occupation_index_cache is None:
        index_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../output/depth_10_size_1000000/occupation_names.json'
            )
        )
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                _occupation_index_cache = json.load(f)
        else:
            _occupation_index_cache = {}
    return _occupation_index_cache


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


class PhantomWikiQA(dspy.Signature):
    """You are a meticulous researcher answering questions from PhantomWiki, a fictional encyclopedia.

    IMPORTANT: Questions frequently have MULTIPLE correct answers (sometimes 10 or more). Your mission is to find ALL of them — not just the first one you encounter.

    HOW TO SEARCH:
    1. Break down the question into its component parts: identify anchor entities, relationships, and target attributes
    2. Search for each component separately with targeted queries
    3. For ancestry/family questions: after tracing one branch, ALWAYS search for siblings and other family branches — every sibling of an ancestor is another potential answer path
    4. Even when a question uses singular form ("the great-grandparent of X", "the cousin of Y"), there may be MULTIPLE entities at that hop — enumerate ALL of them before moving forward
    5. For attribute-lookup questions ("What is X of the person whose Y is Z"): multiple people may share the same attribute value — after finding one match, keep searching for others
    6. After every answer found, ask yourself: "Are there more?" and issue additional searches
    7. If a search fails, try completely different angles: search by a related name, a different relationship, or a nearby attribute

    HOW MANY QUESTIONS — CRITICAL FORMAT RULE:
    - "How many X does Y have?" → The answer MUST be a NUMBER (count), NOT a list of entity names.
    - WRONG: answer=['Alice Smith', 'Bob Jones']   RIGHT: answer=['2']
    - NEVER return entity names as the answer to a "how many" question — always return the integer count as a string.
    - After finding entities, COUNT them and return that integer as your answer.
    - If the chain in the question resolves to MULTIPLE distinct entities (e.g., multiple great-grandparents, multiple people with the same DOB), count X separately for EACH entity and return ALL distinct count values as separate string answers.

    MULTI-ENTITY ENUMERATION — Do NOT stop at the first qualifying entity:
    When the question's subject chain contains an indirect anchor (e.g., "the great-grandparent of Z", "the person whose DOB is D", "the person whose hobby is H"), MULTIPLE people may qualify. Stopping after finding the first qualifying entity is the most common mistake. You MUST search for ALL qualifying entities before computing any count:
    - Ancestor anchors (great-grandparent, grandparent, great-uncle): search BOTH the maternal AND paternal branches of Z — issue separate targeted searches for each grandparent pair (e.g., "grandparents of Z's mother" and "grandparents of Z's father")
    - Date-of-birth anchor: use search_by_date_exact(date_str) — this returns ALL people with that exact DOB from a pre-built exact-match index (no additional searches needed; just read ALL returned articles)
    - Hobby/occupation anchor: after finding person #1 with that attribute, issue 2+ more searches with varied queries to find MORE people sharing the same hobby/occupation
    - Friend/sibling anchor: enumerate ALL friends or siblings listed in the entity's passage before proceeding to count
    Only AFTER exhausting searches for all qualifying entities should you count X for each one and return ALL distinct count values as a list.

    IMPLICIT RELATIONSHIPS — NEVER SEARCH DIRECTLY, ALWAYS DERIVE VIA TRAVERSAL:
    Relationships like "cousin," "nephew," "niece," "great-uncle" are NOT stored as keywords in the wiki. You MUST derive them step by step:
    - "cousin of X" → find X's parents → find those parents' siblings → find those siblings' children (= X's cousins)
    - "nephew of X" → find X's siblings → find siblings' sons
    - "niece of X" → find X's siblings → find siblings' daughters
    - "great-uncle of X" → find X's grandparents → find those grandparents' siblings
    NEVER query "cousin of Alice" or "nephew of Bob" directly — the wiki does not contain those phrases.

    NON-STANDARD KINSHIP TERMS — derive step by step:
    - "great-uncle" or "great-aunt" = grandparent's sibling (go up 2 generations, find siblings)
    - "second uncle" or "second aunt" = great-grandparent's sibling (go up 3 generations, find siblings — ONE MORE generation up than great-uncle)
    - "grand-nephew" or "grand-niece" = sibling's grandchild
    - "first cousin once removed" = your first cousin's child, OR your parent's first cousin
    - "second cousin" = parent's first cousin's child (grandparent's sibling's grandchild)

    DATE-OF-BIRTH ANCHOR SEARCHES — use the exact-match index for complete recall:
    - ALWAYS use search_by_date_exact("YYYY-MM-DD") for questions involving "the person whose date of birth is X"
    - This tool uses a pre-built exact index and returns EVERY person in the wiki born on that date (sometimes 10-20 people!)
    - Read ALL returned articles and extract the relevant attribute (occupation, hobby, etc.) for EACH person — they are all valid answers
    - Tautological case: if the question asks "What is the DOB of the person whose DOB is X?", the answer is X directly — do NOT search, just return X

    DO NOT:
    - Stop after finding just one answer
    - Assume a question has a unique answer
    - Return entity names when the question asks "how many" — always return a numeric count as a string
    - Query for implicit relationships like "cousin of X" directly — derive them via the step-by-step traversal above
    - Give up with "unknown" or "cannot determine" after only a few searches — try at least 5 distinct approaches before concluding
    """

    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities, possibly requiring multi-hop reasoning"
    )
    answer: list[str] = dspy.OutputField(
        desc="A complete list of ALL correct answers found. Most questions have multiple answers. Search exhaustively before finishing. IMPORTANT: Return ONLY the exact answer values — do NOT include person names, attributions, or extra context alongside the answers. For example, if the question asks for occupations, return ['teacher', 'doctor'] NOT ['John Smith — teacher', 'Jane Doe — doctor']. For 'how many' questions, return the COUNT as a string (e.g., ['3']), NOT the entity names."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.retrieve_broad = dspy.Retrieve(k=50)

        # Single-phase ReAct (for entity, attribute, count_answer questions)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki, self.search_wiki_broad, self.search_by_date_exact],
            max_iters=50,
        )

        # Two-phase: Phase 1 finds pivot entities for count_pivot questions
        self.entity_finder = dspy.ReAct(
            signature=EntityFinderSig,
            tools=[self.search_wiki, self.search_wiki_broad, self.search_by_date_exact],
            max_iters=40,
        )

        # Two-phase: Phase 2 counts X for a single pivot entity
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

    def search_by_hobby(self, hobby_value: str) -> str:
        """Search PhantomWiki for ALL entities with exactly this hobby.
        Uses a pre-built exact-match index — guaranteed to return EVERY person in the wiki
        with this hobby. Far more complete than semantic search.
        Returns person names (up to 60), one per line.
        ALWAYS use this tool for questions involving 'the person whose hobby is X'."""
        index = _load_hobby_index()
        names = index.get(hobby_value, [])
        if not names:
            # Try case-insensitive match
            for key, val in index.items():
                if key.lower() == hobby_value.lower():
                    names = val
                    break
        if not names:
            return f"No entries found in hobby index for: {hobby_value}. Try search_wiki with 'hobby {hobby_value}'."
        sample = names[:60]
        header = f"EXACT MATCH: Found {len(names)} people with hobby '{hobby_value}'. Listing first {len(sample)}:\n"
        return header + "\n".join(sample)

    def search_by_occupation(self, occupation_value: str) -> str:
        """Search PhantomWiki for ALL entities with exactly this occupation.
        Uses a pre-built exact-match index — guaranteed to return EVERY person in the wiki
        with this occupation. Far more complete than semantic search.
        Returns person names (up to 60), one per line.
        ALWAYS use this tool for questions involving 'the person whose occupation is X'."""
        index = _load_occupation_index()
        names = index.get(occupation_value, [])
        if not names:
            # Try case-insensitive match
            for key, val in index.items():
                if key.lower() == occupation_value.lower():
                    names = val
                    break
        if not names:
            return f"No entries found in occupation index for: {occupation_value}. Try search_wiki with 'occupation {occupation_value}'."
        sample = names[:60]
        header = f"EXACT MATCH: Found {len(names)} people with occupation '{occupation_value}'. Listing first {len(sample)}:\n"
        return header + "\n".join(sample)

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
        q_lower = question.strip().lower()

        if question_type == 'count_pivot':
            # Detect hobby/occupation anchor in the question
            hobby_val = None
            occ_val = None
            hm = re.search(r"whose hobby is (.+?)(?:\s+have|\s+does|\s+did|\s*\?|$)", q_lower)
            om = re.search(r"whose occupation is (.+?)(?:\s+have|\s+does|\s+did|\s*\?|$)", q_lower)
            if hm:
                hobby_val = hm.group(1).strip().rstrip('?').strip()
            if om:
                occ_val = om.group(1).strip().rstrip('?').strip()

            # Detect if this is a SIMPLE anchor=pivot question:
            # "How many X does the person whose hobby/occupation is Y have?"
            # (No chain between "does the" and "person whose")
            is_simple_anchor = bool(
                re.search(r"does the person whose (?:hobby|occupation) is", q_lower)
            )

            if is_simple_anchor and (hobby_val or occ_val):
                # Bypass Phase 1: use exact index directly for pivot entities
                if hobby_val:
                    all_names = _load_hobby_index().get(hobby_val, [])
                else:
                    all_names = _load_occupation_index().get(occ_val, [])
                entities = all_names[:30] if all_names else []

                if not entities:
                    # Fallback to single-phase
                    result = self.react(question=question)
                    return dspy.Prediction(answer=result.answer)

            else:
                # Complex chain: let Phase 1 use its own ColBERT-based search strategy.
                # Hint injection was removed because prepending anchor names caused Phase 1 to
                # exhaust its iteration budget on wrong traversal paths instead of using
                # efficient broad search.
                phase1 = self.entity_finder(question=question, question_type=question_type)
                entities = phase1.target_entities or []

                if not entities:
                    # Fallback to single-phase when Phase 1 found nothing
                    result = self.react(question=question)
                    return dspy.Prediction(answer=result.answer)

            counts = []
            # Call count_computer once per pivot entity
            for entity in entities[:30]:  # cap at 30 to improve count distribution coverage
                try:
                    phase2 = self.count_computer(question=question, pivot_entity=entity)
                    c = phase2.count if phase2.count else '0'
                    c_stripped = c.strip()
                    if c_stripped.isdigit():
                        counts.append(c_stripped)
                    else:
                        m = re.match(r'^(\d+)', c_stripped)
                        counts.append(m.group(1) if m else '0')
                except Exception:
                    counts.append('0')
            return dspy.Prediction(answer=counts)

        else:
            # Single-phase for entity, attribute, count_answer
            result = self.react(question=question)
            return dspy.Prediction(answer=result.answer)
