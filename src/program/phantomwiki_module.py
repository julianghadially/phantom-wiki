import dspy
import json
import os

_date_index_cache = None


def _load_date_index():
    global _date_index_cache
    if _date_index_cache is None:
        index_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../output/depth_10_size_1000000/date_passages.json"
        )
        with open(index_path, "r") as f:
            _date_index_cache = json.load(f)
    return _date_index_cache


class EntityFinderSig(dspy.Signature):
    """You are a TRAVERSAL SPECIALIST for PhantomWiki, a fictional encyclopedia.

    Your ONLY job is to identify ALL person names at the END of the relationship chain in the question.
    Do NOT compute attributes (hobbies, occupations, dates) or counts — that happens separately later.

    CRITICAL RULES:
    - Output ONLY person names — NEVER attributes, occupations, hobbies, dates, or counts
    - Find ALL qualifying persons, not just the first one
    - For "What is X of CHAIN?" → traverse CHAIN, output all persons at its end
    - For "How many X does CHAIN have?" → traverse CHAIN, output all persons CHAIN resolves to
    - For "Who is CHAIN?" or "Which person is CHAIN?" → traverse CHAIN, output all persons (these ARE the final answer)

    SEARCH STRATEGY:
    1. DATE-OF-BIRTH ANCHOR: If question mentions a specific date (YYYY-MM-DD), call search_by_date_exact(date) FIRST — returns ALL people born on that date with 100% recall
    2. ATTRIBUTE ANCHOR: If question mentions a job/occupation/hobby as an anchor, use search_wiki_broad and try at least 5 different query phrasings (e.g. "occupation video editor", "job video editor", "video editor PhantomWiki", "works as video editor", "career video editor") to find ALL people with that attribute
    3. ANCESTOR CHAINS: Always search BOTH maternal AND paternal branches — families split across different surnames. Search the mother's family AND father's family separately.
    4. After finding one person, always ask "are there more?" and issue additional searches
    5. For multi-hop chains: traverse each step in order, finding ALL entities at each hop before proceeding

    NON-STANDARD KINSHIP TERMS:
    - Second aunt/uncle = sibling of a GREAT-GRANDPARENT (traverse up 3 generations to great-grandparent, then find their siblings)
    - Second cousin = grandparent's sibling's grandchild
    - First cousin once removed = parent's cousin OR cousin's child
    - Great-uncle/aunt = grandparent's sibling
    - Grand-nephew/niece = sibling's grandchild

    DO NOT stop after finding 1-2 entities. There may be 5, 10, or more valid entities the chain resolves to.
    For tautological DOB questions ("What is the DOB of person born on DATE?"), include the date-string itself as the sole entity: ["DATE"].
    """
    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities requiring multi-hop reasoning"
    )
    target_entities: list[str] = dspy.OutputField(
        desc="ALL person names at the end of the traversal chain. Names only — NO attributes, occupations, hobbies, dates, or counts."
    )


class AnswerComputerSig(dspy.Signature):
    """You are an ANSWER SPECIALIST for PhantomWiki, a fictional encyclopedia.

    The traversal phase has already identified the target entities. Your job is to compute the FINAL ANSWER from those entities.

    Inputs:
    - question: the original question
    - target_entities: list of ALL persons (or the answer directly) from the traversal phase

    YOUR TASK depends on the question type:

    1. ENTITY QUESTIONS ("Who is...?" / "Which person is...?"):
       → Return target_entities as-is — they ARE the answer, no additional search needed

    2. ATTRIBUTE QUESTIONS ("What is the HOBBY/OCCUPATION/DOB/etc. of CHAIN?"):
       → For EACH entity in target_entities, search for and retrieve their specific attribute
       → Combine all found attribute values into one list
       → ANTI-CONTAMINATION: Only return attributes of entities IN target_entities — do NOT return attributes of other people you encounter in search results
       → Return ONLY the attribute values (e.g., ["botany", "astronomy"]), NOT person names alongside them

    3. COUNTING QUESTIONS ("How many X does CHAIN have?"):
       → Process EACH entity in target_entities SEPARATELY and INDEPENDENTLY:
         Step 1: Search specifically for "[entity name] [relation]" (e.g., "Alice Smith cousins", "Bob Jones siblings")
         Step 2: Count the results for that specific entity
         Step 3: Record the count as a numeric string (e.g., "3")
         Step 4: Move to the NEXT entity in the list and repeat
       → After processing ALL entities, return the DISTINCT set of count strings
       → HOW MANY FORMAT: counts MUST be numeric strings ("3"), NEVER entity names
       → Include "0" if an entity truly has zero of the relation
       → If there are N entities in target_entities, you should compute N counts (then deduplicate)

    IMPORTANT: Process EVERY entity in target_entities — not just the first one.
    If target_entities is empty, search for the answer from scratch using the question.
    """
    question: str = dspy.InputField(desc="The original question")
    target_entities: list[str] = dspy.InputField(
        desc="ALL persons identified by the traversal phase. Process every single one of them."
    )
    answer: list[str] = dspy.OutputField(
        desc="Complete list of ALL correct answers. Attributes: all attribute values. Counts: distinct count strings. Entities: all entity names."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.retrieve_broad = dspy.Retrieve(k=30)

        all_tools = [self.search_wiki, self.search_wiki_broad, self.search_by_date_exact]
        lookup_tools = [self.search_wiki, self.search_wiki_broad]

        self.entity_finder = dspy.ReAct(
            signature=EntityFinderSig,
            tools=all_tools,
            max_iters=35,
        )

        self.answer_computer = dspy.ReAct(
            signature=AnswerComputerSig,
            tools=lookup_tools,
            max_iters=20,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Try different query angles: by person name, by relationship type, or by attribute value.
        Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_broad(self, query: str) -> str:
        """Broader search returning top 30 results (3x more than search_wiki).
        Use for attribute-anchor questions (finding all people with a given hobby/occupation)
        or when regular search misses entities. Returns many relevant passages."""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def search_by_date_exact(self, date_str: str) -> str:
        """Exact date-of-birth lookup. Given a date in YYYY-MM-DD format, returns ALL article texts
        for every person born on that exact date with 100% recall — no semantic approximation.
        Use for ANY question anchored by a specific date of birth."""
        date_index = _load_date_index()
        passages = date_index.get(date_str, [])
        if not passages:
            return f"No people found with date of birth {date_str}."
        return f"Found {len(passages)} people born on {date_str}:\n\n" + "\n\n".join(passages)

    def forward(self, question):
        # Phase 1: Find all target entities via traversal
        phase1_result = self.entity_finder(question=question)
        target_entities = phase1_result.target_entities or []

        # Phase 2: Compute final answer given the identified entities
        phase2_result = self.answer_computer(
            question=question,
            target_entities=target_entities,
        )

        return dspy.Prediction(answer=phase2_result.answer)
