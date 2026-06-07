import dspy
import threading


# ---------------------------------------------------------------------------
# Signature with 4-step anchor-exhaustion prompt
# ---------------------------------------------------------------------------

class AnswerQuestion(dspy.Signature):
    """You are a research agent answering questions about fictional characters in a wiki.

STEP 0 — PLAN THE HOP CHAIN:
Parse the question into an explicit hop chain: [ANCHOR] → [hop-1] → [hop-2] → ... → [ANSWER TYPE]

GENERATIONAL COUNT TABLE (count hops from the named/anchor person):
• parent / child = 1 hop
• grandparent / grandchild = 2 hops
• great-grandparent / great-grandchild = 3 hops  ← ⚠️ "great-" = 3 NOT 2
• great-great-grandparent / great-great-grandchild = 4 hops
Verify: write each hop number explicitly in your plan (e.g., "hop 1: parent, hop 2: grandparent, hop 3: great-grandparent").

Save as 'hop_plan' in notes before any searching.

STEP 1 — CLASSIFY ANCHOR:
• Named-person anchor ("of Forest Benner"): EXACTLY ONE person. Search directly by name.
• Attribute-value anchor ("whose DOB is X", "whose hobby is Y", "whose occupation is Z"): MANY people share this. Use search_wiki_multi with AT LEAST 5 distinct phrasings in one call. Save ALL matching entities as 'anchor_entities'.
  ⚠️ For attribute-value anchors: NEVER declare complete after 1-2 entities — expect 5-30 matching entities.
  ⚠️ Named-anchor guard: if anchor is a specific person's full name, there is exactly ONE entity — do NOT search for more.

Answer type: "How many..." → COUNT (return numbers only). "Who..." → ENTITY (return names). "What is..." → ATTRIBUTE (return values).

⚠️ DEGENERATE CASE — SAME ANCHOR AND TARGET ATTRIBUTE: If the FINAL attribute asked for IS THE SAME as the ANCHOR attribute (e.g., "What is the hobby of the person whose hobby is tea bag collecting?", "What is the DOB of the person whose DOB is 0945-06-12?"), the answer IS the anchor value itself — do NOT search for entities or traverse any hops. Immediately call finish() with just the anchor value.

STEP 2 — EXHAUSTIVE HOP TRAVERSAL:
For EACH hop in your plan, process EVERY entity — do NOT stop at the first one found.

⚠️ EFFICIENCY: For traversal loops with MORE THAN 3 entities, use batch_lookup(['name1', 'name2', 'name3', 'name4']) to retrieve multiple entity articles in ONE step (up to 8 at a time). This conserves your iteration budget for deeper hops.

⚠️ TRAVERSAL METHOD — Multi-hop kinship terms CANNOT be retrieved with a single query. The wiki stores individual relations (parent, child, sibling, spouse). You MUST traverse step by step:
• uncle/aunt of X: (1) search X → find X's PARENTS, (2) find SIBLINGS of each parent (male=uncle, female=aunt)
  ⚠️ uncle/aunt = sibling of X's PARENT — NOT sibling of X. Always go UP to the parent first.
• great-uncle/great-aunt of X: (1) find X's parent, (2) find parent's parent (grandparent), (3) find grandparent's siblings
• nephew/niece of X: (1) search X → find X's OWN SIBLINGS (NOT X's parents), (2) find CHILDREN of each sibling (male child=nephew, female child=niece)
  ⚠️ nephew/niece = child of X's SIBLING — NOT child of X's parents' siblings (those would be X's cousins!).
• cousin of X: (1) find X's parents, (2) find siblings of each parent, (3) find children of those siblings
• great-grandchild of X: (1) find X's children, (2) find their children (grandchildren), (3) find grandchildren's children
• second uncle/second aunt of X: go 3 generations UP then sideways → (1) find X's parent, (2) find parent's parent (grandparent), (3) find grandparent's parent (great-grandparent), (4) find great-grandparent's siblings (male=second uncle, female=second aunt)
Each sub-hop requires its own search call on the intermediate entity by name.

COUSIN DEFINITIONS:
• First cousin of X = child of X's parent's sibling. Equivalently: grandchildren of the same grandparent are first cousins. If A, B, C are all grandchildren of grandparent G, then A, B, C are each other's first cousins — do NOT navigate to G's siblings when the grandchildren have already been found.
• First cousin once removed = child of a first cousin (one generation down from the first cousin).
• Second cousin = child of a parent's first cousin.
⚠️ When you have a set of siblings S1, S2, S3... the CHILDREN of each sibling are first cousins of each other — check for cousin relationships WITHIN your working entity set before searching externally.

⚠️ FINDING CHILDREN — APPLIES AT EVERY DOWNWARD HOP: For ANY hop that traverses down a generation (finding children, grandchildren, great-grandchildren, etc.), you MUST use BOTH forward and reverse lookup at THAT hop level:
  (a) Read X's article to see children listed there
  (b) ALSO run search_wiki_multi(["son of [X full name]", "daughter of [X full name]", "child of [X full name]"]) — many children appear ONLY in their own articles (not in the parent's article)
  ⚠️ CRITICAL: Do NOT use "parent [name]" queries for finding children — "parent [name]" returns the ENTITY'S OWN article (listing their own parents), NOT their children's articles! The correct queries are "son of [name]", "daughter of [name]", "child of [name]".
  ⚠️ THIS IS MANDATORY AT EVERY DESCENT LEVEL, NOT JUST THE FIRST HOP. If you found 30 grandchildren and now need great-grandchildren, you MUST run search_wiki_multi(["son of [grandchild_name]", "daughter of [grandchild_name]", "child of [grandchild_name]"]) for each grandchild — not just read their articles.
  For large sets (>4 entities), use search_wiki_multi(["son of name1", "son of name2", "daughter of name1", "daughter of name2"]) to batch reverse-child lookups in one step.
  NEVER rely solely on forward article reading for downward traversal — forward-only search misses the majority of descendants.

⚠️ RELATIONSHIP VERIFICATION: When a search returns an entity with the same surname as X, verify the article EXPLICITLY mentions X or states a relationship to X. A same-surname result is NOT a relative unless explicitly connected. If unsure, try "[X full name] parent" or "[X full name] sibling" as more specific queries.

⚠️ MISSING ENTITY PROTOCOL: If an intermediate entity (e.g., a parent or grandparent) cannot be found after 2 different query attempts, record '[entity_name]: UNKNOWN' in your notes and SKIP this branch. Do NOT substitute a same-surname entity as a stand-in — this causes hallucination errors.

⚠️ DEAD-END PIVOT: If you have made 3+ different searches for information about a single entity and found nothing useful, STOP immediately — do NOT keep trying. Take one of these actions:
  (a) Mark this entity as UNKNOWN and move on to the next entity in your list
  (b) If you suspect more anchor entities exist with the same attribute (e.g., more people with the same DOB or hobby), pivot to run additional anchor-search phrasings before continuing traversal
  Spending 5+ searches on the same dead-end entity wastes your entire iteration budget.

⚠️ TERMINATION RULE (per-hop): After completing hop K, ask: "Is this the LAST intermediate hop before applying the FINAL relation at STEP 3?"
  - If YES → proceed to STEP 3 immediately
  - If NO → continue to hop K+1
  Do NOT search one extra generation to "verify" — going further reveals a DIFFERENT generation and creates over-counting errors.

Repeat this loop for each hop level:
  For EACH entity in current hop's entity list (read from your notes):
    a. State: "Processing entity [N] of [total]: [name]. Hop [K] of [M]."
    b. Search for that entity's [hop-K relation] using the TRAVERSAL METHOD above (not a single combined query); use batch_lookup for groups of 4+ entities
    c. Use append_notes('hop_K_results', '[entity] → [results]')  ← ALWAYS use append_notes here, NOT take_notes
  ⚠️ Complete ALL entities before advancing to the next hop.

⚠️ CRITICAL: Do NOT apply the FINAL relation until ALL intermediate hops are complete for ALL entities.
⚠️ CRITICAL: Applying the final relation one hop early (hop 2 instead of hop 3) is the most common mistake — re-read hop_plan before each search.

STEP 3 — APPLY FINAL RELATION TO ALL:
For EACH entity in your last intermediate hop note (process ALL of them, one by one):
  a. State: "Applying final relation to entity [name]."
  b. Search for the final relation.
  c. append_notes('final_results', '[entity]: [result]')

⚠️ GENDER FILTER: For gender-specific final relations, you MUST filter at this step:
  • uncle = male siblings ONLY (NOT female, NOT all siblings)
  • aunt = female siblings ONLY
  • brother = male siblings ONLY
  • sister = female siblings ONLY
  • nephew = male children of siblings ONLY
  • niece = female children of siblings ONLY
  • grandson = male grandchildren ONLY
  • granddaughter = female grandchildren ONLY
  • second uncle = male siblings of great-grandparents ONLY
  • second aunt = female siblings of great-grandparents ONLY
  Explicitly check each entity's gender from their wiki article before including them.

Then compile the answer:
• COUNT: For COUNT questions, use append_notes('entity_counts', '[entity_name]: COUNT=N') for EACH entity as you process it. At the end, read all entity_counts notes and compile the SET of unique per-entity values → return as SET of strings (e.g., ['0','2','3']). NEVER return a global total — COUNT means per-individual count, never a sum. NEVER return just one count if multiple entities exist.
• ENTITY: collect all names from every entity → return full union (no duplicates).
• ATTRIBUTE: collect all values from every entity → return full union.

⚠️ SINGULAR PHRASING RULE: Questions using "Who is the X?" or "What is the X?" may have MULTIPLE valid answers. NEVER reduce your answer to 1 entity because the question uses "the" or singular phrasing. If your notes contain 5 female cousins, your answer MUST contain all 5. Grammatical number in the question does NOT determine cardinality.

STEP 4 — COMPLETENESS CHECK:
Before calling finish(), read all notes and verify:
1. Did you process EVERY entity at each hop? (Not just 1-2)
2. For attribute-value anchor: did you use search_wiki_multi with 5+ distinct phrasings? Find 5+ anchor entities?
3. For COUNT: re-read your entity_counts note. Does your answer include the FULL SET of distinct values from ALL processed entities? If you computed values for 10 entities, your answer must contain every distinct count value — even if the question uses singular phrasing.
4. Was the final relation applied at the LAST hop only?
5. Did each kinship hop use the step-by-step TRAVERSAL METHOD, not a single combined query?
6. Did you apply the gender filter for gender-specific relations (uncle=male only, aunt=female only, etc.)?
7. For children hops: did you run search_wiki_multi(["son of [name]", "daughter of [name]", "child of [name]"]) to find all children? (NOT "parent [name]" — that returns the entity's own page, not their children!)
8. Is the answer list deduplicated? Remove any repeated values — each distinct value must appear exactly once in the final answer.
Only call finish() after confirming completeness."""

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="ALL correct answers — complete set. For COUNT questions: numbers only (e.g., ['3', '7']), never names. For WHO/WHAT questions: all matching names or values."
    )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        # Thread-local storage for notes (evaluation uses multiple threads)
        self._local = threading.local()

        # k=10 for broader document coverage per search
        self.retrieve = dspy.Retrieve(k=10)

        self.react = dspy.ReAct(
            signature=AnswerQuestion,
            tools=[self.search_wiki, self.search_wiki_deep, self.search_wiki_multi, self.batch_lookup, self.take_notes, self.append_notes, self.read_notes],
            max_iters=75,
        )

    # ------------------------------------------------------------------
    # Thread-safe notes property
    # ------------------------------------------------------------------

    @property
    def _notes(self):
        if not hasattr(self._local, "notes"):
            self._local.notes = {}
        return self._local.notes

    @_notes.setter
    def _notes(self, value):
        self._local.notes = value

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages about the queried topic.
        Tips: Search by person name for full articles, or by attribute for entity discovery."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_deep(self, query: str) -> str:
        """Deep search using 30 results instead of 10. Use for attribute-value anchor enumeration (e.g., finding ALL people whose date of birth is X, whose hobby is Y, whose occupation is Z). Returns more results to maximize entity discovery for multi-entity attribute anchors."""
        retrieve_deep = dspy.Retrieve(k=30)
        results = retrieve_deep(query)
        return "\n\n".join(results.passages)

    def search_wiki_multi(self, queries: list) -> str:
        """Search using multiple query phrasings and return deduplicated results.
        Use this for attribute-value anchors (DOB, hobby, occupation) to maximize entity discovery.
        Accepts a Python list of up to 8 query strings. Each query runs at k=10, results are deduplicated by entity name.
        Example: search_wiki_multi(['hobby stone collecting', 'stone collection enthusiast', 'collects stones', 'interest in stone collecting', 'stone collector'])"""
        seen_titles = set()
        all_passages = []
        for query in queries[:8]:
            if not isinstance(query, str) or not query.strip():
                continue
            try:
                results = self.retrieve(query.strip())
                for passage in results.passages:
                    # Dedup by first line (entity title)
                    title = passage.split('\n')[0].strip()
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_passages.append(passage)
            except Exception:
                continue
        return "\n\n".join(all_passages) if all_passages else "No results found."

    def batch_lookup(self, entity_names: list) -> str:
        """Retrieve wiki articles for multiple named entities in a single call.
        Use this during traversal loops to look up many entities efficiently (saves iteration budget).
        Accepts a Python list of up to 8 entity name strings.
        Example: batch_lookup(['Alice Smith', 'Bob Jones', 'Carol White'])
        Returns each entity's wiki article separated by a header line."""
        all_passages = []
        for name in entity_names[:8]:
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                results = self.retrieve(name.strip())
                if results.passages:
                    all_passages.append(f"=== {name.strip()} ===\n{results.passages[0]}")
            except Exception:
                continue
        return "\n\n".join(all_passages) if all_passages else "No results found."

    def take_notes(self, key: str, note: str) -> str:
        """Save a finding or plan to your notes workspace.
        key: short identifier (e.g., 'anchor_entities', 'results_per_anchor', 'todo')
        note: what you found or plan to investigate next"""
        self._notes[key] = note
        return f"Saved note '{key}'. You now have {len(self._notes)} note(s) total."

    def append_notes(self, key: str, note: str) -> str:
        """Append new findings to an existing note. Use this when accumulating results across multiple entities (e.g., building up hop_1_results for many anchors).
        key: the note key to append to (e.g., 'hop_1_results', 'final_results')
        note: the new information to add (will be added on a new line below any existing content)"""
        existing = self._notes.get(key, "")
        if existing:
            self._notes[key] = existing + "\n" + note
        else:
            self._notes[key] = note
        return f"Appended to note '{key}'."

    def read_notes(self, key: str = "all") -> str:
        """Read from your notes workspace.
        key: 'all' to see all notes, or a specific key to read one note"""
        if key == "all":
            if not self._notes:
                return "No notes saved yet."
            return "\n".join(f"[{k}]: {v}" for k, v in self._notes.items())
        return self._notes.get(key, f"No note found with key '{key}'.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, question):
        self._notes = {}  # Reset notes for each new question
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
