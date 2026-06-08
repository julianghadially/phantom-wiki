import dspy


# ── Anchor Expansion ─────────────────────────────────────────────────────────

class AnchorExpansionSig(dspy.Signature):
    """Find ALL people in the wiki that match the anchor property in the question.

    Your ONLY goal is to enumerate every person who has the specific property
    mentioned in the question. Do NOT attempt to answer the full question.

    PATTERNS TO DETECT in the question:
    - "the person whose occupation is X" -> find ALL people with occupation X
    - "the person whose hobby is X"      -> find ALL people with hobby X
    - "the person whose date of birth is X" -> find ALL people born on date X
    - If no property lookup (question directly names a person): return just that person's name

    CRITICAL: In this wiki, occupations, hobbies, and birthdates are shared by
    MANY people (typically 5-15+ each). You MUST run at least 4-5 different
    search queries (varying the phrasing) to find ALL of them.
    Examples:
      - "occupation financial controller", "job financial controller", "works as financial controller"
      - "hobby microbiology", "microbiology hobby", "interest microbiology"
      - "born on 1050-09-16", "date of birth 1050-09-16", "birthday 1050-09-16", "born 1050"
    Keep searching until two consecutive searches yield no new names.
    Extract EVERY person name you see in the results that matches the property.
    """
    question: str = dspy.InputField(
        desc="Question containing a property-based entity lookup"
    )
    anchor_entities: list[str] = dspy.OutputField(
        desc="ALL person names matching the anchor property. Must be comprehensive - run multiple searches."
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

    def forward(self, question):
        # Phase 1: expand anchor entities
        try:
            anchor_result = self.anchor_expander(question=question)
            anchor_entities = anchor_result.anchor_entities or []
        except Exception:
            anchor_entities = []

        if anchor_entities:
            count = len(anchor_entities)
            anchor_context = (
                f"Found {count} anchor {'entity' if count == 1 else 'entities'}: "
                f"{anchor_entities}. Process EACH one to collect all answers."
            )
        else:
            anchor_context = (
                "No anchor entities pre-identified. Determine them during your search."
            )

        # Phase 2: chain traversal using full anchor list
        result = self.react(
            question=question,
            anchor_entities_context=anchor_context,
        )

        return dspy.Prediction(answer=result.answer)
