import glob
import os
import pickle
import re
import threading

import dspy


# ---------------------------------------------------------------------------
# Non-semantic exact-match lookup support
# ---------------------------------------------------------------------------
# The ColBERT retriever does semantic (fuzzy) matching which fails to find
# exact attribute values — dates of birth ("0946-07-14") return ZERO hits, and
# occupations/hobbies return only the top-k ranked by similarity, missing the
# vast majority of matching people.  A pre-built inverted index
# (_corpus_index.pkl) maps attribute values → person names and person names →
# prolog fact lists, enabling an exact-match retrieval tool that finds ALL
# people with a given attribute.  See memory iterations 4-15 for the
# decade-long retrieval cap on date-anchored and attribute-anchored questions.

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_FACT_RE = re.compile(r'(\w+)\("([^"]+)",\s*"([^"]+)"\)\.')

_REL_SUFFIX = re.compile(
    r"\s+(?:son|daughter|sons|daughters|father|mother|husband|wife|sister|sisters|"
    r"brother|brothers|child|children|parent|parents|spouse|sibling|siblings|"
    r"cousin|cousins|niece|nephew|aunt|uncle|grandparent|grandmother|grandfather|"
    r"grandchild|grandson|granddaughter|family|relatives?)\s+of\b.*$",
    re.IGNORECASE,
)


def _normalize_query(query):
    """Normalize a search query for dedup: strip quotes and relation suffixes
    like 'son of', 'daughter of' so that '"Logan Kong" son of' and
    'Logan Kong' share the same cache entry."""
    q = query.strip().strip("\"'").strip()
    q = _REL_SUFFIX.sub("", q).strip()
    q = q.strip("\"'").strip()
    return q.lower() if q else query.strip().lower()

_FAMILY_RELATION_TEXT = {
    "mother": "The mother of {name} is {value}.",
    "father": "The father of {name} is {value}.",
    "sister": "The sister of {name} is {value}.",
    "brother": "The brother of {name} is {value}.",
    "son": "The son of {name} is {value}.",
    "daughter": "The daughter of {name} is {value}.",
    "husband": "The husband of {name} is {value}.",
    "wife": "The wife of {name} is {value}.",
}
_ATTR_RELATION_TEXT = {
    "dob": "The date of birth of {name} is {value}.",
    "job": "The occupation of {name} is {value}.",
    "hobby": "The hobby of {name} is {value}.",
    "gender": "The gender of {name} is {value}.",
}
_FAMILY_RELS = set(_FAMILY_RELATION_TEXT)


def _format_article(name, facts):
    """Convert a list of prolog fact strings into article-style text matching
    the ColBERT corpus format (## Family / ## Friends / ## Attributes)."""
    family_lines = []
    friend_names = []
    attr_lines = []
    for fact in facts:
        m = _FACT_RE.match(fact.strip())
        if not m:
            continue
        rel, n, val = m.group(1), m.group(2), m.group(3)
        if n != name:
            continue
        if rel in _FAMILY_RELS:
            family_lines.append(
                _FAMILY_RELATION_TEXT[rel].format(name=name, value=val)
            )
        elif rel == "friend":
            friend_names.append(val)
        elif rel in _ATTR_RELATION_TEXT:
            attr_lines.append(
                _ATTR_RELATION_TEXT[rel].format(name=name, value=val)
            )
    parts = [f"# {name}"]
    if family_lines:
        parts.append("\n## Family")
        parts.extend(family_lines)
    if friend_names:
        parts.append("\n## Friends")
        parts.append(f"The friends of {name} are {', '.join(friend_names)}.")
    if attr_lines:
        parts.append("\n## Attributes")
        parts.extend(attr_lines)
    return "\n".join(parts)


def _load_corpus_index():
    """Locate and load the pre-built _corpus_index.pkl.  Returns a dict with
    keys 'dob', 'hobby', 'job' (inverted indices: value → [names]) and
    'name2facts' (name → [fact strings]), or None if unavailable."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(module_dir))
    candidates = glob.glob(
        os.path.join(project_root, "output", "*", "_corpus_index.pkl")
    )
    if not candidates:
        return None
    try:
        with open(candidates[0], "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


class PhantomWikiPlanSignature(dspy.Signature):
    """You are a planning module. Given a PhantomWiki question, decompose it into an
    explicit, ordered hop-by-hop plan that a retrieval agent will execute. You CANNOT
    search the corpus, so do NOT try to answer the question. Only produce the
    STRUCTURAL plan: the ordered base-relation hops, the hop count, and the answer type.

    PhantomWiki articles list only BASE relations: mother, father, sister, brother,
    son, daughter, husband, wife, friend; and ATTRIBUTES: date of birth (YYYY-MM-DD),
    occupation, hobby, gender. Any relation named in the question that is NOT one of
    these base relations is DERIVED and must be decomposed into base hops.

    GENERATION COUNT — the most common error is MISCOUNTING generations. Be precise:
    - child of X                  = 1 hop (read X, take sons/daughters)
    - grandchild of X             = 2 hops (X -> child -> child)
    - great-grandchild of X       = 3 hops (X -> child -> child -> child)
    - great-great-grandchild of X = 4 hops
    - every extra "great-" adds exactly ONE more child-hop
    - parent of X                 = 1 hop (read X, check BOTH mother AND father)
    - grandparent of X            = 2 hops (X -> parent -> parent — 4 paths: MM, MF, FM, FF)
    - great-grandparent of X      = 3 hops
    - aunt/uncle of X             = 2 hops (X -> parent -> sibling — check BOTH parents' siblings)
    - great-aunt/great-uncle of X = 3 hops (X -> parent -> parent -> sibling)
    - second aunt of X   = 4 hops (X -> parent -> parent -> parent -> sister)
      (the sister of X's great-grandparent; "second" is a RELATION NAME, not an
      ordinal — do NOT interpret it as "the 2nd aunt by age")
    - second uncle of X  = 4 hops (X -> parent -> parent -> parent -> brother)
    - niece/nephew of X           = 2 hops (X -> sibling -> child)
    - grandniece/grandnephew of X = 3 hops
    - cousin of X                 = 3 hops (X -> parent -> sibling -> child)
    - second cousin of X          = 5 hops (X -> parent -> parent -> sibling -> child -> child)
    - "once removed" adds one child-hop; "twice removed" adds two
    - mother-in-law of X = mother of spouse = 2 hops (X -> spouse -> mother)
    - father-in-law of X = father of spouse = 2 hops (X -> spouse -> father)
    - brother-in-law of X = brother of spouse = 2 hops (X -> spouse -> brother)
    - sister-in-law of X = sister of spouse = 2 hops (X -> spouse -> sister)
    - son-in-law of X = husband of X's child = 2 hops (X -> child -> husband)
    - daughter-in-law of X = wife of X's child = 2 hops (X -> child -> wife)

    HOW TO BUILD THE PLAN:
    1. Identify the anchor: a named person, OR an attribute value (a date like
       "0918-01-17", an occupation, a hobby, a gender). If the anchor is an attribute
       value, the first hop is "search that attribute value and enumerate ALL people
       whose attribute matches it exactly".
    2. Decompose every derived relation in the question into the ordered base hops
       above. Count the hops precisely — "great-grandchild" is 3 child-hops, NOT 2.
    3. State the answer TYPE:
       - "Who is/are ..." or "Which ..." -> a SET of full names. Enumerate ALL, branch
         at every hop, do not stop after finding one.
       - "How many X does Y have?" -> if Y is a SINGLE anchor, the answer is ONE count
         digit string (e.g. ["3"]). If Y is itself MULTI-VALUED (the question chains
         through a relation that yields MULTIPLE Y's, e.g. "the grandchild of the
         great-grandchild of Z" -> many great-grandchildren -> many grandchildren),
         compute the count for EACH Y and return the SET of distinct count strings
         (e.g. ["0","2","3"]).
       - "What is the <attribute> of ..." -> a single attribute value.
    4. Flag whether branching/enumeration is needed at each hop (yes for any
       "who/which/how many" question and any multi-valued-Y count question).

    Keep the plan concise: ordered hops, hop count, answer type, branching flag.
    This plan is the agent's roadmap — it will traverse the hops in order.
    """

    question: str = dspy.InputField()
    plan: str = dspy.OutputField()


class PhantomWikiQASignature(dspy.Signature):
    """You answer a multi-hop question over PhantomWiki, a fictional knowledge base of
    people. Each search returns one or more short articles; each article is about ONE
    person and lists, under "## Family", the people linked to them by BASE relations:
    mother, father, sisters, brothers, sons, daughters, husband, wife; under
    "## Friends", their friends; under "## Attributes", their date of birth
    (YYYY-MM-DD), occupation, hobby, and gender.

    The question chains relations across several hops (e.g. "the brother of the child
    of the person whose occupation is structural engineer"). Many relations named in the
    question are NOT written verbatim anywhere -- aunt, uncle, cousin, grandparent,
    grandchild, great-grandparent, niece, nephew, in-law, mother-in-law, second cousin,
    cousin once removed, second aunt, etc. are all DERIVED. Derive them by chaining the
    BASE relations that ARE in articles: mother, father, sister, brother, son, daughter,
    husband, wife, friend. Example: the mother-in-law of X = the mother of X's spouse, so
    first read X's article to find X's husband/wife, then search that spouse's name and
    read the spouse's article to find the spouse's mother.

    FOLLOW THE PRE-COMPUTED PLAN (the `plan` field provided with each question):
    - A structural hop plan has already been generated for this question. It lists the
      ordered base hops, the exact hop count, the answer type, and whether branching is
      needed. Follow it: traverse the hops in the listed order and count generations
      EXACTLY as the plan states (a "great-grandchild" is 3 child-hops, not 2; a
      "great-great-grandchild" is 4, not 3).
    - Respect the answer type the plan specifies: a SET of names (enumerate ALL, branch at
      every hop), a SINGLE count, or a SET of distinct counts (when the counted entity Y is
      itself multi-valued, compute the count for EACH Y and return every distinct count).
    - The plan is a roadmap, not the answer. You must still search the corpus and read
      articles to resolve each hop. If the plan says the anchor is an attribute value,
      search that value first and enumerate ALL matching people before proceeding.

    HOW TO SEARCH AND TRAVERSE:
    - When you retrieve a person's article, READ it and extract the names/relations that
      are written directly inside it. Do NOT issue a separate search like "X husband" or
      "X children" to find a relation that is already listed in X's own article -- the
      spouse, children, parents, and siblings of X are right there in X's article.
    - If the question names a person, first search that person's full name to retrieve
      their article, then read the relevant listed relation to advance one hop.
    TWO SEARCH TOOLS — choose the right one for each hop:
    - search_wiki(query): SEMANTIC search. Use for NAME searches — pass a person's full
      name to retrieve their article. Returns the top-ranked passages.
    - search_exact(query): EXACT-MATCH lookup for ATTRIBUTE VALUES. Use when the anchor
      (or an intermediate hop) is an attribute value — a date of birth (YYYY-MM-DD like
      "0917-08-17"), an occupation (like "structural engineer"), or a hobby. Pass ONLY the
      attribute value as the query (e.g. search_exact("0917-08-17") or
      search_exact("structural engineer")). Returns the full articles of ALL people in the
      corpus whose attribute matches exactly — this finds people that semantic search
      misses (exact dates return zero hits with search_wiki, and occupations/hobbies only
      return the top few). When the question says "the person whose [attribute] is
      [value]", call search_exact(value) FIRST.
    - For NAME hops after the anchor, use search_wiki to retrieve each next-hop person's
      article by full name, read it, and extract the next-hop entities.
    - Use base relations directly: the CHILDREN of X are the sons/daughters listed in X's
      article; the PARENTS of X are the mother/father in X's article; the SIBLINGS of X
      are the sisters/brothers in X's article; the SPOUSE of X is the husband/wife in X's
      article; the FRIENDS of X are listed in X's article.
    - For a derived relation, decompose it into base hops and traverse one hop at a time:
      search a name, read its article, collect the next-hop entities, then search each of
      those names, and so on, branching at every hop.
    - Issue multiple, varied queries to surface documents a single query might miss, but
      never repeat the same query expecting different results.

    TRACK PROGRESS WITH THE note TOOL (avoids losing the thread on deep chains):
    - The note(text) tool appends text to a persistent workspace and returns the FULL
      workspace back to you, so you can reorient without re-deriving earlier hops.
    - At the START, call note() with the hop plan: the ordered base hops the chain
      requires (e.g. "PLAN: anchor=DOB 0917-08-17 -> child -> brother -> hobby"). Count
      the hops: a derived relation N hops deep needs N base-hop traversals.
    - After EACH resolved hop, call note() with the entities found there (e.g. "HOP2
      children of X: A, B, C"). Use the returned workspace to track what is confirmed
      and which next-hop entities still need to be searched.
    - Do NOT re-search an anchor or hop you have already resolved -- re-read your
      workspace instead. Re-searching a resolved hop wastes steps and loses the thread.
    - Before finishing, call note() and confirm every hop is resolved AND every branch
      has been followed to its final entity and READ. Never finish with an empty answer
      if you have already found the final entity: retrieve its article and report its
      attribute value.

    ENUMERATION DISCIPLINE (CRITICAL):
    - Most questions have MULTIPLE correct answers. Recall is scored just as heavily as
      precision, so you must find ALL of them, not just one or a few.
    - At every hop, enumerate EVERY entity that satisfies that hop's relation, then branch
      and follow EACH one through the remaining hops. Do not pick a single branch.
    - Keep a running, deduplicated set of candidate final answers as you go (record it
      with note()).
    - Do NOT stop early just because you have found some answers. Continue searching until
      you have traced every branch from the anchor all the way through every hop. Only
      call "finish" once you have systematically covered every hop for every branch and
      are confident the answer set is complete.

    AGGREGATION QUESTIONS ("How many X does Y have?"):
    - Enumerate ALL the X for the given Y, then report the distinct COUNT as a string
      digit, e.g. ["3"]. NEVER return a person's name as the answer to a "How many"
      question -- the answer must be a count.
    - If Y is itself multi-valued (several Y's), compute the count for each Y and return
      the SET of distinct counts (e.g. ["0", "2", "3"]).

    ANSWER FORMAT:
    - Return `answer` as a list of strings. Use full names exactly as they appear in
      articles (e.g. "Alison Lanza"). Use dates in YYYY-MM-DD format. Use counts as digit
      strings. Include only entities that satisfy the COMPLETE chain of relations, and
      omit entities that match only some hops, so that precision stays high. If after
      thorough searching no entity satisfies the full chain, return an empty list.
    """

    question: str = dspy.InputField()
    plan: str = dspy.InputField(
        desc="A pre-computed structural hop plan: the ordered base-relation hops, "
        "the exact hop count, the answer type, and whether branching is needed. "
        "Follow it to traverse the chain in the correct order."
    )
    answer: list[str] = dspy.OutputField()


class PhantomWikiVerifySignature(dspy.Signature):
    """You are a VERIFICATION agent for a multi-hop PhantomWiki question. A previous
    agent already investigated the question and produced a PLAN (ordered base-relation
    hops + hop count + answer type), a WORKSPACE (its hop-by-hop notes), and a DRAFT
    ANSWER. PhantomWiki articles list only BASE relations (mother, father, sister,
    brother, son, daughter, husband, wife, friend) and ATTRIBUTES (date of birth
    YYYY-MM-DD, occupation, hobby, gender); derived relations (aunt, cousin,
    grandchild, in-law, etc.) are chains of base relations.

    Your job: VERIFY COMPLETENESS and return the FINAL, COMPLETE answer. The single
    most common failure is STOPPING EARLY -- the previous agent found SOME matching
    entities but not ALL of them. Your value is finding what it MISSED, not redoing
    what it already did.

    PROCEDURE:
    1. Read the PLAN to learn the ordered hops, hop count, and answer TYPE (a SET of
       names, a SINGLE count, or a SET of distinct counts when the counted entity Y is
       itself multi-valued).
    2. Read the WORKSPACE to see which hops are already resolved. Do NOT redo a
       resolved hop or re-search an entity already recorded there -- that wastes your
       limited steps. Re-read the workspace instead.
    3. Identify GAPS: which next-hop names were found but never searched, which
       branches at a branching hop were not followed, or whether an attribute anchor
       was incompletely enumerated.
    4. SEARCH only for those gaps: retrieve articles for un-searched next-hop names
       with search_wiki, or re-enumerate an attribute value whose matches were
       incomplete using search_exact (which finds ALL people with a given attribute
       value — dates, occupations, hobbies — that semantic search missed).
    5. Use note() to record anything new you find, so the workspace stays current.

    COMPLETENESS RULES:
    - "Who/Which" (SET of names): every entity satisfying the COMPLETE chain must be
      in the answer. Search for any the draft missed; branch at every hop. Remove a
      draft name ONLY if you confirm from its article that it does NOT satisfy the full
      chain (a precision check).
    - "How many" (SINGLE count): re-derive the one correct count by enumerating every
      matching entity, then return it as a single digit string (e.g. ["3"]). The draft
      may be wrong, so verify and correct it.
    - Multi-valued-Y count (SET of distinct counts): counts are DERIVED aggregates,
      not directly retrievable from any single article, so you CANNOT reliably verify
      a draft count by lookup. Therefore KEEP every count already in the draft and ONLY
      ADD newly confirmed counts for branches the draft missed. NEVER remove a draft
      count -- a count the first agent found is evidence it traced that branch; trust
      it and focus on finding branches that yield counts the draft is MISSING.

    If the draft is already complete, return it unchanged. Return `final_answer` as a
    list of strings: full names exactly as in articles, dates as YYYY-MM-DD, counts as
    digit strings. For names include only those satisfying the COMPLETE chain; for
    multi-valued counts keep every draft count plus any newly confirmed ones.
    """

    question: str = dspy.InputField()
    plan: str = dspy.InputField(
        desc="The pre-computed hop plan from the planner: ordered base hops, hop "
        "count, answer type, branching flag. Use it as your completeness checklist."
    )
    workspace: str = dspy.InputField(
        desc="The previous agent's accumulated hop-by-hop notes. Re-read it to avoid "
        "redoing resolved hops."
    )
    draft_answer: list[str] = dspy.InputField()
    final_answer: list[str] = dspy.OutputField()


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        # Pre-loop planner: decomposes the question into an ordered base-hop plan
        # with the correct generation count and answer type, before the ReAct loop
        # runs. Targets the dominant hop-counting failure mode (e.g. miscounting
        # "great-grandchild" as 2 hops instead of 3) without a heavy control overlay.
        self.planner = dspy.ChainOfThought(PhantomWikiPlanSignature)
        # Per-thread scratchpad so concurrent eval workers do not clobber each
        # other's progress notes. Each worker handles one question at a time, so
        # resetting in forward() is safe and isolates state per question.
        self._tls = threading.local()
        # Non-semantic exact-match lookup index: maps attribute values (dates,
        # occupations, hobbies) to person names and person names to fact lists.
        # Enables the search_exact tool that finds ALL people with a given
        # attribute value — fixing the ColBERT retrieval cap on date-anchored
        # and attribute-anchored questions (the dominant remaining failure class).
        # Loaded once at init (~1.5s); None if the index file is unavailable.
        self._corpus_idx = _load_corpus_index()
        if self._corpus_idx is not None:
            self._job_index_lower = {
                k.lower(): v for k, v in self._corpus_idx["job"].items()
            }
            self._hobby_index_lower = {
                k.lower(): v for k, v in self._corpus_idx["hobby"].items()
            }
        else:
            self._job_index_lower = {}
            self._hobby_index_lower = {}
        # Phase 1: investigation ReAct loop.
        self.react = dspy.ReAct(
            signature=PhantomWikiQASignature,
            tools=[self.search_wiki, self.search_exact, self.note],
            max_iters=50,
        )
        # Phase 2: a short verification pass seeded with phase 1's workspace +
        # draft answer and the plan, with a completeness mandate. Targets the
        # open early-stopping / enumeration-incompleteness failure mode (the
        # agent often finds SOME answers but not ALL on multi-answer questions)
        # by giving it a second chance to find missed branches WITHOUT re-deriving
        # resolved hops. The shared `note` workspace provides continuity between
        # phases. Kept short (max_iters=15) to bound cost; the verifier is told to
        # start from the draft and only fill gaps, not restart the investigation.
        self.verify_react = dspy.ReAct(
            signature=PhantomWikiVerifySignature,
            tools=[self.search_wiki, self.search_exact, self.note],
            max_iters=15,
        )

    def _workspace(self):
        if not hasattr(self._tls, "notes"):
            self._tls.notes = []
        return self._tls.notes

    def _search_cache(self):
        if not hasattr(self._tls, "search_cache"):
            self._tls.search_cache = {}
        return self._tls.search_cache

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        norm = _normalize_query(query)
        cache = self._search_cache()
        if norm in cache:
            return (
                f"[CACHED — already searched '{norm}'. The article with family "
                f"relations is below. If you already noted this person's family, "
                f"search a DIFFERENT person; otherwise re-read below to verify.]\n\n"
                + cache[norm]
            )
        results = self.retrieve(query)
        passages = "\n\n".join(results.passages)
        cache[norm] = passages
        return passages

    def search_exact(self, query: str) -> str:
        """Exact-match lookup for an ATTRIBUTE VALUE (date of birth YYYY-MM-DD,
        occupation, or hobby). Returns the full articles of ALL people in the
        corpus whose attribute matches the query exactly. Use this when the
        question says "the person whose [attribute] is [value]" — pass ONLY the
        value (e.g. search_exact("0946-07-14") or search_exact("financial
        controller")). For names, use search_wiki instead.
        """
        if self._corpus_idx is None:
            return (
                "Exact-match search unavailable. Use search_wiki with the "
                "attribute value as the query."
            )
        q = query.strip().strip("\"'").strip()
        names = []
        is_date = False
        date_match = _DATE_RE.search(q)
        if date_match:
            is_date = True
            names = list(self._corpus_idx["dob"].get(date_match.group(), []))
        else:
            ql = q.lower()
            names = list(self._job_index_lower.get(ql, []))
            if not names:
                names = list(self._hobby_index_lower.get(ql, []))
        if not names:
            return f"No exact matches found for '{q}'."
        total = len(names)
        # For small result sets (≤ 20, e.g. dates with at most ~18 people) return
        # FULL articles so the agent can extract family relations directly without
        # additional searches.  For large sets (occupations/hobbies with thousands
        # of matches) return a COMPACT one-line-per-person summary — full articles
        # for 50 people would be ~27K chars and overwhelm the agent, causing it to
        # trace FEWER chains than the k=7 baseline.  The compact form gives the
        # agent the names to trace with search_wiki while keeping the context small.
        if total <= 20:
            header = f"Found {total} people matching '{q}':\n\n"
            articles = []
            for name in names:
                facts = self._corpus_idx["name2facts"].get(name, [])
                if facts:
                    articles.append(_format_article(name, facts))
            return header + "\n\n---\n\n".join(articles)
        else:
            cap = 50
            header = (
                f"Found {total} people matching '{q}' (showing first {cap}). "
                f"Use search_wiki on a person's name to read their full article.\n\n"
            )
            lines = []
            for name in names[:cap]:
                facts = self._corpus_idx["name2facts"].get(name, [])
                dob = occ = hob = gen = ""
                for f in facts:
                    m = _FACT_RE.match(f.strip())
                    if not m or m.group(2) != name:
                        continue
                    rel, val = m.group(1), m.group(3)
                    if rel == "dob":
                        dob = val
                    elif rel == "job":
                        occ = val
                    elif rel == "hobby":
                        hob = val
                    elif rel == "gender":
                        gen = val
                parts = [name]
                if dob:
                    parts.append(f"DOB: {dob}")
                if occ:
                    parts.append(f"occupation: {occ}")
                if hob:
                    parts.append(f"hobby: {hob}")
                if gen:
                    parts.append(f"gender: {gen}")
                lines.append(f"{len(lines) + 1}. {' — '.join(parts)}")
            return header + "\n".join(lines)

    def note(self, text: str) -> str:
        """Record a structured progress note (hop plan, entities found per hop,
        remaining branches) and get back the FULL workspace so you can reorient.
        Call this at the start with the hop plan, after each resolved hop, and
        before finishing. Example: note("HOP2 children of Forest Benner: A, B, C").
        """
        ws = self._workspace()
        ws.append(text)
        if not ws:
            return "WORKSPACE is empty."
        return "WORKSPACE:\n" + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(ws))

    def forward(self, question):
        self._tls.notes = []
        self._tls.search_cache = {}
        plan = self.planner(question=question).plan
        # Phase 1: investigation
        result = self.react(question=question, plan=plan)
        draft = result.answer
        # Phase 2: verification & completion, seeded with phase 1's workspace so
        # the verifier continues the investigation rather than restarting it.
        ws = self._workspace()
        workspace_str = (
            "WORKSPACE:\n" + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(ws))
            if ws else "(no notes recorded)"
        )
        verified = self.verify_react(
            question=question,
            plan=plan,
            workspace=workspace_str,
            draft_answer=draft,
        )
        return dspy.Prediction(answer=verified.final_answer)
