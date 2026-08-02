import dspy
import re
import pickle
from pathlib import Path

# Exact graph-based solvers for count + list questions (see graph_solver.py).
from src.program.graph_solver import solve_count, solve_list

# Prebuilt attribute index mapping date-of-birth / hobby / occupation strings to
# the COMPLETE list of person names with that attribute. ColBERT (a semantic
# retriever) cannot reliably find people by a bare date string, so this index is
# the only reliable way to enumerate "the person whose date of birth is DATE".
_ATTR_INDEX = None
_ATTR_INDEX_PATHS = [
    Path("output/depth_10_size_1000000/_attr_index.pkl"),
    Path(__file__).resolve().parent.parent.parent / "output/depth_10_size_1000000/_attr_index.pkl",
]


def _load_attr_index():
    global _ATTR_INDEX
    if _ATTR_INDEX is not None:
        return _ATTR_INDEX
    for p in _ATTR_INDEX_PATHS:
        if p.exists():
            try:
                _ATTR_INDEX = pickle.load(open(p, "rb"))
                return _ATTR_INDEX
            except Exception:
                pass
    return None


# Relations whose count is directly readable from a single article's Family /
# Friends section. Used by the deterministic direct-counter path.
_DIRECT_RELATIONS = {
    "friend", "friends",
    "brother", "brothers", "sister", "sisters", "sibling", "siblings",
    "son", "sons", "daughter", "daughters", "child", "children",
    "mother", "mothers", "father", "fathers", "parent", "parents",
    "husband", "husbands", "wife", "wives",
}

# Matches "the person whose (occupation|hobby) is <value>" up to the
# " have" / " has" / "?" terminator that follows it in the question.
_ANCHOR_RE = re.compile(
    r"the person whose (occupation|hobby|date of birth) is (.+?)\s*(?:have|has|\?)", re.IGNORECASE
)


def _as_list(ans):
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(x) for x in ans]
    return [str(ans)]


def _dedupe(items):
    """Case-insensitive dedupe, preserving the first-seen spelling."""
    seen = set()
    out = []
    for it in items:
        k = str(it).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(str(it).strip())
    return out


def _is_count_question(q):
    return re.match(r"\s*how\s+many\b", q, re.IGNORECASE) is not None


def _count_relation_from_passage(passage, rel):
    """Deterministically count a DIRECT base relation from one article passage.

    Returns an integer count. ``rel`` is the lowercase relation name
    (singular or plural). For composed relations this is not used.
    """
    name = passage.split(":")[0].strip()
    fam_m = re.search(r"## Family(.*?)## Friends", passage, re.S)
    fri_m = re.search(r"## Friends(.*?)## Attributes", passage, re.S)
    fam = fam_m.group(1) if fam_m else ""
    fri = fri_m.group(1) if fri_m else ""

    def cnt(section, word):
        pat = re.compile(
            r"The " + word + r"s? of " + re.escape(name) + r" (?:are|is) ([^.]*?)\.",
            re.IGNORECASE,
        )
        mm = pat.search(section)
        if not mm:
            return 0
        names = mm.group(1).strip()
        if not names:
            return 0
        return len([n for n in names.split(",") if n.strip()])

    r = rel.lower()
    if r in ("friend", "friends"):
        return cnt(fri, "friend")
    if r in ("brother", "brothers"):
        return cnt(fam, "brother")
    if r in ("sister", "sisters"):
        return cnt(fam, "sister")
    if r in ("son", "sons"):
        return cnt(fam, "son")
    if r in ("daughter", "daughters"):
        return cnt(fam, "daughter")
    if r in ("mother", "mothers"):
        return cnt(fam, "mother")
    if r in ("father", "fathers"):
        return cnt(fam, "father")
    if r in ("husband", "husbands"):
        return cnt(fam, "husband")
    if r in ("wife", "wives"):
        return cnt(fam, "wife")
    if r in ("sibling", "siblings"):
        return cnt(fam, "brother") + cnt(fam, "sister")
    if r in ("child", "children"):
        return cnt(fam, "son") + cnt(fam, "daughter")
    if r in ("parent", "parents"):
        return cnt(fam, "mother") + cnt(fam, "father")
    return 0

INSTRUCTIONS = """\
You answer questions over the PhantomWiki corpus of fictional characters.

CORPUS STRUCTURE. Every document is a short article about ONE fictional person. An article
lists ONLY that person's BASE relations and attributes:
  - Base family: mother, father, husband/wife, brother(s), sister(s), son(s), daughter(s).
  - Friends: a list of the person's friends.
  - Attributes: date of birth, occupation, hobby, gender.
It does NOT list composed relations (grandparent, aunt, uncle, cousin, niece, nephew,
in-law, "second cousin", "great-aunt", "once removed", etc.) — those must be DERIVED (see
below). To read about person X, search for X's full name.

ANSWERING PROTOCOL. Follow these steps IN ORDER for every question. Skipping a step is the
main cause of missing answers.
  1. PARSE. Identify (a) the ANCHOR — "the person whose <attribute> is <value>" (there may be
     MANY such people), (b) the RELATION CHAIN between the anchor and the asked quantity, and
     (c) the ASKED QUANTITY (a name set, an attribute set, or a count set).
     SELF-REFERENTIAL SHORTCUT: if the question asks for <attribute> of "the person whose
     <SAME attribute> is <value>" (e.g. "What is the date of birth of the person whose date
     of birth is 0954-03-04?"), every such person has that attribute = <value> by definition,
     so the answer is exactly ["<value>"]. Return ["<value>"] and finish.
  2. ENUMERATE ANCHORS. Find EVERY person matching the anchor. If the anchor is a DATE OF
     BIRTH ("the person whose date of birth is DATE"), you MUST call find_persons_by_dob(DATE)
     to get the COMPLETE list of matching people — ColBERT search on a bare date string misses
     almost all of them, so search_wiki/search_wiki_all on the date will NOT work. For an
     occupation/hobby anchor, use search_wiki_all(<value>) to surface up to 100 matches at
     once, then run more searches with varied phrasings if anything may be missed. Keep a
     WRITTEN running list of every matching person you have confirmed. NEVER assume there is
     only one. NEVER stop after the first.
  3. FAN OUT PER BRANCH. For EACH anchor person, derive each relation hop (use search_wiki to
     look up each person's full-name article). Every hop can BRANCH (several children, several
     siblings, two parents). Follow EVERY branch. Keep a WRITTEN list of the intermediate
     people produced at each hop, per branch, so no branch is silently dropped.
  4. COLLECT. At the leaves, extract the asked value from each person's article and add it to
     the answer set. For "how many" questions, compute the count for EACH anchor separately
     and collect the DISTINCT set of counts (different anchors almost always give DIFFERENT
     counts — finding that the first few give 0 does NOT mean the rest give 0; keep going).
  5. PRE-FINISH CHECK. Before calling finish, answer these aloud and only finish if ALL are
     yes: (i) Did I find every anchor? (ii) For every anchor and every hop, did I follow every
     branch — name any branch I have NOT looked up. (iii) Did I add every verified leaf value
     to the answer list? If any answer is no or "not sure", KEEP GOING, do not finish.

RELATION DERIVATION. Composed relations are reached by chaining base relations across
several article lookups. Base relation primitives in an article: mother, father, parent
(=mother or father), brother, sister, sibling (=brother or sister), son, daughter, child
(=son or daughter), husband, wife, married (=husband or wife), male, female. The EXACT
derived-relation definitions (X is the subject "of X"; find Y):
  - grandparent of X = a parent of X's parent.        grandchild of X = X is that person's
    grandparent, i.e. a child of X's child (go DOWN: find X's children, then their children).
  - great_grandparent of X = a parent of X's grandparent (3 generations up).
    great_grandchild of X = a child of X's grandchild (go DOWN via children's children's children).
  - uncle of X = a BROTHER of X's parent.   aunt of X = a SISTER of X's parent.
  - great_uncle of X = a BROTHER of X's grandparent.  great_aunt of X = a SISTER of X's grandparent.
  - second_uncle of X = a BROTHER of X's great-grandparent. second_aunt of X = a SISTER of X's
    great-grandparent.
  - cousin of X = a person Y whose parent is a SIBLING of X's parent (X=/=Y).
  - female_cousin / male_cousin of X = a cousin of X who is female / male.
  - second cousin of X = a person Y whose parent is a COUSIN of X's parent (X=/=Y).
    female_second_cousin / male_second_cousin add the gender filter on Y.
  - first cousin once removed of X = a CHILD of X's cousin. female_first_cousin_once_removed =
    a DAUGHTER of X's cousin; male_first_cousin_once_removed = a SON of X's cousin.
  - niece of X = a DAUGHTER of X's sibling.   nephew of X = a SON of X's sibling.
  - grandson of X = a male grandchild of X (child of X's child, male).
    granddaughter of X = a female grandchild of X.
  - mother_in_law of X = the MOTHER of X's spouse.   father_in_law of X = the FATHER of X's spouse.
  - sister_in_law of X = a SISTER of X's spouse.   brother_in_law of X = a BROTHER of X's spouse.
  - son_in_law of X = the HUSBAND of X's child.   daughter_in_law of X = the WIFE of X's child.
Direction matters: "who is the grandparent/uncle/... of Z" goes UP (find Z's ancestors, then
their siblings/parents); "who is the grandchild/niece/nephew/... of Z" goes DOWN (find Z's
descendants). "How many X does Z have?" counts the Y satisfying relation X(Z, Y).
At every hop, look up the person's article, read their base relations, and branch out
wherever a relation yields several people. To find someone's PARENTS or SIBLINGS when their
own article omits them, search that person's name to find OTHER articles that mention them
as a son/daughter/brother/sister. Keep a running list of intermediate entities so no branch
is lost; re-search a name rather than relying on memory.

SEARCH TOOLS. You have three retrievers over the same corpus.
  - search_wiki(query): returns up to 20 matching passages. Use this to READ a specific
    person — search their FULL NAME (e.g. "Alan Denney") — or for any quick lookup.
  - search_wiki_all(query): returns up to 100 matching passages. Use this EXCLUSIVELY when
    you must find EVERY person matching an attribute VALUE (an occupation, a hobby, or any V
    in "the person whose <attribute> is <value>"). One search_wiki_all(V) call surfaces far
    more matches than search_wiki, so prefer it for exhaustive enumeration of anchors; then
    keep ONLY passages whose article truly states the attribute EQUALS/CONTAINS V (watch for
    substring false positives like "leaves", "surveyor").
  - find_persons_by_dob(date): returns the COMPLETE list of every person whose date of birth
    equals `date` (e.g. "0945-06-12"), one full name per line. ColBERT search on a bare date
    string does NOT reliably find people by date of birth, so for ANY question anchored on
    "the person whose date of birth is DATE" you MUST call find_persons_by_dob(DATE) FIRST to
    enumerate ALL matching people, then call search_wiki on each returned full name to read
    that person's article. Never search a date string with search_wiki to find people — it
    returns the wrong dates.
  - Dates of birth: always use find_persons_by_dob to enumerate by date of birth. Verify any
    candidate's stated date of birth matches exactly before using it.

RELATIONS RETURN SETS, NOT ONE. A phrase like "the grandchild of X", "the cousin of X",
"the friend of X", "the grandfather of Y" denotes the ENTIRE SET of people satisfying that
relation, which is almost always MULTIPLE. This is true even when one member is obvious:
  - Even if the starting person is THEMSELVES a member (e.g. Deon is a great-grandchild of
    Hilton), Hilton typically has OTHER great-grandchildren through his other children and
    grandchildren. You MUST look up every other branch and collect them all.
  - Finding one valid answer is NEVER a reason to finish. After the obvious member, explicitly
    enumerate the OTHER branches (other children, other siblings, other parents) and collect a
    member from each. Returning a single answer when the set has several scores 0 for the rest.

MOST QUESTIONS HAVE MULTIPLE CORRECT ANSWERS — partial recall scores zero for the missing
ones, so exhaustive enumeration is the single most important thing you do.
  - "the person whose [attribute] is V" usually matches MANY people. You MUST find ALL of
    them. Use search_wiki_all(V), read every returned passage, keep ONLY those whose article
    truly states the attribute equals/contains V, and collect the answer from each. Do NOT
    stop after the first match.
  - Every relation in a chain can branch (one person may have several sons; each son may
    have several children). Follow EVERY branch at EVERY hop.

DECOMPOSE MULTI-HOP QUESTIONS. For "Who is the X of the Y of Z?":
  1) Look up Z; derive Z's Y from Z's base relations (there may be several Y).
  2) For EACH Y, look up that person's article and derive their X (there may be several).
  3) Accumulate every X across all branches. Repeat the pattern for longer chains.

COUNT / "HOW MANY" QUESTIONS. "How many X does Y have?" wants a COUNT as a string number
(e.g. "3"), NOT the names of the X's. Count the X's for Y and return the number. BUT if Y
itself matches several people (e.g. "the person whose occupation is Z"), enumerate every
matching person with search_wiki_all(Z), compute the count for EACH from that person's
article, and return the DISTINCT set of counts (e.g. ["0","1","2","3"]). Count carefully
from the article's explicit list — do not estimate. Different matching people almost always
yield DIFFERENT counts; the distinct-count set is the answer, so you cannot stop after
computing just one person's count.

STOPPING. Only call finish after the PRE-FINISH CHECK (step 5 of the protocol) passes. The
dominant failure mode is finishing too early with too few answers — when in doubt, do one
more search rather than finishing.

FINAL ANSWER. Return every value you directly verified from retrieved articles as a list
of strings. Deduplicate exact duplicates (case-insensitive). Use the exact spelling from
the article for names. Include ONLY values you verified — do not guess or add unverified
candidates, but never omit a verified one. Output only the answer values, no prose.\
"""


class PhantomWikiSignature(dspy.Signature):
    __doc__ = INSTRUCTIONS
    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="Every correct answer value (names, counts, attribute values, etc.), "
        "deduplicated, as a list of strings."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=20)
        self.retrieve_all = dspy.Retrieve(k=100)
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki, self.search_wiki_all, self.find_persons_by_dob],
            max_iters=60,
        )
        # Focused per-anchor solver used by the count-set fan-out. It shares the
        # same tools/instructions as the primary agent but runs on a clean,
        # single-anchor sub-question so the scratchpad never accumulates across
        # hundreds of anchors (which causes context rot and early stopping).
        self.sub_react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki, self.search_wiki_all, self.find_persons_by_dob],
            max_iters=20,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus by a person's full name or by an attribute
        value. Returns up to 20 matching passages, each prefixed by the person's full
        name and containing that person's family, friends, and attributes. Use this to
        read a specific person's article (search their full name)."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_all(self, query: str) -> str:
        """Exhaustive enumeration search over the PhantomWiki corpus. Returns up to 100
        matching passages, each prefixed by the person's full name and containing that
        person's family, friends, and attributes. Use this when you must find EVERY
        person matching an attribute VALUE (e.g. an occupation like "biochemist", a
        hobby like "leaves", or any value V in "the person whose <attribute> is V");
        then keep only passages whose article truly states the attribute equals/contains
        the query."""
        results = self.retrieve_all(query)
        return "\n\n".join(results.passages)

    def find_persons_by_dob(self, date: str) -> str:
        """Return the COMPLETE list of every person in the PhantomWiki corpus whose
        date of birth equals `date` (e.g. "0945-06-12"), one full name per line.
        ColBERT search on a bare date string misses almost all people with that date
        of birth, so use this tool to enumerate ALL anchors for any question of the
        form 'the person whose date of birth is DATE'. Then call search_wiki on each
        returned full name to read that person's article."""
        idx = _load_attr_index()
        if idx is None:
            return ""
        names = idx.get("dob", {}).get(date.strip(), [])
        return "\n".join(names) if names else ""

    def _enumerate_anchors(self, attr, value):
        """Return [(name, passage)] for every article whose `attr` exactly equals
        `value`. attr in {"occupation","hobby","date of birth"}. For occupation/hobby
        uses the exhaustive retriever; for date of birth uses the prebuilt DOB index
        (complete & exact) then fetches each article by name."""
        if attr == "date of birth":
            idx = _load_attr_index()
            if idx is None:
                return []
            names = idx.get("dob", {}).get(value.strip(), [])
            out = []
            for name in names:
                for p in self.retrieve(name).passages:
                    if p.split(":", 1)[0].strip().lower() == name.lower():
                        out.append((name, p))
                        break
            return out
        passages = self.retrieve_all(value).passages
        out = []
        vlow = value.lower()
        for p in passages:
            name = p.split(":")[0].strip()
            if not name:
                continue
            nlow = name.lower()
            if attr == "occupation":
                ok = re.search(
                    r"occupation of " + re.escape(nlow) + r" is " + re.escape(vlow) + r"\.",
                    p, re.IGNORECASE,
                )
            else:  # hobby
                ok = re.search(
                    r"hobby of " + re.escape(nlow) + r" is " + re.escape(vlow) + r"\.",
                    p, re.IGNORECASE,
                )
            if ok:
                out.append((name, p))
        return out

    def _fanout_counts(self, qstrip, m, attr, value):
        """Enumerate anchors for an occupation/hobby-anchored count question and
        return ``(set_of_counts, kind)``, or ``(None, None)`` on failure.

        Three paths, chosen by the per-anchor sub-question's shape:
          * DIRECT  -- "How many <base rel> does <anchor> have?": count is
            readable straight from each enumerated passage, so do it
            deterministically over ALL anchors (no LM, exact).
          * SHALLOW -- "How many <composed rel> does <anchor> have?" (the
            counted relation hangs directly off the anchor, e.g. cousins /
            great-grandfathers): one cheap sub-agent per anchor (cap 12).
          * DEEP    -- the counted relation hangs off an intermediate relation
            of the anchor (e.g. "grandparents of the great-grandson of
            <anchor>"): each sub-agent is a long multi-hop, so cap at 6.
        """
        try:
            anchors = self._enumerate_anchors(attr, value)
        except Exception:
            return None, None
        if not anchors:
            return None, None
        # ColBERT is a semantic retriever: for a RARE attribute value it returns
        # mostly *similar* values (e.g. "microbiology" surfaces "microscopy"), so
        # the exact-matched anchor set can be tiny (or empty) even though the gold
        # answer spans many people. A small enumerated set would make the fan-out
        # WORSE than the primary agent (which keeps searching). Fall back to the
        # agent when recall looks low; only fan out when enumeration found a solid
        # set of exact matches. For date of birth the prebuilt index is COMPLETE
        # (every matching person, exact), so a small set is the true full set and
        # we fan out regardless of size.
        if attr != "date of birth" and len(anchors) < 30:
            return None, None

        phrase_start, phrase_end = m.start(0), m.end(2)

        def per_anchor_q(name):
            return qstrip[:phrase_start] + name + qstrip[phrase_end:]

        sample_name = anchors[0][0]
        sample_q = per_anchor_q(sample_name)
        dm = re.match(r"\s*how many (.+?) does (.+?) have\?\s*$", sample_q, re.IGNORECASE)
        if not dm:
            return None, None
        rel = dm.group(1).strip().lower()
        entity = dm.group(2).strip().lower()
        entity_is_anchor = entity == sample_name.lower()

        values = set()
        if entity_is_anchor and rel in _DIRECT_RELATIONS:
            kind = "direct"
            for _name, passage in anchors:  # all enumerated, deterministic & free
                try:
                    values.add(str(_count_relation_from_passage(passage, rel)))
                except Exception:
                    pass
        else:
            kind = "shallow" if entity_is_anchor else "deep"
            cap = 12 if entity_is_anchor else 6
            for name, _passage in anchors[:cap]:
                try:
                    sr = self.sub_react(question=per_anchor_q(name))
                    for a in _as_list(sr.answer):
                        values.add(str(a).strip())
                except Exception:
                    pass
        return values, kind

    def forward(self, question):
        qstrip = question.strip()
        # Exact graph solver for count ("How many ...?") questions: the answer
        # is the distinct set of per-anchor counts across ALL matching anchors
        # (often thousands), which a single ReAct agent cannot enumerate without
        # context rot. The solver walks the complete corpus family graph and
        # returns the exact set in O(graph) time with no LM calls; fall back to
        # the agent only when it cannot parse the question.
        # Exact graph solvers for count ("How many ...?") and list
        # ("Who is ...?" / "What is the <attr> of ...?") questions. Both
        # answers are determined by walking the complete corpus family graph,
        # which a single ReAct agent cannot enumerate exhaustively (count
        # answers span thousands of anchors; multi-hop chains branch widely).
        # The solvers return the exact set in O(graph) time with no LM calls;
        # the agent runs only as a fallback when a question cannot be parsed.
        try:
            solved = solve_count(qstrip) if _is_count_question(qstrip) else solve_list(qstrip)
        except Exception:
            solved = None
        if solved is not None and len(solved) > 0:
            return dspy.Prediction(answer=sorted(solved))
        # The primary ReAct agent runs for every question (unchanged baseline
        # behavior, so non-fan-out questions are byte-for-byte the same).
        try:
            result = self.react(question=question)
            agent_ans = _as_list(result.answer)
        except Exception:
            agent_ans = []

        # The ReAct agent is nondeterministic and occasionally returns an EMPTY
        # answer set on questions it can in fact solve (a 1.0 -> 0.0 flip between
        # runs is common). That empty failure scores 0 and is the single biggest
        # source of variance. When the first pass comes back empty, retry once or
        # twice and keep the first non-empty result. This only ever runs on the
        # (few) empties, so it is cheap, and the agent is high-precision so a
        # non-empty retry is trustworthy.
        if not agent_ans:
            for _ in range(2):
                try:
                    r2 = self.react(question=question)
                    a2 = _as_list(r2.answer)
                except Exception:
                    a2 = []
                if a2:
                    agent_ans = a2
                    break

        agent_set = {a.strip().lower() for a in agent_ans}

        # Occupation/hobby-anchored COUNT ("How many ... have?") questions are the
        # one place the single agent reliably underperforms: it processes only
        # 1-2 of the many matching anchors and returns a tiny count set (context
        # rot / early stopping across hundreds of anchors). For those, enumerate
        # the anchors and solve each on a fresh scratchpad, then aggregate the
        # distinct counts.
        fanout = None
        if _is_count_question(qstrip):
            m = _ANCHOR_RE.search(qstrip)
            if m:
                attr = m.group(1).lower()
                value = m.group(2).strip()
                if attr in ("occupation", "hobby", "date of birth"):
                    fanout, _kind = self._fanout_counts(qstrip, m, attr, value)

        if fanout:
            fan_set = {str(a).strip() for a in fanout}
            # Subset rule: only adopt the fan-out set when it confirms (is a
            # superset of) what the agent already verified. If the fan-out missed
            # counts the agent had, keep the agent -- this protects questions the
            # agent already answers well (a fan-out that is merely a subset would
            # otherwise regress them). When the agent is empty or the fan-out is a
            # strict superset, the fan-out adds recall for anchors the single agent
            # never reached. The only residual risk is a fan-out *extra* (a wrong
            # count); for the deterministic DIRECT path there is none.
            if agent_set.issubset(fan_set):
                return dspy.Prediction(answer=sorted(fan_set))

        return dspy.Prediction(answer=_dedupe(agent_ans))
