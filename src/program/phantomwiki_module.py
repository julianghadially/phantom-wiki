import dspy

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
  2. ENUMERATE ANCHORS. Find EVERY person matching the anchor. Use search_wiki_all(<value>)
     to surface up to 100 matches at once, then run more searches with varied phrasings if
     anything may be missed. Keep a WRITTEN running list of every matching person you have
     confirmed. NEVER assume there is only one. NEVER stop after the first.
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

SEARCH TOOLS. You have two retrievers over the same corpus.
  - search_wiki(query): returns up to 20 matching passages. Use this to READ a specific
    person — search their FULL NAME (e.g. "Alan Denney") — or for any quick lookup.
  - search_wiki_all(query): returns up to 100 matching passages. Use this EXCLUSIVELY when
    you must find EVERY person matching an attribute VALUE (an occupation, a hobby, or any V
    in "the person whose <attribute> is <value>"). One search_wiki_all(V) call surfaces far
    more matches than search_wiki, so prefer it for exhaustive enumeration of anchors; then
    keep ONLY passages whose article truly states the attribute EQUALS/CONTAINS V (watch for
    substring false positives like "leaves", "surveyor").
  - Dates of birth are hard to retrieve by searching the date string alone; if a question
    anchors on a date of birth, you may still need other handles, and you must verify any
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
            tools=[self.search_wiki, self.search_wiki_all],
            max_iters=60,
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

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
