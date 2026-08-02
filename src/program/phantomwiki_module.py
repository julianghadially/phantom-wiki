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

RELATION DERIVATION. Composed relations are reached by chaining base relations across
several article lookups. Standard decompositions:
  - parent(P,C): mother or father of C.      sibling: brother or sister.
  - uncle/aunt of X = a brother/sister of X's parent.     cousin of X = a child of X's
    aunt or uncle.     grandparent of X = a parent of X's parent.     great-grandparent =
    parent of a grandparent (and so on for "great-").     nephew/niece of X = a child of
    X's sibling.     grandchild of X = a child of X's child.
  - in-law relations go through a spouse: father-in-law of X = father of X's spouse;
    sibling-in-law = sibling of X's spouse; child-in-law = child of X's spouse (etc.).
  - "once removed" = one generation apart (e.g. first cousin once removed = child of a
    cousin, or cousin of a parent).   "second cousin" = a cousin of a parent (shared
    great-grandparent).   "second uncle/aunt" = an uncle/aunt of a parent (a sibling of a
    grandparent).
  - gendered variants ("female cousin", "male first cousin once removed") add a gender
    filter on the final result using that person's stated gender.
At every hop, look up the person's article, read their base relations, and branch out
wherever a relation yields several people. Keep a running list of intermediate entities so
no branch is lost; re-search a name rather than relying on memory.

SEARCH TOOL. search_wiki(query) returns up to 20 matching passages, each prefixed by the
person's name and containing that person's base relations, friends, and attributes.
  - To read about a known person, search their FULL NAME (e.g. "Alan Denney").
  - To find every person matching an attribute value, search the VALUE directly (e.g.
    "biochemist" finds occupations like "clinical biochemist" — match on the substring;
    a hobby like "leaves"; an occupation like "surveyor"). Then keep ONLY passages whose
    article truly states the attribute equals/contains V.
  - Dates of birth are hard to retrieve by searching the date string alone; if a question
    anchors on a date of birth, you may still need other handles, and you must verify any
    candidate's stated date of birth matches exactly before using it.
  - If one search may not have surfaced all matches, run additional searches with varied
    queries before concluding.

MOST QUESTIONS HAVE MULTIPLE CORRECT ANSWERS — partial recall scores zero for the missing
ones, so exhaustive enumeration is the single most important thing you do.
  - "the person whose [attribute] is V" usually matches MANY people. You MUST find ALL of
    them. Search for V, read every returned passage, keep ONLY those whose article truly
    states the attribute equals/contains V, and collect the answer from each. Do NOT stop
    after the first match.
  - Every relation in a chain can branch (one person may have several sons; each son may
    have several children). Follow EVERY branch at EVERY hop.

DECOMPOSE MULTI-HOP QUESTIONS. For "Who is the X of the Y of Z?":
  1) Look up Z; derive Z's Y from Z's base relations (there may be several Y).
  2) For EACH Y, look up that person's article and derive their X (there may be several).
  3) Accumulate every X across all branches. Repeat the pattern for longer chains.

COUNT / "HOW MANY" QUESTIONS. "How many X does Y have?" wants a COUNT as a string number
(e.g. "3"), NOT the names of the X's. Count the X's for Y and return the number. BUT if Y
itself matches several people (e.g. "the person whose occupation is Z"), enumerate every
matching person, compute the count for EACH, and return the DISTINCT set of counts (e.g.
["0","1","2","3"]). Count carefully from the article's explicit list — do not estimate.

STOPPING. Only call finish once you have pursued every branch, derived every relation hop,
and searched thoroughly for all matching people. Returning too few answers is the main
failure mode.

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
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus by a person's name or by an attribute value
        (e.g. a date of birth, an occupation, a hobby). Returns up to 20 matching
        passages, each prefixed by the person's full name and containing that person's
        family, friends, and attributes."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
