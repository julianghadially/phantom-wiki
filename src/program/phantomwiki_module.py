import dspy


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

    HOW TO SEARCH AND TRAVERSE:
    - When you retrieve a person's article, READ it and extract the names/relations that
      are written directly inside it. Do NOT issue a separate search like "X husband" or
      "X children" to find a relation that is already listed in X's own article -- the
      spouse, children, parents, and siblings of X are right there in X's article.
    - If the question names a person, first search that person's full name to retrieve
      their article, then read the relevant listed relation to advance one hop.
    - If the anchor is an attribute ("the person whose occupation is structural engineer"
      or "the person whose date of birth is 0917-08-17"), search the attribute VALUE
      itself (e.g. "structural engineer" or "0917-08-17") to retrieve articles of people
      who have that attribute.
    - Use base relations directly: the CHILDREN of X are the sons/daughters listed in X's
      article; the PARENTS of X are the mother/father in X's article; the SIBLINGS of X
      are the sisters/brothers in X's article; the SPOUSE of X is the husband/wife in X's
      article; the FRIENDS of X are listed in X's article.
    - For a derived relation, decompose it into base hops and traverse one hop at a time:
      search a name, read its article, collect the next-hop entities, then search each of
      those names, and so on, branching at every hop.
    - Issue multiple, varied queries to surface documents a single query might miss, but
      never repeat the same query expecting different results.

    ENUMERATION DISCIPLINE (CRITICAL):
    - Most questions have MULTIPLE correct answers. Recall is scored just as heavily as
      precision, so you must find ALL of them, not just one or a few.
    - At every hop, enumerate EVERY entity that satisfies that hop's relation, then branch
      and follow EACH one through the remaining hops. Do not pick a single branch.
    - Keep a running, deduplicated set of candidate final answers as you go.
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
    answer: list[str] = dspy.OutputField()


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature=PhantomWikiQASignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
