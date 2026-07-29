import dspy

INSTRUCTIONS = """You answer multi-hop reasoning questions over the PhantomWiki corpus, a fictional Wikipedia where every article is about ONE person.

CORPUS STRUCTURE (critical — read directly, do NOT over-search):
- Each retrieved passage is a short article beginning with '# <Full Name>'. Its sections are:
  ## Family  -- the person's OWN relatives, stated DIRECTLY:
      The mother of X is <Y>.   The father of X is <Y>.
      The son/sons of X are <...>.   The daughter(s) of X are <...>.        <-- CHILDREN are listed directly
      The brother(s) of X are <...>.   The sister(s) of X are <...>.         <-- SIBLINGS are listed directly
      The husband of X is <Y>. / The wife of X is <Y>.                        <-- SPOUSE is listed directly
  ## Friends -- The friends of X are <...>.
  ## Attributes -- date of birth, occupation, hobby, gender of X.
- IMPORTANT CORRECTION: children, siblings AND spouses ARE listed directly in a person's own Family section. Do NOT reverse-engineer them unnecessarily. Read the relevant Family line straight from X's article. The reverse-lookup trick below is a FALLBACK / cross-check only, for the cases where a needed Family line is absent from a particular article.
- Relations NEVER listed directly and therefore always DERIVED by chaining: grandparents, grandchildren, aunts/uncles, nieces/nephews, cousins, great-/second-/once-removed kin, and all in-laws.
- Every article lists mother AND father (when the person has parents). Repeatedly reading parent articles takes you UP the tree one generation at a time.
- Article gender attribute is 'male' or 'female'. There is no 'nonbinary' in the answer set.

DIRECTION CONVENTION: relation(X, Y) means "Y is the <relation> of X". A question "Who is the R of A?" asks for the Y such that R(A, Y). Enumerate ALL such Y.

RELATION DEFINITIONS (exact for this corpus; 'parent' = mother or father; 'child' = son or daughter):
- parent(X,Y): Y is mother or father of X.   child(X,Y): Y is son or daughter of X.   sibling(X,Y): Y shares a parent with X (full OR half sibling).
- grandparent(X,Y): Y is parent of parent of X.   great-grandparent(X,Y): parent of grandparent = go UP 3 parents.
- grandchild / great-grandchild of X: go DOWN via child repeatedly (2 / 3 steps). Note children are listed directly in the relevant ancestor's Family section.
- aunt(X,Y): sister of a parent of X.   uncle(X,Y): brother of a parent of X.   (collect over BOTH mother and father's siblings.)
- great-aunt(X,Y): sister of a grandparent of X.   great-uncle(X,Y): brother of a grandparent of X.
- second-aunt(X,Y): sister of a GREAT-grandparent of X.   second-uncle(X,Y): brother of a GREAT-grandparent of X.   (NOT grandparent — go one generation higher than great-aunt/uncle.)
- niece(X,Y): daughter of a sibling of X.   nephew(X,Y): son of a sibling of X.
- cousin(X,Y): Y is a child of a sibling of X's parent (i.e. child of parent's sibling).   female/male cousin: cousin + gender == female/male.
- second cousin(X,Y): Y's parent and X's parent are cousins (equivalently, child of a parent's cousin).
- first cousin once removed(X,Y): Y is a CHILD (daughter or son) of X's cousin. (Only this direction exists; do NOT also go to a cousin of a parent.)
- reverse-lookup FALLBACK (use when a Family line is missing): searching a person's full name returns their OWN article PLUS the articles of everyone who mentions that name.
  * To find the CHILDREN of P when not listed: search P's name; any result article stating 'The mother/father of <X> is P' makes X a child of P.
  * To find the SIBLINGS of P when not listed: search P's mother's name and P's father's name; collect children (excluding P itself).

IN-LAWS ARE THROUGH MARRIAGE (critical — do NOT substitute siblings):
- The spouse of X is the person listed as 'The husband/wife of X is <S>' in X's Family section. (married(X,S).)
- mother-in-law(X,Y): mother of X's spouse.   father-in-law(X,Y): father of X's spouse.
- brother-in-law(X,Y): brother of X's spouse.   sister-in-law(X,Y): sister of X's spouse.
- son-in-law(X,Y): the HUSBAND of X's child.   daughter-in-law(X,Y): the WIFE of X's child.
So to resolve an in-law RELATION at some chain endpoint P (e.g. 'mother-in-law of the friend of P'): first resolve the chain up to that endpoint P (here: friends of P), then for EACH endpoint person take their SPOUSE, then take the spouse's parent/sibling as the in-law answer. Never take a sibling of the endpoint itself as an in-law.

GENDERED / QUALIFIED RELATIONS: resolve the full relation first, then keep only members whose gender attribute matches the qualifier (male / female).

AMBIGUITY — MOST QUESTIONS ARE AMBIGUOUS (this drives almost all of the score):
- The described subject ALMOST ALWAYS matches MULTIPLE different people. Examples: 'the person whose hobby is microbiology', 'the nephew of the grandson of the person whose date of birth is 0918-01-17'. Each may identify SEVERAL distinct individuals. Assume the subject is plural until proven otherwise; a single match is the exception, not the rule.
- Attribute / date selectors are loose: searching a bare value (a date like '0918-01-17', a hobby, an occupation) via the retriever returns passages ranked by RELEVANCE, not by exact match — MANY returned passages will NOT actually contain that exact value. You MUST open and read each candidate passage and CONFIRM the attribute equals the question value before counting it as a matching anchor. Conversely, the TRUE matches may be scattered across multiple pages of results, so do NOT stop at the first relevant-looking passage; re-issue the same value query as needed to scan more candidates, and keep a written list of every CONFIRMED matching individual.
- The COMPLETE correct answer is the UNION over EVERY matching individual. Find ALL of them, not just the first.
- 'How many X does [subject] have?': the subject may be several people, each with its own count. Enumerate every matching individual, compute that count for each, and report EVERY distinct count as a SEPARATE answer string (e.g. ["1","3","12"], not a single number). When counting a relation, use the DIRECT Family lines where possible (e.g. number of daughters = count the daughter line(s)).
- 'Who ...' / list questions: union all matching entities and deduplicate by full name.
- DEAD-END FALL-THROUGH (critical): if ONE matching individual's chain dead-ends (e.g. has no sibling, child, or nephew), that does NOT make the whole answer empty — there are almost certainly OTHER matching individuals whose chains DO yield answers. Keep enumerating the remaining confirmed anchors and aggregate whatever each produces. Only after exhausting ALL confirmed matching individuals may you consider the answer empty, and even then include any partial candidates you found earlier.

ATTRIBUTE-SELECTOR ENUMERATION PROTOCOL (the retriever caps at ~64 results, and semantic ranking SCATTERS exact matches — a single bare query usually surfaces only 0-2 of the true matches, so you would wrongly conclude "no match" and score 0):
- For ANY selector 'the person whose date of birth is <DATE>': call search_wiki_many (k=64, the high-recall variant) with SEVERAL DIFFERENT phrasings of the SAME value and UNION the CONFIRMED matches:
    1. the bare date: "<DATE>"
    2. "The date of birth of <DATE>"
    3. "born on <DATE>"
    4. "<DATE> date of birth"
  Different phrasings return DIFFERENT top-64 sets, so each surfaces different exact-match people. Combine ALL of them. For EACH returned passage, ONLY count it as a matching anchor if its Attributes line literally reads "The date of birth of <Name> is <DATE>" EXACTLY matching the question date. Repeating the SAME phrasing returns the SAME results — never re-issue an identical query; instead use a new phrasing.
- For 'the person whose hobby is <VALUE>': run multi-phrasing union against the high-recall retrieve — "<VALUE>", "hobby of <VALUE>", "The hobby of X is <VALUE>", "<VALUE> hobby" — confirming each candidate's hobby line matches exactly. Hobbies are multi-word phrases (e.g. "stone collecting", "video gaming"), so match the FULL phrase, not just a keyword.
- For 'the person whose occupation is <VALUE>': same — "<VALUE>", "occupation of <VALUE>", "The occupation of X is <VALUE>", "<VALUE> occupation".
- ALWAYS write a running list of the CONFIRMED matching individuals (full names) as you go, and tick off which phrasings you have already tried so you never waste an iteration re-running one.
- CRITICAL — NEVER return [] when you have confirmed AT LEAST ONE matching individual. The retriever caps at ~64 results so you will often be unable to find EVERY matching person; that is EXPECTED. Output the answer derived from EVERY confirmed individual even if you believe more exist — partial recall is heavily rewarded (e.g. finding 4 of 7 correct occupations scores ~0.57, while returning [] scores 0.00). Try a handful of phrasings (3-4 is usually enough); you do NOT need to exhaust every possible phrasing — once you have confirmed several matching individuals and their derived answers/counts, AGGREGATE and FINISH rather than spend more iterations. Returning a partial list (some confirmed counts or names) ALWAYS beats burning the rest of the trace and returning []. If you have resolved only ONE matching individual's count for a how-many question, output that single count rather than returning nothing.

OUTPUT TYPE RULES (critical to avoid a zero score):
- 'How many ...' questions -> answer entries are COUNTS as digit strings (e.g. "0", "2"). Never answer a how-many question with a person's name. For each distinct matching individual report its own count.
- 'Who ...' / 'list ...' questions -> answer entries are FULL people's names exactly as printed in the article title ('# <Full Name>').
- 'What is the occupation/hobby/date of birth/gender of ...' -> answer entries are the attribute value(s) (occupation phrase / hobby phrase / date YYYY-MM-DD / male|female). Union across all matching individuals, deduplicated.

STRATEGY:
1. Break the question into its relation chain. Identify the anchor (a named person, if present) and the sequence of relations. Identify any attribute-based selector ('whose hobby is X', 'whose date of birth is Y'); that selector is AMBIGUOUS, so search it to find ALL matching people and confirm each.
2. Prefer DIRECT reads: to get a person's parents, children, siblings, spouse, friends, or own attributes, open that person's article and read the relevant Family/Friends/Attributes line. Only fall back to the reverse-lookup trick when the needed line is genuinely absent.
3. For UPWARD steps (parent, grandparent, great-grandparent), read the mother's and father's articles, one generation per search.
4. For in-laws at a chain endpoint P: read P's spouse from P's article, then read the spouse's parents/siblings from the spouse's article.
5. In your thoughts, keep a running tally of: (a) which matching individuals you have CONFIRMED, (b) which derived answer(s) each yields, and (c) which selector candidates remain UNVERIFIED. Keep searching until no unverified candidate remains.

FINAL ANSWER:
- Deduplicate case-insensitively. Output a flat list of strings using exact full names / exact attribute wording from retrieved articles; do not invent names.
- NEVER return an empty list unless you are certain no answer exists. If you found ANY candidate at all (even partial), include it — partial credit beats a guaranteed zero.
- For how-many questions, list every distinct count you observed across matching individuals."""


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=30)
        self.retrieve_many = dspy.Retrieve(k=64)
        self.react = dspy.ReAct(
            signature=dspy.Signature("question -> answer: list[str]", INSTRUCTIONS),
            tools=[self.search_wiki, self.search_wiki_many],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus by a person's full name, a relation phrase, an attribute value (e.g. 'microbiology'), or a date string (e.g. '0918-01-17'). Returns up to ~30 matching passages (each begins with '# <Full Name>'). Use this for DIRECT reads of a named person's article (parents/children/siblings/spouse/attributes) and for friends/relations of a single named person."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_many(self, query: str) -> str:
        """High-recall variant (~64 passages) of search_wiki. Use it to ENUMERATE the multiple different people behind an AMBIGUOUS attribute selector (a date of birth like '0946-07-14', a hobby value, or an occupation value), and for reverse-lookup sweeps (e.g. scanning who names a given person). Because the ColBERT server caps results and semantic ranking scatters exact matches, you must call this with SEVERAL DIFFERENT phrasings of the SAME value and UNION the confirmed matches (see ATTRIBUTE-SELECTOR ENUMERATION PROTOCOL)."""
        results = self.retrieve_many(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)