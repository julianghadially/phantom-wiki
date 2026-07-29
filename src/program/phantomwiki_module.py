import os
import pickle
import re

import dspy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORPUS_INDEX_PATH = os.path.join(
    _REPO_ROOT, "output", "depth_10_size_1000000", "_corpus_index.pkl"
)

# Process-wide cache of the local corpus index (loaded once, shared across
# threads / pipeline instantiations). Loading the ~500 MB pickle is a one-time
# cost paid on first instantiation.
_INDEX_CACHE = None


def _build_index_from_articles(articles_path, out_path):
    """Build {name2facts, dob, hobby, job} from articles.json using orjson."""
    import json

    try:
        import orjson

        with open(articles_path, "rb") as f:
            arts = orjson.loads(f.read())
    except ImportError:  # fall back to stdlib json (slower)
        with open(articles_path) as f:
            arts = json.load(f)

    def s(x):
        return x.decode() if isinstance(x, bytes) else x

    name2facts, dob, hobby, job = {}, {}, {}, {}
    pat = re.compile(r'(\w+)\("(.+?)", "(.+?)"\)\.')
    for a in arts:
        nm = s(a["title"])
        fs = [s(x) for x in a["facts"]]
        name2facts[nm] = fs
        for fact in fs:
            m = pat.match(fact)
            if not m:
                continue
            rel, obj = m.group(1), m.group(3)
            if rel == "dob":
                dob.setdefault(obj, []).append(nm)
            elif rel == "hobby":
                hobby.setdefault(obj, []).append(nm)
            elif rel == "job":
                job.setdefault(obj, []).append(nm)
    with open(out_path, "wb") as f:
        pickle.dump(
            {"name2facts": name2facts, "dob": dob, "hobby": hobby, "job": job},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {"name2facts": name2facts, "dob": dob, "hobby": hobby, "job": job}


def _load_corpus_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if os.path.exists(_CORPUS_INDEX_PATH):
        with open(_CORPUS_INDEX_PATH, "rb") as f:
            _INDEX_CACHE = pickle.load(f)
        return _INDEX_CACHE
    # Fallback: build from articles.json if the pickle is absent.
    articles_path = os.path.join(
        _REPO_ROOT, "output", "depth_10_size_1000000", "articles.json"
    )
    if not os.path.exists(articles_path):
        raise FileNotFoundError(
            f"No corpus index at {_CORPUS_INDEX_PATH} and no articles.json at {articles_path}."
        )
    _INDEX_CACHE = _build_index_from_articles(articles_path, _CORPUS_INDEX_PATH)
    return _INDEX_CACHE


def _format_facts(name, facts):
    """Render a person's compact Prolog-style facts into the EXACT article format
    the PhantomWiki corpus uses (so the rest of the prompt's language applies
    unchanged). Only lines that exist in the facts are emitted."""
    buckets = {
        "mother": [], "father": [], "son": [], "daughter": [],
        "brother": [], "sister": [], "husband": [], "wife": [],
        "friend": [], "dob": [], "job": [], "hobby": [], "gender": [],
    }
    for f in facts:
        m = re.match(r'(\w+)\("(.+?)", "(.+?)"\)\.', f)
        if not m:
            continue
        rel, _subj, obj = m.group(1), m.group(2), m.group(3)
        if rel in buckets:
            buckets[rel].append(obj)

    def line(label_single, label_plural, items):
        if not items:
            return None
        if len(items) == 1:
            return f"{label_single} {name} is {items[0]}."
        return f"{label_plural} {name} are {', '.join(items)}."

    parts = [f"# {name}"]
    fam = []
    for rel, lsing, lplur in [
        ("mother", "The mother of", "The mothers of"),
        ("father", "The father of", "The fathers of"),
        ("son", "The son of", "The sons of"),
        ("daughter", "The daughter of", "The daughters of"),
        ("brother", "The brother of", "The brothers of"),
        ("sister", "The sister of", "The sisters of"),
        ("husband", "The husband of", "The husbands of"),
        ("wife", "The wife of", "The wives of"),
    ]:
        ln = line(lsing, lplur, buckets[rel])
        if ln:
            fam.append(ln)
    if fam:
        parts.append("## Family")
        parts.extend(fam)
    if buckets["friend"]:
        fr = (f"The friend of {name} is {buckets['friend'][0]}."
              if len(buckets["friend"]) == 1
              else f"The friends of {name} are {', '.join(buckets['friend'])}.")
        parts.append("## Friends")
        parts.append(fr)
    attr = []
    if buckets["dob"]:
        attr.append(f"The date of birth of {name} is {buckets['dob'][0]}.")
    if buckets["job"]:
        attr.append(f"The occupation of {name} is {buckets['job'][0]}.")
    if buckets["hobby"]:
        attr.append(f"The hobby of {name} is {buckets['hobby'][0]}.")
    if buckets["gender"]:
        attr.append(f"The gender of {name} is {buckets['gender'][0]}.")
    if attr:
        parts.append("## Attributes")
        parts.extend(attr)
    return "\n".join(parts)


INSTRUCTIONS = """You answer multi-hop reasoning questions over the PhantomWiki corpus, a fictional Wikipedia where every article is about ONE person.

CORPUS ACCESS TOOLS (these are EXACT over the corpus index — prefer them always):
- get_person_facts(full_name): returns the FULL article (Family/Friends/Attributes) for an EXACT full name, and NOTHING else (no other people). This is the primary read tool — use it for every direct read of a person whose name you already have (parents, children, siblings, spouse, friends, own attributes).
- find_people_with_date_of_birth(date): returns EVERY person whose Date of Birth EXACTLY equals the date (YYYY-MM-DD). COMPLETE and EXACT. USE THIS for the selector "the person whose date of birth is <DATE>" — ColBERT semantic search does NOT reliably surface date matches.
- find_people_with_hobby(value): returns EVERY person whose hobby EXACTLY equals the value (e.g. "tea bag collecting", "crystals"). COMPLETE and EXACT. Use for "the person whose hobby is <VALUE>".
- find_people_with_occupation(value): same for occupation (e.g. "farm manager").
- search_wiki / search_wiki_many: ColBERT semantic search (fallback only). The exact tools above ALWAYS beat ColBERT for attribute enumeration and direct reads.

CORPUS STRUCTURE & RELATION SEMANTICS:
- Each article's Family section lists, DIRECTLY, that person's OWN: mother, father, sons, daughters, brothers, sisters, husband/wife (spouse). Friends section lists friends. Attributes section lists date of birth, occupation, hobby, gender ('male'/'female' only).
- Relations NEVER listed directly and therefore always DERIVED by chaining: grandparents, grandchildren, aunts/uncles, nieces/nephews, cousins, great-/second-/once-removed kin, and all in-laws.

DIRECTION CONVENTION: relation(X, Y) means "Y is the <relation> of X". A question "Who is the R of A?" asks for every Y with R(A, Y). Enumerate ALL such Y.

RELATION DEFINITIONS ('parent' = mother or father; 'child' = son or daughter):
- parent(X,Y): mother or father of X.  child(X,Y): son or daughter of X.  sibling(X,Y): shares a parent with X (full OR half).
- grandparent(X,Y): parent of parent of X.  great-grandparent(X,Y): parent of grandparent of X (UP 3 parents).
- grandchild / great-grandchild of X: go DOWN via child repeatedly (2 / 3 steps). Children are listed directly in the ancestor's Family section.
- aunt(X,Y): sister of a parent of X.  uncle(X,Y): brother of a parent of X.  (collect over BOTH parents' siblings.)
- great-aunt(X,Y): sister of a grandparent of X.  great-uncle(X,Y): brother of a grandparent of X.
- second-aunt(X,Y): sister of a GREAT-grandparent of X.  second-uncle(X,Y): brother of a GREAT-grandparent of X.  (NOT grandparent — one generation higher than great-aunt/uncle.)
- niece(X,Y): daughter of a sibling of X.  nephew(X,Y): son of a sibling of X.
- cousin(X,Y): Y is a child of a sibling of X's parent.  female/male cousin: cousin + gender == female/male.
- second cousin(X,Y): Y's parent and X's parent are cousins (child of a parent's cousin).
- first cousin once removed(X,Y): Y is a CHILD (daughter or son) of X's cousin. (Only this direction.)

EXHAUSTIVE UPWARD TRAVERSAL (critical — second-uncle/great-aunt/great-grandparent questions lose almost all score when a branch is dropped):
- When tracing grandparents / great-grandparents / second-uncles, you MUST recurse through BOTH the mother AND the father at EVERY level and follow EVERY branch to its end. A great-grandparent = a parent of a parent of a parent of X; there are up to 2^3 = 8 great-grandparent slots (X's 2 parents, their 4 parents, their 8 parents). Do NOT stop after one branch dead-ends — the answer may exist ONLY via a different branch. If the question asks for siblings of a great-grandparent (second-uncle/second-aunt), open EVERY great-grandparent you can reach on BOTH maternal and paternal lines and read each one's siblings.
- Procedure for going UP N generations: start from subject X, get_person_facts(X) -> mother M, father F. For each of {M,F} read their parents (X's grandparents); for each of those read their parents (X's great-grandparents); one generation per get_person_facts call. At the top level read each ancestor's SIBLINGS to get aunts/uncles/great-/second- aunts/uncles. Keep a WRITTEN list of every ancestor name still UNEXPANDED; process them ALL before concluding the answer is empty.

IN-LAWS ARE THROUGH MARRIAGE (do NOT substitute siblings):
- spouse(X) = the person listed as husband/wife of X in X's article.
- mother-in-law(X,Y): mother of X's spouse.  father-in-law(X,Y): father of X's spouse.  brother-in-law(X,Y): brother of X's spouse.  sister-in-law(X,Y): sister of X's spouse.  son-in-law(X,Y): husband of X's child.  daughter-in-law(X,Y): wife of X's child.
- To resolve an in-law RELATION at a chain endpoint P: first resolve the chain up to endpoint P, then for EACH P call get_person_facts(P) to get P's spouse S, then call get_person_facts(S) and take S's parent/sibling as the in-law answer. Never take a sibling of P itself as an in-law.

GENDERED / QUALIFIED RELATIONS: resolve the full relation first, then keep only members whose gender attribute matches the qualifier (male / female). Call get_person_facts on each candidate and check its gender line.

AMBIGUITY — almost every question's described subject matches MULTIPLE people:
- Attribute selectors ("whose date of birth is D", "whose hobby is H", "whose occupation is O") are AMBIGUOUS: call the matching find_people_with_* tool to get the COMPLETE list of matching anchors. The COMPLETE correct answer is the UNION over EVERY matching anchor. Do NOT stop after the first one.
- The returned list from find_people_with_* is the COMPLETE set. For date selectors usually only a handful of people share a date — CHAIN ALL of them. For hobby/occupation selectors the list can be hundreds-to-thousands; you cannot chain ALL of them within your iteration budget, so SAMPLE BROADLY across the list (pick names spread across the FULL list, not just the first few), chain each to its derived answer/count, and report the UNION of distinct results. Keep sampling until repeated broad sampling returns no new distinct answer.
- 'How many X does [subject] have?': the subject may be several people, each with its own count. Enumerate every matching anchor, compute that count for each, and report EVERY distinct count as a SEPARATE answer string (e.g. ["1","3","12"], NOT a single number). Count a relation from that anchor's DIRECT Family lines (e.g. number of daughters = count the daughter lines in that anchor's article).
- 'Who ...' / 'list ...': union all matching entities, deduplicate by full name.
- DEAD-END FALL-THROUGH (critical): if ONE matching anchor's chain dead-ends (no sibling/child/nephew/etc.), that does NOT make the whole answer empty — almost always OTHER anchors DO yield answers. Keep enumerating the remaining anchors and aggregate whatever each produces. Only after exhausting ALL anchors (and, for upward questions, ALL branches) may you consider the answer empty, and even then include any partial candidates found earlier.

OUTPUT TYPE RULES (critical to avoid a zero score):
- 'How many ...' -> answer entries are COUNTS as digit strings (e.g. "0", "2"). NEVER answer a how-many question with a name. Output the SET of distinct counts observed across anchors.
- 'Who ...' / 'list ...' -> answer entries are FULL people's names exactly as printed (the article title '# <Full Name>').
- 'What is the occupation/hobby/date of birth/gender of ...' -> answer entries are the attribute value(s) (occupation phrase / hobby phrase / date YYYY-MM-DD / male|female). Union across all matching anchors, deduplicated.

STRATEGY:
1. Break the question into its relation chain. Identify the anchor (a named person, if present) and the sequence of relations. Identify any attribute selector.
2. If there is an attribute selector, FIRST call the matching find_people_with_* tool to obtain the COMPLETE list of matching anchors.
3. For each anchor (or the single named subject), drive the relation chain using get_person_facts: one call per person whose family/attributes you need. For UPWARD steps read mother's and father's articles and recurse through BOTH branches. For DOWNWARD / niece / nephew / cousin steps read children/siblings and chain.
4. In your thoughts keep a running tally of: (a) which anchors are CONFIRMED, (b) which derived answer(s)/count each yields, (c) which anchors or branches remain UNEXPANDED. Keep searching until no unexpanded anchor/branch remains.

FINAL ANSWER:
- Deduplicate case-insensitively. Output a flat list of exact full names / exact attribute wording; never invent names.
- NEVER return an empty list unless you have exhausted every anchor/branch and genuinely found nothing. Partial credit beats a guaranteed zero: returning the answers/counts you DID confirm ALWAYS beats returning [].
- For how-many questions, list every distinct count you observed across anchors."""


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        idx = _load_corpus_index()
        self._name2facts = idx["name2facts"]
        self._dob = idx["dob"]
        self._hobby = idx["hobby"]
        self._job = idx["job"]
        self.retrieve = dspy.Retrieve(k=30)
        self.retrieve_many = dspy.Retrieve(k=100)
        self.react = dspy.ReAct(
            signature=dspy.Signature("question -> answer: list[str]", INSTRUCTIONS),
            tools=[
                self.get_person_facts,
                self.find_people_with_date_of_birth,
                self.find_people_with_hobby,
                self.find_people_with_occupation,
                self.search_wiki,
                self.search_wiki_many,
            ],
            max_iters=60,
        )

    # ---- exact local-corpus tools (primary) ----
    def get_person_facts(self, full_name: str) -> str:
        """Return the EXACT, complete PhantomWiki article (Family/Friends/Attributes) for a person given their exact full name (e.g. 'Loraine Moritz'). This is the PRIMARY read tool: use it to read a named person's parents, children, siblings, spouse, friends, or own attributes. Returns 'Person not found: <name>' if the name is not in the corpus. Only that one person is returned — no other people, no noise."""
        facts = self._name2facts.get(full_name)
        if not facts:
            return "Person not found: " + full_name
        return _format_facts(full_name, facts)

    def find_people_with_date_of_birth(self, date: str) -> str:
        """Return EVERY person whose Date of Birth EXACTLY equals the given date string (format YYYY-MM-DD, e.g. '0918-01-17'). COMPLETE and EXACT over the whole corpus. Use it for the selector 'the person whose date of birth is <date>'. Returns 'Found N people ...: name1; name2; ...'."""
        names = self._dob.get(date, [])
        if not names:
            return f"Found 0 people with date of birth {date}."
        return "Found {} people with date of birth {}: {}".format(
            len(names), date, "; ".join(names)
        )

    def find_people_with_hobby(self, value: str) -> str:
        """Return EVERY person whose hobby EXACTLY equals the given phrase (e.g. 'tea bag collecting', 'crystals', 'stone collecting'). COMPLETE and EXACT. Use for 'the person whose hobby is <value>'. For common hobbies the list is large (the tool returns the full list + total count) — SAMPLE BROADLY across it."""
        names = self._hobby.get(value, [])
        if not names:
            return f"Found 0 people with hobby '{value}'."
        return "Found {} people with hobby '{}': {}".format(
            len(names), value, "; ".join(names)
        )

    def find_people_with_occupation(self, value: str) -> str:
        """Return EVERY person whose occupation EXACTLY equals the given phrase (e.g. 'farm manager'). COMPLETE and EXACT. Use for 'the person whose occupation is <value>'. The list may be large — sample broadly across it."""
        names = self._job.get(value, [])
        if not names:
            return f"Found 0 people with occupation '{value}'."
        return "Found {} people with occupation '{}': {}".format(
            len(names), value, "; ".join(names)
        )

    # ---- ColBERT semantic search (fallback) ----
    def search_wiki(self, query: str) -> str:
        """ColBERT semantic search over the PhantomWiki corpus (~30 passages). Fallback only — prefer get_person_facts for reads and the find_people_with_* tools for attribute enumeration."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_many(self, query: str) -> str:
        """High-recall ColBERT search (~100 passages). Fallback only."""
        results = self.retrieve_many(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        last_err = None
        for _attempt in range(3):
            try:
                result = self.react(question=question)
                return dspy.Prediction(answer=result.answer)
            except Exception as exc:  # AdapterParseError / decode hiccups — retry
                last_err = exc
        raise last_err