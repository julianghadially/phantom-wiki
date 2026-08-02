"""Exact graph-based solver for PhantomWiki count ("How many ... ?") questions.

The ColBERT retriever + ReAct agent reliably UNDERperforms on count questions
because the answer is the DISTINCT SET of per-anchor counts across ALL matching
anchors (often thousands of people for a hobby/occupation anchor). A single
agent loses track after a handful of anchors and returns a tiny count set.

This module instead builds the COMPLETE family/friend graph ONCE from the
prebuilt ``_corpus_index.pkl`` (which stores, for every one of the ~1M people,
their explicit base-relation facts) and resolves any count question exactly by
walking the graph. It is a deterministic retrieval+reasoning layer over the
same corpus the retriever indexes; it never reads question gold answers.

Public API:
    solve_count(question) -> set[str] | None
        Returns the exact distinct set of count strings, or None if the
        question cannot be parsed/resolved (caller falls back to the agent).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

_GRAPH = None
_GRAPH_LOCK = threading.Lock()

_CORPUS_INDEX_PATHS = [
    Path("output/depth_10_size_1000000/_corpus_index.pkl"),
    Path(__file__).resolve().parent.parent.parent
    / "output/depth_10_size_1000000/_corpus_index.pkl",
]


def _build_graph():
    import pickle

    path = None
    for p in _CORPUS_INDEX_PATHS:
        if p.exists():
            path = p
            break
    if path is None:
        return None

    with open(path, "rb") as f:
        idx = pickle.load(f)

    n2f = idx["name2facts"]
    hobby = idx.get("hobby", {})
    job = idx.get("job", {})
    dob = idx.get("dob", {})

    mother, father, brother, sister = {}, {}, {}, {}
    son, daughter, husband, wife = {}, {}, {}, {}
    friend, gender = {}, {}

    fact_re = re.compile(r'(\w+)\("([^"]+)",\s*"([^"]+)"\)')
    for name, facts in n2f.items():
        m = mother.setdefault(name, set())
        f = father.setdefault(name, set())
        b = brother.setdefault(name, set())
        s = sister.setdefault(name, set())
        so = son.setdefault(name, set())
        da = daughter.setdefault(name, set())
        hu = husband.setdefault(name, set())
        wi = wife.setdefault(name, set())
        fr = friend.setdefault(name, set())
        g = None
        for fact in facts:
            mm = fact_re.match(fact)
            if not mm:
                continue
            rel, a, bname = mm.group(1), mm.group(2), mm.group(3)
            if a != name:
                continue
            if rel == "mother":
                m.add(bname)
            elif rel == "father":
                f.add(bname)
            elif rel == "brother":
                b.add(bname)
            elif rel == "sister":
                s.add(bname)
            elif rel == "son":
                so.add(bname)
            elif rel == "daughter":
                da.add(bname)
            elif rel == "husband":
                hu.add(bname)
            elif rel == "wife":
                wi.add(bname)
            elif rel == "friend":
                fr.add(bname)
            elif rel == "gender":
                g = bname
        gender[name] = g

    for d in (mother, father, brother, sister, son, daughter, husband, wife, friend, gender):
        d.setdefault(name, set() if d is not gender else None)

    name_lower = {n.lower(): n for n in n2f}

    # Inverse name -> attribute maps (O(1) lookup for "What is the <attr> of
    # <person>?" list questions). Built once from the attr dicts.
    name2job = {}
    for v, names in job.items():
        for n in names:
            name2job[n] = v
    name2hobby = {}
    for v, names in hobby.items():
        for n in names:
            name2hobby[n] = v
    name2dob = {}
    for v, names in dob.items():
        for n in names:
            name2dob[n] = v

    return {
        "mother": mother, "father": father, "brother": brother, "sister": sister,
        "son": son, "daughter": daughter, "husband": husband, "wife": wife,
        "friend": friend, "gender": gender, "name_lower": name_lower,
        "hobby": hobby, "job": job, "dob": dob, "names": n2f,
        "name2job": name2job, "name2hobby": name2hobby, "name2dob": name2dob,
    }


def _get_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    with _GRAPH_LOCK:
        if _GRAPH is None:
            _GRAPH = _build_graph()
    return _GRAPH


def _union(sets):
    out = set()
    for s in sets:
        out |= s
    return out


def _make_resolvers(G):
    mother, father = G["mother"], G["father"]
    brother, sister = G["brother"], G["sister"]
    son, daughter = G["son"], G["daughter"]
    husband, wife = G["husband"], G["wife"]
    friend, gender = G["friend"], G["gender"]

    def is_male(n):
        return gender.get(n) == "male"

    def is_female(n):
        return gender.get(n) == "female"

    # Base-relation accessors (the dicts are NOT callable, so wrap them once).
    def mothers(x):
        return mother.get(x, set())

    def fathers(x):
        return father.get(x, set())

    def brothers(x):
        return brother.get(x, set())

    def sisters(x):
        return sister.get(x, set())

    def sons(x):
        return son.get(x, set())

    def daughters(x):
        return daughter.get(x, set())

    def husbands(x):
        return husband.get(x, set())

    def wives(x):
        return wife.get(x, set())

    def friends(x):
        return friend.get(x, set())

    def parents(x):
        return mothers(x) | fathers(x)

    def children(x):
        return sons(x) | daughters(x)

    def siblings(x):
        return brothers(x) | sisters(x)

    def spouses(x):
        return husbands(x) | wives(x)

    def grandparents(x):
        return _union(parents(p) for p in parents(x))

    def grandchildren(x):
        return _union(children(c) for c in children(x))

    def great_grandparents(x):
        return _union(parents(g) for g in grandparents(x))

    def great_grandchildren(x):
        return _union(children(g) for g in grandchildren(x))

    def female_of(setfn):
        def f(x):
            return {y for y in setfn(x) if is_female(y)}
        return f

    def male_of(setfn):
        def f(x):
            return {y for y in setfn(x) if is_male(y)}
        return f

    def uncles(x):
        return _union(brothers(p) for p in parents(x))

    def aunts(x):
        return _union(sisters(p) for p in parents(x))

    def great_uncles(x):
        return _union(brothers(g) for g in grandparents(x))

    def great_aunts(x):
        return _union(sisters(g) for g in grandparents(x))

    def second_uncles(x):
        return _union(brothers(g) for g in great_grandparents(x))

    def second_aunts(x):
        return _union(sisters(g) for g in great_grandparents(x))

    def cousins(x):
        out = set()
        for p in parents(x):
            for sib in siblings(p):
                out |= children(sib)
        out.discard(x)
        return out

    def second_cousins(x):
        out = set()
        for p in parents(x):
            for c in cousins(p):
                out |= children(c)
        out.discard(x)
        return out

    def first_cousin_once_removed(x):
        return _union(children(c) for c in cousins(x))

    def nieces(x):
        return _union(daughters(s) for s in siblings(x))

    def nephews(x):
        return _union(sons(s) for s in siblings(x))

    def mother_in_law(x):
        return _union(mothers(s) for s in spouses(x))

    def father_in_law(x):
        return _union(fathers(s) for s in spouses(x))

    def sister_in_law(x):
        return _union(sisters(s) for s in spouses(x))

    def brother_in_law(x):
        return _union(brothers(s) for s in spouses(x))

    def son_in_law(x):
        return _union(husbands(c) for c in children(x))

    def daughter_in_law(x):
        return _union(wives(c) for c in children(x))

    R = {
        "parent": parents, "parents": parents,
        "mother": lambda x: mother.get(x, set()), "mothers": lambda x: mother.get(x, set()),
        "father": lambda x: father.get(x, set()), "fathers": lambda x: father.get(x, set()),
        "child": children, "children": children,
        "son": lambda x: son.get(x, set()), "sons": lambda x: son.get(x, set()),
        "daughter": lambda x: daughter.get(x, set()), "daughters": lambda x: daughter.get(x, set()),
        "sibling": siblings, "siblings": siblings,
        "brother": lambda x: brother.get(x, set()), "brothers": lambda x: brother.get(x, set()),
        "sister": lambda x: sister.get(x, set()), "sisters": lambda x: sister.get(x, set()),
        "spouse": spouses, "spouses": spouses,
        "husband": lambda x: husband.get(x, set()), "husbands": lambda x: husband.get(x, set()),
        "wife": lambda x: wife.get(x, set()), "wives": lambda x: wife.get(x, set()),
        "friend": lambda x: friend.get(x, set()), "friends": lambda x: friend.get(x, set()),
        "grandparent": grandparents, "grandparents": grandparents,
        "grandmother": female_of(grandparents), "grandmothers": female_of(grandparents),
        "grandfather": male_of(grandparents), "grandfathers": male_of(grandparents),
        "grandchild": grandchildren, "grandchildren": grandchildren,
        "grandson": male_of(grandchildren), "grandsons": male_of(grandchildren),
        "granddaughter": female_of(grandchildren), "granddaughters": female_of(grandchildren),
        "great_grandparent": great_grandparents, "great_grandparents": great_grandparents,
        "great_grandmother": female_of(great_grandparents), "great_grandmothers": female_of(great_grandparents),
        "great_grandfather": male_of(great_grandparents), "great_grandfathers": male_of(great_grandparents),
        "great_grandchild": great_grandchildren, "great_grandchildren": great_grandchildren,
        "great_grandson": male_of(great_grandchildren), "great_grandsons": male_of(great_grandchildren),
        "great_granddaughter": female_of(great_grandchildren), "great_granddaughters": female_of(great_grandchildren),
        "uncle": uncles, "uncles": uncles,
        "aunt": aunts, "aunts": aunts,
        "great_uncle": great_uncles, "great_uncles": great_uncles,
        "great_aunt": great_aunts, "great_aunts": great_aunts,
        "second_uncle": second_uncles, "second_uncles": second_uncles,
        "second_aunt": second_aunts, "second_aunts": second_aunts,
        "cousin": cousins, "cousins": cousins,
        "female_cousin": female_of(cousins), "female_cousins": female_of(cousins),
        "male_cousin": male_of(cousins), "male_cousins": male_of(cousins),
        "second_cousin": second_cousins, "second_cousins": second_cousins,
        "female_second_cousin": female_of(second_cousins), "female_second_cousins": female_of(second_cousins),
        "male_second_cousin": male_of(second_cousins), "male_second_cousins": male_of(second_cousins),
        "first_cousin_once_removed": first_cousin_once_removed,
        "female_first_cousin_once_removed": female_of(first_cousin_once_removed),
        "male_first_cousin_once_removed": male_of(first_cousin_once_removed),
        "niece": nieces, "nieces": nieces,
        "nephew": nephews, "nephews": nephews,
        "mother_in_law": mother_in_law, "mothers_in_law": mother_in_law,
        "father_in_law": father_in_law, "fathers_in_law": father_in_law,
        "sister_in_law": sister_in_law, "sisters_in_law": sister_in_law,
        "brother_in_law": brother_in_law, "brothers_in_law": brother_in_law,
        "son_in_law": son_in_law, "sons_in_law": son_in_law,
        "daughter_in_law": daughter_in_law, "daughters_in_law": daughter_in_law,
    }
    return R


_RESOLVERS = None


def _get_resolvers():
    global _RESOLVERS
    if _RESOLVERS is None:
        G = _get_graph()
        if G is None:
            return None
        _RESOLVERS = _make_resolvers(G)
    return _RESOLVERS


def _canonicalize(surface):
    s = surface.strip().lower()
    s = re.sub(r"^the\s+", "", s)
    R = _get_resolvers()
    if R is None:
        return None
    if s in R:
        return s
    s2 = s.replace("-", " ")
    s2 = re.sub(r"\s+in\s+law\b", "_in_law", s2)
    s2 = re.sub(r"\s+", "_", s2)
    if s2 in R:
        return s2
    cands = set()
    for suf in ("ies", "es", "s"):
        if s2.endswith(suf):
            base = s2[: -len(suf)]
            cands.add(base)
            if suf == "ies":
                cands.add(base + "y")
    m = re.match(r"^(.+?)s_in_law$", s2)
    if m:
        cands.add(m.group(1) + "_in_law")
    if s2.endswith("children"):
        cands.add(s2[: -len("ren")])
    m = re.match(r"^(.+?)cousins_once_removed$", s2)
    if m:
        cands.add(m.group(1) + "cousin_once_removed")
    for c in cands:
        if c in R:
            return c
    if s2.endswith("s") and not s2.endswith("ss"):
        cand = s2[:-1]
        if cand in R:
            return cand
    return None


_ANCHOR_PAT = re.compile(
    r"(?:the\s+)?person whose (occupation|hobby|date of birth) is (.+)$",
    re.IGNORECASE,
)
_Q_PAT = re.compile(r"\s*how many (.+?) does (.+?) have\?\s*$", re.IGNORECASE)


def _lookup_name(G, n):
    names = G["names"]
    if n in names:
        return n
    return G["name_lower"].get(n.lower())


def _parse_count(q, G, R):
    m = _Q_PAT.match(q.strip())
    if not m:
        return None
    r1 = _canonicalize(m.group(1).strip())
    if r1 is None:
        return None
    entity = m.group(2).strip()

    chain = []
    am = _ANCHOR_PAT.search(entity)
    if am:
        attr = am.group(1).lower()
        value = am.group(2).strip()
        chain_text = entity[: am.start()]
        src = {"occupation": G["job"], "hobby": G["hobby"], "date of birth": G["dob"]}.get(attr)
        if src is None:
            return None
        anchors = list(src.get(value, []))
    else:
        parts = re.split(r"\s+of\s+(?:the\s+)?", entity)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        name = re.sub(r"^the\s+", "", parts[-1]).strip()
        resolved = _lookup_name(G, name)
        anchors = [resolved] if resolved else []
        chain_text = " of ".join(parts[:-1])

    if chain_text.strip():
        rels = re.split(r"\s+of\s+(?:the\s+)?", chain_text)
        rels = [r for r in rels if r.strip()]
        for r in reversed(rels):
            c = _canonicalize(r)
            if c is None:
                return None
            chain.append(c)
    return r1, chain, anchors


def solve_count(question):
    """Return the exact distinct count set (set[str]) for a "How many ...?"
    question, or None if it cannot be parsed/resolved."""
    G = _get_graph()
    if G is None:
        return None
    R = _get_resolvers()
    if R is None:
        return None
    parsed = _parse_count(question, G, R)
    if parsed is None:
        return None
    r1, chain, anchors = parsed
    if not anchors:
        return None
    current = set(anchors)
    for rel in chain:
        nxt = set()
        fn = R[rel]
        for p in current:
            nxt |= fn(p)
        current = nxt
    counts = set()
    fn1 = R[r1]
    for e in current:
        counts.add(str(len(fn1(e))))
    return counts


# ---------------------------------------------------------------------------
# List ("Who is ...?" / "What is the <attr> of ...?") questions
# ---------------------------------------------------------------------------
_WHO_RE = re.compile(r"\s*who is the (.+?)\?\s*$", re.IGNORECASE)
_WHAT_RE = re.compile(
    r"\s*what is the (occupation|hobby|date of birth|gender) of the (.+?)\?\s*$",
    re.IGNORECASE,
)


def _parse_entity(entity, G, R):
    """Parse a relation-chain entity into (chain innermost-first, anchors)."""
    chain = []
    am = _ANCHOR_PAT.search(entity)
    if am:
        attr = am.group(1).lower()
        value = am.group(2).strip()
        chain_text = entity[: am.start()]
        src = {"occupation": G["job"], "hobby": G["hobby"], "date of birth": G["dob"]}.get(attr)
        if src is None:
            return None
        anchors = list(src.get(value, []))
    else:
        parts = re.split(r"\s+of\s+(?:the\s+)?", entity)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        name = re.sub(r"^the\s+", "", parts[-1]).strip()
        resolved = _lookup_name(G, name)
        anchors = [resolved] if resolved else []
        chain_text = " of ".join(parts[:-1])

    if chain_text.strip():
        rels = re.split(r"\s+of\s+(?:the\s+)?", chain_text)
        rels = [r for r in rels if r.strip()]
        for r in reversed(rels):
            c = _canonicalize(r)
            if c is None:
                return None
            chain.append(c)
    return chain, anchors


def solve_list(question):
    """Return the exact answer set (set[str]) for a "Who is ...?" or
    "What is the <attr> of ...?" question, or None if it cannot be
    parsed/resolved."""
    G = _get_graph()
    if G is None:
        return None
    R = _get_resolvers()
    if R is None:
        return None
    q = question.strip()
    m = _WHO_RE.match(q)
    asked = None
    if m:
        entity = "the " + m.group(1).strip()
    else:
        m = _WHAT_RE.match(q)
        if m is None:
            return None
        asked = m.group(1).lower()
        entity = "the " + m.group(2).strip()
    parsed = _parse_entity(entity, G, R)
    if parsed is None:
        return None
    chain, anchors = parsed
    if not anchors:
        return None
    current = set(anchors)
    for rel in chain:
        nxt = set()
        fn = R[rel]
        for p in current:
            nxt |= fn(p)
        current = nxt
    if asked is None:
        # "Who is ...?" -> the resolved people themselves.
        return set(current)
    # "What is the <attr> of ...?" -> distinct attribute values.
    vals = set()
    if asked == "occupation":
        nm = G["name2job"]
    elif asked == "hobby":
        nm = G["name2hobby"]
    elif asked == "date of birth":
        nm = G["name2dob"]
    elif asked == "gender":
        nm = None
    else:
        return None
    if asked == "gender":
        for p in current:
            g = G["gender"].get(p)
            if g:
                vals.add(g)
    else:
        for p in current:
            v = nm.get(p)
            if v is not None:
                vals.add(v)
    return vals
