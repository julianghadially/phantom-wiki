"""Deterministic relation solver for PhantomWiki templated questions.

The PhantomWiki questions are template-based and reference a single fixed corpus.
The exact Prolog semantics of every derived relation are defined in
``output/depth_10_size_1000000/facts.pl``.  This module reconstructs those
relations from the base facts (``parent`` + ``gender`` + ``friend_``) that are
exposed in the article ``mother``/``father`` lines, parses the templated English
question into a relation chain, and computes the exact answer deterministically.

When :func:`solve` cannot parse a question it returns ``None`` so the caller can
fall back to the ReAct agent.
"""

from __future__ import annotations

import re
import threading

from src.program.phantomwiki_module import _load_corpus_index


# ---------------------------------------------------------------------------
# Index construction (process-wide cache)
# ---------------------------------------------------------------------------

_INDEX = None
_INDEX_LOCK = threading.Lock()


def _build_index():
    idx = _load_corpus_index()
    name2facts = idx["name2facts"]

    parents = {}
    children = {}
    gender = {}
    friend = {}
    dob = {}
    job = {}
    hobby = {}

    pat = re.compile(r'(\w+)\("(.+?)", "(.+?)"\)\.')

    def add_parent(x, y):
        s = parents.get(x)
        if s is None:
            s = set()
            parents[x] = s
        s.add(y)
        sc = children.get(y)
        if sc is None:
            sc = set()
            children[y] = sc
        sc.add(x)

    for name, facts in name2facts.items():
        for f in facts:
            m = pat.match(f)
            if not m:
                continue
            rel, subj, obj = m.group(1), m.group(2), m.group(3)
            if rel in ("mother", "father"):
                add_parent(subj, obj)
            elif rel == "gender":
                gender[subj] = obj
            elif rel == "dob":
                dob[subj] = obj
            elif rel == "job":
                job[subj] = obj
            elif rel == "hobby":
                hobby[subj] = obj
            elif rel == "friend":
                fs = friend.get(subj)
                if fs is None:
                    fs = set()
                    friend[subj] = fs
                fs.add(obj)
                fs2 = friend.get(obj)
                if fs2 is None:
                    fs2 = set()
                    friend[obj] = fs2
                fs2.add(subj)

    return {
        "parents": parents,
        "children": children,
        "gender": gender,
        "friend": friend,
        "dob": dob,
        "job": job,
        "hobby": hobby,
        # reverse attribute indices (value -> set of people) from the corpus index
        "dob_rev": idx["dob"],
        "job_rev": idx["job"],
        "hobby_rev": idx["hobby"],
    }


def get_index():
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = _build_index()
    return _INDEX


# ---------------------------------------------------------------------------
# Base relation helpers
# ---------------------------------------------------------------------------

def _set(d, k):
    s = d.get(k)
    return s if s is not None else set()


def _female(x, gender):
    return gender.get(x) == "female"


def _male(x, gender):
    return gender.get(x) == "male"


# ---------------------------------------------------------------------------
# Relation predicates. Each returns the set of Y such that relation(X, Y).
# ---------------------------------------------------------------------------

def _rel_parents(I, x):
    return set(_set(I["parents"], x))


def _rel_children(I, x):
    return set(_set(I["children"], x))


def _rel_mothers(I, x):
    g = I["gender"]
    return {p for p in _set(I["parents"], x) if _female(p, g)}


def _rel_fathers(I, x):
    g = I["gender"]
    return {p for p in _set(I["parents"], x) if _male(p, g)}


def _rel_sibling(I, x):
    ps = _set(I["parents"], x)
    if not ps:
        return set()
    out = set()
    ch = I["children"]
    for p in ps:
        cp = ch.get(p)
        if cp:
            out |= cp
    out.discard(x)
    return out


def _rel_brother(I, x):
    g = I["gender"]
    return {s for s in _rel_sibling(I, x) if _male(s, g)}


def _rel_sister(I, x):
    g = I["gender"]
    return {s for s in _rel_sibling(I, x) if _female(s, g)}


def _rel_spouse(I, x):
    kids = _set(I["children"], x)
    if not kids:
        return set()
    out = set()
    pr = I["parents"]
    for c in kids:
        cp = pr.get(c)
        if cp:
            out |= cp
    out.discard(x)
    return out


def _rel_husband(I, x):
    g = I["gender"]
    return {s for s in _rel_spouse(I, x) if _male(s, g)}


def _rel_wife(I, x):
    g = I["gender"]
    return {s for s in _rel_spouse(I, x) if _female(s, g)}


def _rel_friend(I, x):
    return set(_set(I["friend"], x))


def _rel_grandparent(I, x):
    out = set()
    pr = I["parents"]
    for p in _set(pr, x):
        pp = pr.get(p)
        if pp:
            out |= pp
    return out


def _rel_children_many(I, xs):
    out = set()
    ch = I["children"]
    for x in xs:
        cp = ch.get(x)
        if cp:
            out |= cp
    return out


def _rel_parents_many(I, xs):
    out = set()
    pr = I["parents"]
    for x in xs:
        pp = pr.get(x)
        if pp:
            out |= pp
    return out


def _rel_grandchild(I, x):
    return _rel_children_many(I, _rel_children(I, x))


def _rel_great_grandparent(I, x):
    return _rel_parents_many(I, _rel_grandparent(I, x))


def _rel_great_grandchild(I, x):
    return _rel_children_many(I, _rel_grandchild(I, x))


def _gfilter(I, ys, which):
    g = I["gender"]
    if which == "female":
        return {y for y in ys if _female(y, g)}
    elif which == "male":
        return {y for y in ys if _male(y, g)}
    return ys


def _rel_uncle(I, x):
    g = I["gender"]
    out = set()
    for p in _set(I["parents"], x):
        for s in _rel_sibling(I, p):
            if _male(s, g):
                out.add(s)
    return out


def _rel_aunt(I, x):
    g = I["gender"]
    out = set()
    for p in _set(I["parents"], x):
        for s in _rel_sibling(I, p):
            if _female(s, g):
                out.add(s)
    return out


def _rel_great_uncle(I, x):
    g = I["gender"]
    out = set()
    for gp in _rel_grandparent(I, x):
        for s in _rel_sibling(I, gp):
            if _male(s, g):
                out.add(s)
    return out


def _rel_great_aunt(I, x):
    g = I["gender"]
    out = set()
    for gp in _rel_grandparent(I, x):
        for s in _rel_sibling(I, gp):
            if _female(s, g):
                out.add(s)
    return out


def _rel_second_uncle(I, x):
    g = I["gender"]
    out = set()
    for ggp in _rel_great_grandparent(I, x):
        for s in _rel_sibling(I, ggp):
            if _male(s, g):
                out.add(s)
    return out


def _rel_second_aunt(I, x):
    g = I["gender"]
    out = set()
    for ggp in _rel_great_grandparent(I, x):
        for s in _rel_sibling(I, ggp):
            if _female(s, g):
                out.add(s)
    return out


def _rel_nephew(I, x):
    g = I["gender"]
    out = set()
    ch = I["children"]
    for s in _rel_sibling(I, x):
        for c in ch.get(s, set()):
            if _male(c, g):
                out.add(c)
    return out


def _rel_niece(I, x):
    g = I["gender"]
    out = set()
    ch = I["children"]
    for s in _rel_sibling(I, x):
        for c in ch.get(s, set()):
            if _female(c, g):
                out.add(c)
    return out


def _rel_cousin(I, x):
    # cousin(X,Y) :- parent(X,A), parent(Y,B), sibling(A,B), X\=Y
    out = set()
    ch = I["children"]
    for a in _set(I["parents"], x):
        for sib in _rel_sibling(I, a):
            cp = ch.get(sib)
            if cp:
                out |= cp
    out.discard(x)
    return out


def _rel_second_cousin(I, x):
    # second_cousin(X,Y) :- parent(X,A), parent(Y,B), cousin(A,B), X\=Y
    out = set()
    ch = I["children"]
    for a in _set(I["parents"], x):
        for bc in _rel_cousin(I, a):
            cp = ch.get(bc)
            if cp:
                out |= cp
    out.discard(x)
    return out


def _rel_first_cousin_once_removed(I, x):
    # first_cousin_once_removed(X,Y) :- cousin(X,A), child(A,Y), X\=Y
    ch = I["children"]
    out = set()
    for a in _rel_cousin(I, x):
        cp = ch.get(a)
        if cp:
            out |= cp
    out.discard(x)
    return out


def _rel_mother_in_law(I, x):
    g = I["gender"]
    pr = I["parents"]
    out = set()
    for s in _rel_spouse(I, x):
        for p in pr.get(s, set()):
            if _female(p, g):
                out.add(p)
    return out


def _rel_father_in_law(I, x):
    g = I["gender"]
    pr = I["parents"]
    out = set()
    for s in _rel_spouse(I, x):
        for p in pr.get(s, set()):
            if _male(p, g):
                out.add(p)
    return out


def _rel_brother_in_law(I, x):
    g = I["gender"]
    out = set()
    for s in _rel_spouse(I, x):
        for b in _rel_sibling(I, s):
            if _male(b, g):
                out.add(b)
    return out


def _rel_sister_in_law(I, x):
    g = I["gender"]
    out = set()
    for s in _rel_spouse(I, x):
        for b in _rel_sibling(I, s):
            if _female(b, g):
                out.add(b)
    return out


def _rel_son_in_law(I, x):
    # son_in_law(X,Y) :- child(X,A), husband(A,Y)
    g = I["gender"]
    ch = I["children"]
    out = set()
    for a in ch.get(x, set()):
        for s in _rel_spouse(I, a):
            if _male(s, g):
                out.add(s)
    return out


def _rel_daughter_in_law(I, x):
    g = I["gender"]
    ch = I["children"]
    out = set()
    for a in ch.get(x, set()):
        for s in _rel_spouse(I, a):
            if _female(s, g):
                out.add(s)
    return out


PREDICATES = {
    "mother": _rel_mothers,
    "father": _rel_fathers,
    "parent": _rel_parents,
    "son": lambda I, x: _gfilter(I, _rel_children(I, x), "male"),
    "daughter": lambda I, x: _gfilter(I, _rel_children(I, x), "female"),
    "child": _rel_children,
    "brother": _rel_brother,
    "sister": _rel_sister,
    "sibling": _rel_sibling,
    "spouse": _rel_spouse,
    "husband": _rel_husband,
    "wife": _rel_wife,
    "friend": _rel_friend,
    "grandparent": _rel_grandparent,
    "grandmother": lambda I, x: _gfilter(I, _rel_grandparent(I, x), "female"),
    "grandfather": lambda I, x: _gfilter(I, _rel_grandparent(I, x), "male"),
    "grandchild": _rel_grandchild,
    "grandson": lambda I, x: _gfilter(I, _rel_grandchild(I, x), "male"),
    "granddaughter": lambda I, x: _gfilter(I, _rel_grandchild(I, x), "female"),
    "great_grandparent": _rel_great_grandparent,
    "great_grandmother": lambda I, x: _gfilter(I, _rel_great_grandparent(I, x), "female"),
    "great_grandfather": lambda I, x: _gfilter(I, _rel_great_grandparent(I, x), "male"),
    "great_grandchild": _rel_great_grandchild,
    "great_grandson": lambda I, x: _gfilter(I, _rel_great_grandchild(I, x), "male"),
    "great_granddaughter": lambda I, x: _gfilter(I, _rel_great_grandchild(I, x), "female"),
    "aunt": _rel_aunt,
    "uncle": _rel_uncle,
    "great_aunt": _rel_great_aunt,
    "great_uncle": _rel_great_uncle,
    "second_aunt": _rel_second_aunt,
    "second_uncle": _rel_second_uncle,
    "niece": _rel_niece,
    "nephew": _rel_nephew,
    "cousin": _rel_cousin,
    "second_cousin": _rel_second_cousin,
    "first_cousin_once_removed": _rel_first_cousin_once_removed,
    "female_cousin": lambda I, x: _gfilter(I, _rel_cousin(I, x), "female"),
    "male_cousin": lambda I, x: _gfilter(I, _rel_cousin(I, x), "male"),
    "female_second_cousin": lambda I, x: _gfilter(I, _rel_second_cousin(I, x), "female"),
    "male_second_cousin": lambda I, x: _gfilter(I, _rel_second_cousin(I, x), "male"),
    "female_first_cousin_once_removed": lambda I, x: _gfilter(I, _rel_first_cousin_once_removed(I, x), "female"),
    "male_first_cousin_once_removed": lambda I, x: _gfilter(I, _rel_first_cousin_once_removed(I, x), "male"),
    "mother_in_law": _rel_mother_in_law,
    "father_in_law": _rel_father_in_law,
    "brother_in_law": _rel_brother_in_law,
    "sister_in_law": _rel_sister_in_law,
    "son_in_law": _rel_son_in_law,
    "daughter_in_law": _rel_daughter_in_law,
}


# ---------------------------------------------------------------------------
# Surface-phrase -> predicate-name map (singular + plural)
# ---------------------------------------------------------------------------

def _norm(s):
    return s.lower().replace("-", " ").strip()


_BASE_PHRASES = {
    "mother": "mother", "mother": "mother",
    "father": "father",
    "parent": "parent", "parents": "parent",
    "son": "son",
    "daughter": "daughter",
    "child": "child", "children": "child",
    "brother": "brother",
    "sister": "sister",
    "sibling": "sibling", "siblings": "sibling",
    "spouse": "spouse",
    "husband": "husband",
    "wife": "wife",
    "friend": "friend", "friends": "friend",
    "grandparent": "grandparent", "grandparents": "grandparent",
    "grandmother": "grandmother",
    "grandfather": "grandfather",
    "grandchild": "grandchild", "grandchildren": "grandchild",
    "grandson": "grandson", "grandsons": "grandson",
    "granddaughter": "granddaughter", "granddaughters": "granddaughter",
    "great-grandparent": "great_grandparent", "great-grandparents": "great_grandparent",
    "great-grandmother": "great_grandmother",
    "great-grandfather": "great_grandfather",
    "great-grandchild": "great_grandchild", "great-grandchildren": "great_grandchild",
    "great-grandson": "great_grandson", "great-grandsons": "great_grandson",
    "great-granddaughter": "great_granddaughter", "great-granddaughters": "great_granddaughter",
    "aunt": "aunt", "aunts": "aunt",
    "uncle": "uncle", "uncles": "uncle",
    "great-aunt": "great_aunt", "great-aunts": "great_aunt",
    "great-uncle": "great_uncle", "great-uncles": "great_uncle",
    "second-aunt": "second_aunt", "second-aunts": "second_aunt",
    "second-uncle": "second_uncle", "second-uncles": "second_uncle",
    "niece": "niece", "nieces": "niece",
    "nephew": "nephew", "nephews": "nephew",
    "cousin": "cousin", "cousins": "cousin",
    "second-cousin": "second_cousin", "second-cousins": "second_cousin",
    "first-cousin-once-removed": "first_cousin_once_removed",
    "first-cousins-once-removed": "first_cousin_once_removed",
    "female cousin": "female_cousin", "female cousins": "female_cousin",
    "male cousin": "male_cousin", "male cousins": "male_cousin",
    "female second cousin": "female_second_cousin", "female second cousins": "female_second_cousin",
    "male second cousin": "male_second_cousin", "male second cousins": "male_second_cousin",
    "female first cousin once removed": "female_first_cousin_once_removed",
    "female first cousins once removed": "female_first_cousin_once_removed",
    "male first cousin once removed": "male_first_cousin_once_removed",
    "male first cousins once removed": "male_first_cousin_once_removed",
    "mother-in-law": "mother_in_law", "mothers-in-law": "mother_in_law",
    "father-in-law": "father_in_law", "fathers-in-law": "father_in_law",
    "brother-in-law": "brother_in_law", "brothers-in-law": "brother_in_law",
    "sister-in-law": "sister_in_law", "sisters-in-law": "sister_in_law",
    "son-in-law": "son_in_law", "sons-in-law": "son_in_law",
    "daughter-in-law": "daughter_in_law", "daughters-in-law": "daughter_in_law",
}

PHRASE_MAP = {_norm(k): v for k, v in _BASE_PHRASES.items()}
# add simple -s plurals for plain nouns not explicitly listed above
for _sing, _pl in [
    ("mother", "mothers"), ("father", "fathers"), ("son", "sons"),
    ("daughter", "daughters"), ("brother", "brothers"), ("sister", "sisters"),
    ("grandmother", "grandmothers"), ("grandfather", "grandfathers"),
    ("grandson", "grandsons"), ("granddaughter", "granddaughters"),
    ("great-grandmother", "great-grandmothers"),
    ("great-grandfather", "great-grandfathers"),
    ("parent", "parents"), ("child", "children"), ("sibling", "siblings"),
    ("cousin", "cousins"), ("niece", "nieces"), ("nephew", "nephews"),
    ("friend", "friends"), ("aunt", "aunts"), ("uncle", "uncles"),
    ("husband", "husbands"), ("wife", "wives"), ("spouse", "spouses"),
]:
    _ns = _norm(_pl)
    if _ns not in PHRASE_MAP:
        PHRASE_MAP[_ns] = PHRASE_MAP[_norm(_sing)]
PHRASES_SORTED = sorted(PHRASE_MAP.keys(), key=len, reverse=True)


_ATTR_CANON = {"date of birth": "dob", "occupation": "job", "hobby": "hobby"}


# ---------------------------------------------------------------------------
# Question parser
# ---------------------------------------------------------------------------

def _map_rel(phrase):
    return PHRASE_MAP.get(_norm(phrase))


def _parse_anchor(text):
    text = text.strip()
    low = text.lower()
    # tolerate a leading "the " that survived chain peeling
    if low.startswith("the "):
        text = text[4:].strip()
        low = text.lower()
    if low.startswith("person whose "):
        rest = text[len("person whose "):].strip()
        m = re.match(r"(date of birth|occupation|hobby) is (.+)$", rest, re.IGNORECASE)
        if not m:
            return None
        attr = m.group(1).lower()
        value = m.group(2).strip()
        return {"kind": "attr", "attr": _ATTR_CANON[attr], "value": value}
    if re.fullmatch(r"[A-Z][A-Za-z.\'-]+(?: [A-Z][A-Za-z.\'-]+)+", text):
        return {"kind": "name", "value": text}
    if re.fullmatch(r"[A-Z][A-Za-z.\'-]+", text):
        return {"kind": "name", "value": text}
    return None


def _parse_subject(text):
    rest = text.strip()
    steps = []
    while True:
        rest = rest.strip()
        low = _norm(rest)
        if low.startswith("the "):
            rest = rest[4:]
            continue
        matched = None
        for ph in PHRASES_SORTED:
            if low.startswith(ph):
                after = rest[len(ph):]
                if after and after[0] not in (" ", "-"):
                    continue
                after_strip = after.lstrip()
                asl = after_strip.lower()
                if asl.startswith("of ") or asl == "of":
                    matched = ph
                    rest = after_strip[3:] if asl.startswith("of ") else ""
                    break
        if matched is None:
            break
        steps.append(PHRASE_MAP[matched])
    anchor = _parse_anchor(rest)
    if anchor is None:
        return None
    return anchor, steps


def parse_question(q):
    q = q.strip()
    bare = q[:-1].strip() if q.endswith("?") else q

    m = re.match(r"^How many (.+?) does (.+) have$", bare)
    if m:
        counted = _map_rel(m.group(1).strip())
        if counted is None:
            return None
        parsed = _parse_subject(m.group(2).strip())
        if parsed is None:
            return None
        anchor, steps = parsed
        return {"mode": "count", "counted": counted, "anchor": anchor, "steps": steps}

    m = re.match(r"^Who is (.+)$", bare)
    if m:
        parsed = _parse_subject(m.group(1).strip())
        if parsed is None:
            return None
        anchor, steps = parsed
        return {"mode": "names", "anchor": anchor, "steps": steps}

    for ap in ("date of birth", "occupation", "hobby"):
        m = re.match(r"^What is the " + ap + r" of (.+)$", bare, re.IGNORECASE)
        if m:
            parsed = _parse_subject(m.group(1).strip())
            if parsed is None:
                return None
            anchor, steps = parsed
            return {"mode": "attr", "attr": _ATTR_CANON[ap], "anchor": anchor, "steps": steps}

    return None


# ---------------------------------------------------------------------------
# Anchor resolution -> set of person names
# ---------------------------------------------------------------------------

def _resolve_anchor(I, anchor):
    if anchor["kind"] == "name":
        return {anchor["value"]}
    attr = anchor["attr"]
    value = anchor["value"]
    rev = I.get(attr + "_rev")
    if rev is None:
        return set()
    ppl = rev.get(value)
    return set(ppl) if ppl else set()


def _apply_rel(I, rel, x):
    fn = PREDICATES.get(rel)
    if fn is None:
        return set()
    return fn(I, x)


def compute(question):
    spec = parse_question(question)
    if spec is None:
        return None
    I = get_index()
    anchors = _resolve_anchor(I, spec["anchor"])

    cur = set(anchors)
    for rel in reversed(spec["steps"]):
        nxt = set()
        for x in cur:
            nxt |= _apply_rel(I, rel, x)
        cur = nxt

    if spec["mode"] == "count":
        counted = spec["counted"]
        counts = set()
        for s in cur:
            counts.add(len(_apply_rel(I, counted, s)))
        return sorted((str(c) for c in counts))

    if spec["mode"] == "names":
        return sorted(cur)

    # attribute
    field = I[spec["attr"]]
    vals = set()
    for name in cur:
        v = field.get(name)
        if v is not None:
            vals.add(v)
    return sorted(vals)


def solve(question):
    """Return a list[str] answer, or None if the question cannot be parsed."""
    try:
        return compute(question)
    except Exception:
        return None
