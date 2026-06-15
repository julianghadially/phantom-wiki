import dspy
import json
import os

_date_index_cache = None


def _load_date_index():
    global _date_index_cache
    if _date_index_cache is None:
        index_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../output/depth_10_size_1000000/date_passages.json"
        )
        with open(index_path, "r") as f:
            _date_index_cache = json.load(f)
    return _date_index_cache


class PhantomWikiQA(dspy.Signature):
    """You are a meticulous researcher answering questions from PhantomWiki, a fictional encyclopedia.

    IMPORTANT: Questions frequently have MULTIPLE correct answers (sometimes 10 or more). Your mission is to find ALL of them — not just the first one you encounter.

    HOW TO SEARCH:
    1. Break down the question into its component parts: identify anchor entities, relationships, and target attributes
    2. Search for each component separately with targeted queries
    3. For ancestry/family questions: after tracing one branch, ALWAYS search for siblings and other family branches — every sibling of an ancestor is another potential answer path
    4. For attribute-lookup questions ("What is X of the person whose Y is Z"): multiple people may share the same attribute value — after finding one match, keep searching for others
    5. After every answer found, ask yourself: "Are there more?" and issue additional searches
    6. If a search fails, try completely different angles: search by a related name, a different relationship, or a nearby attribute
    7. Singular-form questions ("the cousin of...") may still resolve to MULTIPLE entities — enumerate all entities at each hop before proceeding

    HOW MANY FORMAT RULE:
    - If the question asks "How many X does Y have?", your answer MUST be the numeric count as a string, not entity names
    - WRONG: ["Alice Smith", "Bob Jones"] — these are names, not a count
    - RIGHT: ["2"] — this is the count
    - If the question asks "How many X does each Y have?" and Y resolves to multiple entities, return the count for EACH Y as a separate string: e.g., ["0", "2", "5"]
    - Count strings must be plain integers: "0", "1", "2", "10" — never "two" or "zero"

    MULTI-ENTITY ENUMERATION:
    - For ancestor-based chains (great-grandparent, great-uncle, second uncle, second cousin): ALWAYS search BOTH the maternal AND paternal branches separately. Families split across different surnames — issue a separate search for each parent's family line.
    - For friend/sibling/colleague chains: enumerate ALL friends/siblings/colleagues of the anchor entity before computing any count or tracing further
    - For DOB/hobby/occupation anchor questions: after finding the first matching person, issue at least 2 more follow-up searches — many more people may share the same attribute
    - After finding N entities, ask: "Am I missing other branches?" and search the other side of the family tree

    ATTRIBUTE FAN-OUT — For attribute-anchored questions ("person whose occupation is X" or "person whose hobby is X"):
    - Issue at least 5 different search query phrasings to find ALL people with that attribute
      Example for "financial controller": try "occupation financial controller", "job financial controller", "financial controller PhantomWiki", "works as financial controller", "career financial controller"
    - After finding initial results, issue at least 2 more follow-up searches with different phrasings
    - Enumerate ALL people found before computing any count or tracing further

    IMPLICIT RELATIONSHIPS — These kinship terms cannot be directly queried; they MUST be derived by traversal:
    - "cousin" = child of parent's sibling (search for parent's siblings, then their children)
    - "nephew" = sibling's son; "niece" = sibling's daughter
    - "uncle" = parent's brother; "aunt" = parent's sister
    - Never search "cousin of X" directly — it won't work. Traverse step by step.

    NON-STANDARD KINSHIP TERMS:
    - Second aunt/uncle = sibling of a GREAT-GRANDPARENT (go UP three generations to the great-grandparent, then find their siblings)
    - Second cousin = grandparent's sibling's grandchild (NOT great-grandparent's sibling)
    - First cousin once removed = parent's cousin OR cousin's child (one generation off from first cousins)
    - Great-uncle/aunt = grandparent's sibling
    - Grand-nephew/niece = sibling's grandchild

    DATE-OF-BIRTH ANCHOR:
    - If the question anchors on a specific date of birth (e.g., "person born on 0946-07-14"), ALWAYS use search_by_date_exact("0946-07-14") — this returns ALL people born on that exact date with 100% recall
    - TAUTOLOGICAL SHORTCUT: If the question asks "What is the date of birth of person whose date of birth is X?", return X directly without searching
    - After getting results from search_by_date_exact, read ALL returned articles and extract the relevant attribute for EACH person listed

    DO NOT:
    - Stop after finding just one answer
    - Assume a question has a unique answer
    - Give up with "unknown" or "cannot determine" after only a few searches — try at least 5 distinct approaches before concluding
    - Return intermediate entity names when the question asks for an attribute value (return "botany", not "Werner Corrigan — botany")
    - Sum counts across multiple entities when the question asks for PER-ENTITY counts — return each count as a separate string
    """

    question: str = dspy.InputField(
        desc="A question about fictional PhantomWiki entities, possibly requiring multi-hop reasoning"
    )
    answer: list[str] = dspy.OutputField(
        desc="A complete list of ALL correct answers found. Most questions have multiple answers. Search exhaustively before finishing. IMPORTANT: Return ONLY the exact answer values — do NOT include person names, attributions, or extra context alongside the answers. For example, if the question asks for occupations, return ['teacher', 'doctor'] NOT ['John Smith — teacher', 'Jane Doe — doctor']."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.retrieve_broad = dspy.Retrieve(k=30)

        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki, self.search_wiki_broad, self.search_by_date_exact],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus for entities, relationships, and attributes.
        Try different query angles: by person name, by relationship type, by attribute value, or by date.
        Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def search_wiki_broad(self, query: str) -> str:
        """Broader search returning top 30 results (3x more than search_wiki).
        Use for attribute-anchor questions (finding all people with a given hobby/occupation),
        or when regular search misses entities.
        Returns many relevant passages."""
        results = self.retrieve_broad(query)
        return "\n\n".join(results.passages)

    def search_by_date_exact(self, date_str: str) -> str:
        """Exact date-of-birth lookup. Given a date in YYYY-MM-DD format, returns ALL article texts
        for every person born on that exact date with 100% recall — no semantic approximation.
        Use for ANY question anchored by a specific date of birth."""
        date_index = _load_date_index()
        passages = date_index.get(date_str, [])
        if not passages:
            return f"No people found with date of birth {date_str}."
        return f"Found {len(passages)} people born on {date_str}:\n\n" + "\n\n".join(passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
