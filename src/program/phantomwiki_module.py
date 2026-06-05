import dspy


class ExhaustiveInvestigationSignature(dspy.Signature):
    """You are investigating a question about fictional characters in PhantomWiki.

    CRITICAL: Many questions have MULTIPLE correct answers. You must find ALL of them.

    STEP 1 — IDENTIFY AND COLLECT ALL ANCHORS:
    - Named anchor (e.g., "Deon Gall"): use search_wiki with the person's name.
    - Attribute anchor (person whose DOB/occupation/hobby is X): ALWAYS use find_all_by_attribute — NOT search_wiki. Multiple people share the same attribute. Collect ALL names from the results before investigating any chain.
    - find_all_by_attribute is ONLY for DOB, occupation, hobby, or nationality lookups. Do NOT use it for names, surnames, or relationship lookups — use search_wiki for those.
    - FORBIDDEN: stopping after finding the first matching person. Investigate EVERY anchor entity found.
    - If an anchor chain leads to a dead end (e.g., no spouse found), DON'T give up — there are likely OTHER anchors with the same attribute. Use find_all_by_attribute again or search_wiki with the other anchor names.

    STEP 2 — INVESTIGATE EACH ANCHOR SYSTEMATICALLY:
    1. For EACH anchor, follow the relationship chain step by step.
    2. When a page lists multiple relatives (children, siblings, friends), write down ALL names and search EACH one individually.
    3. Track which branches you have already investigated vs. which still remain.
    4. Do NOT stop after finding 1-2 answers if the question asks for all members of a set.

    STEP 3 — RELATIONSHIP SEMANTICS AND GENERATION COUNTING:
    - sister-in-law: your SPOUSE'S sister, OR your SIBLING'S wife — NOT your own sister
    - mother-in-law / father-in-law: your SPOUSE'S mother / father
    - grandmother/grandparent: your PARENT'S parent (2 hops up)
    - great-grandchild: child of your grandchild (3 hops down — must descend through grandchild first)
    - great-aunt: your GRANDPARENT'S sister (3 hops total: you→parent→grandparent→their sibling)
    - second cousin: child of your parent's first cousin
    GENERATION RULE: If you ascend N hops to reach an ancestor, great-grandchildren of that ancestor are N hops BELOW THAT ANCESTOR. For example, if you go up 3 hops to reach the great-grandfather, the great-grandchildren are 3 hops below the great-grandfather (the same generation as the original person). You must search the children of the grandchildren of the ancestor — not the children of the ancestor.

    STEP 4 — COUNTING FORMAT:
    - "How many X does each Y have?" → return EACH count separately, not summed.
    - Example: person A has 2, person B has 5 → return ["2", "5"] NOT ["7"].
    - For population queries: use find_all_by_attribute to find multiple people, count for each, return the DISTINCT set of values observed.

    COMPLETENESS CHECK before finalizing:
    - Did I use find_all_by_attribute for ALL DOB/occupation/hobby attribute lookups?
    - Did I investigate EVERY anchor entity found (not just the first)?
    - Did I search ALL entities at each intermediate level (all children, all siblings, all friends)?
    - Did I count generations correctly — am I at the right level of the family tree?
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="Complete list of ALL correct answers. Be exhaustive—include every valid answer found, not just the first few. For counting questions, return each count as a separate string. Return [] only if the entity truly does not exist."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        self.react = dspy.ReAct(
            signature=ExhaustiveInvestigationSignature,
            tools=[self.search_wiki, self.find_all_by_attribute],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus by a person's NAME or a descriptive phrase.
        Returns relevant passages.

        Use this tool for:
        - Looking up a specific person by name (e.g., "Alice Brown family")
        - Finding someone's family members, friends, hobbies, or occupation
        - Following up on a named entity from a previous search

        Do NOT use this for attribute-anchored searches (DOB, occupation, hobby as anchor) — use find_all_by_attribute instead.

        Tips:
        - Search a person's FULL NAME directly (e.g., "Alice Brown") to retrieve their profile
        - Add "family", "siblings", "children", or "friends" for relationship queries
        - If the first query returns wrong results, try the name alone or a different phrasing
        """
        results = self.retrieve(query)
        return "\n\n---\n\n".join(results.passages)

    def find_all_by_attribute(self, attribute_type: str, attribute_value: str) -> str:
        """Find ALL people who share a given attribute value.

        ALWAYS use this tool (instead of search_wiki) when the anchor is an attribute value
        rather than a person's name — e.g., when searching by date of birth, occupation, or hobby.

        This tool issues multiple differently-phrased queries and returns ALL unique matching pages,
        so you get the full set of people with that attribute (not just the first match).

        Args:
            attribute_type: The type of attribute, e.g., 'dob', 'occupation', 'hobby'
            attribute_value: The exact value, e.g., '0918-01-17', 'financial controller', 'archery'
        """
        atype = attribute_type.lower().strip()
        aval = attribute_value.strip()

        # Validate attribute type — only supported for DOB, occupation, hobby, nationality
        SUPPORTED_TYPES = {
            'dob', 'date of birth', 'birth date', 'born', 'birthday',
            'occupation', 'job', 'career', 'profession',
            'hobby', 'hobbies', 'interest',
            'nationality', 'citizenship', 'country',
        }
        if atype not in SUPPORTED_TYPES:
            return (
                f"[Error: find_all_by_attribute only supports DOB, occupation, hobby, or nationality lookups. "
                f"You provided attribute_type='{attribute_type}', which is not supported. "
                f"Use search_wiki instead for name/relationship lookups.]"
            )

        # Build 3 diverse query phrasings
        if atype in ('dob', 'date of birth', 'birth date', 'born', 'birthday'):
            queries = [
                f"date of birth {aval}",
                f"born {aval}",
                f"{aval}",
            ]
        elif atype in ('occupation', 'job', 'career', 'profession'):
            queries = [
                f"occupation {aval}",
                f"{aval}",
                f"job {aval}",
            ]
        elif atype in ('hobby', 'hobbies', 'interest'):
            queries = [
                f"hobby {aval}",
                f"{aval}",
                f"hobbies {aval}",
            ]
        else:
            queries = [
                f"{atype} {aval}",
                f"{aval}",
                f"{aval} {atype}",
            ]

        all_passages = []
        seen_passages = set()
        for q in queries:
            results = self.retrieve(q)
            for p in results.passages:
                if p not in seen_passages:
                    seen_passages.add(p)
                    all_passages.append(p)

        header = f"[find_all_by_attribute: {len(all_passages)} unique pages found for {attribute_type}={attribute_value!r} across {len(queries)} queries]\n\n"
        return header + "\n\n---\n\n".join(all_passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
