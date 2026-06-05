import dspy


class ExhaustiveInvestigationSignature(dspy.Signature):
    """You are investigating a question about fictional characters in PhantomWiki.

    CRITICAL: Many questions have MULTIPLE correct answers. You must find ALL of them.

    STEP 1 — IDENTIFY AND COLLECT ALL ANCHORS:
    - Named anchor (e.g., "Deon Gall"): use search_wiki with the person's name.
    - Attribute anchor (date of birth, occupation, hobby, etc.): ALWAYS use find_all_by_attribute — NOT search_wiki. Multiple people share the same attribute. Collect ALL names from the results before investigating any chain.
    - FORBIDDEN: stopping after finding the first person. Investigate EVERY anchor entity found.
    - If an anchor chain leads to a dead end (e.g., no spouse found), DON'T give up — there are likely OTHER anchors. Use find_all_by_attribute again with a different phrasing to find more.

    STEP 2 — INVESTIGATE EACH ANCHOR SYSTEMATICALLY:
    1. For EACH anchor, follow the relationship chain step by step.
    2. When a page lists multiple relatives (children, siblings, friends), write down ALL names and search EACH one individually.
    3. Track which branches you have already investigated vs. which still remain.
    4. Do NOT stop after finding 1-2 answers if the question asks for all members of a set.

    STEP 3 — RELATIONSHIP SEMANTICS (read carefully):
    - sister-in-law: your SPOUSE'S sister, OR your SIBLING'S wife — NOT your own sister
    - mother-in-law: your SPOUSE'S mother
    - grandmother/grandparent: your PARENT'S parent (TWO hops up — not one)
    - great-aunt: your GRANDPARENT'S sister (three hops: you → parent → grandparent → sibling)
    - great-grandchild: child of your grandchild (three hops down)
    - second cousin: child of your parent's cousin
    These are different from their one-hop counterparts — verify the hop count carefully.

    STEP 4 — COUNTING FORMAT:
    - "How many X does each Y have?" → return EACH count separately, not summed.
    - Example: person A has 2, person B has 5 → return ["2", "5"] NOT ["7"].
    - For population queries (e.g., "how many friends does each biochemist have?"): use find_all_by_attribute to find multiple people, count for each, and return the DISTINCT set of counts observed.

    COMPLETENESS CHECK before finalizing:
    - Did I use find_all_by_attribute for ALL attribute-anchored lookups?
    - Did I investigate EVERY anchor entity found (not just the first)?
    - Did I search ALL entities at each intermediate level (all children, all siblings, all friends)?
    - Am I confident in the relationship semantics for each hop?
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
