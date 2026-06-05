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
    5. NEVER repeat the same search query — if a query returns no useful result, try a DIFFERENT phrasing or search a DIFFERENT person's name.

    STEP 3 — RELATIONSHIP SEMANTICS AND GENERATION COUNTING:
    - aunt: your PARENT's sister
    - uncle: your PARENT's brother
    - cousin (first cousin): child of your PARENT's sibling (aunt/uncle's child — NOT all descendants of the same ancestor, NOT all relatives of the same generation)
    - great-aunt / second aunt: your GRANDPARENT'S sister (3 hops total: you→parent→grandparent→their sibling)
    - great-uncle / second uncle: your GRANDPARENT's brother (your parent's uncle — go up 2 hops, then sibling)
    - sister-in-law: your SPOUSE'S sister, OR your SIBLING'S wife — NOT your own sister
    - mother-in-law / father-in-law: your SPOUSE'S mother / father
    - grandparent / grandfather / grandmother: your PARENT'S parent (exactly 2 hops up)
    - great-grandfather / great-grandparent: your GRANDPARENT'S parent (exactly 3 hops up — NOT the same as grandfather which is 2 hops)
    - great-grandchild: child of your grandchild (3 hops down — must descend through grandchild first)
    - second cousin: child of your parent's first cousin
    GENERATION RULE: Count hops carefully. grandfather = 2 hops up. great-grandfather = 3 hops up. If you ascend N hops to reach an ancestor, great-grandchildren of that ancestor are N hops BELOW THAT ANCESTOR (same generation as the original person). Once you have determined the correct hop count, COMMIT to it — do not add extra hops mid-chain.

    STEP 4 — COUNTING FORMAT (AGGREGATION QUESTIONS):
    - "How many X does each Y have?" questions require: (1) find ALL members of the Y population COMPLETELY before stopping, (2) count X for EACH member individually, (3) return ONLY the DISTINCT unique count values.
    - IMPORTANT: Do NOT stop after finding a few distinct values — exhaust the FULL population first, then collect distinct values at the end.
    - Example: if 8 people have counts [1, 0, 1, 3, 0, 2, 3, 1] → return ["0", "1", "2", "3"], NOT ["1","0","1","3","0","2","3","1"] (raw list), NOT ["7"] (sum).
    - Apply gender filters (male/female) precisely when the question specifies gender — only count qualifying relatives.

    COMPLETENESS CHECK before finalizing:
    - Did I use find_all_by_attribute for ALL DOB/occupation/hobby attribute lookups?
    - Did I investigate EVERY anchor entity found (not just the first)?
    - Did I search ALL entities at each intermediate level (all children, all siblings, all friends)?
    - Did I count generations correctly — am I at the right level of the family tree?
    - For counting/aggregation questions: did I DEDUPLICATE — return DISTINCT count values, NOT raw per-person counts?
    """

    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="Complete list of ALL correct answers. Be exhaustive—include every valid answer found, not just the first few. For aggregation/counting questions ('how many X does each Y have?'), return ONLY the DISTINCT unique count values as separate strings — NOT raw per-person counts. Return [] only if the entity truly does not exist."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=15)
        # Larger k for DOB queries to improve coverage of people sharing the same birth date.
        # After retrieval, results are filtered to exact DOB matches to reduce noise.
        self.dob_retrieve = dspy.Retrieve(k=30)
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
        is_dob = atype in ('dob', 'date of birth', 'birth date', 'born', 'birthday')
        if is_dob:
            queries = [
                f"date of birth {aval}",
                f"born {aval}",
                f"{aval}",
            ]
            retrieve_fn = self.dob_retrieve  # use larger k for DOB to improve coverage
        elif atype in ('occupation', 'job', 'career', 'profession'):
            queries = [
                f"occupation {aval}",
                f"{aval}",
                f"job {aval}",
            ]
            retrieve_fn = self.retrieve
        elif atype in ('hobby', 'hobbies', 'interest'):
            queries = [
                f"hobby {aval}",
                f"{aval}",
                f"hobbies {aval}",
            ]
            retrieve_fn = self.retrieve
        else:
            queries = [
                f"{atype} {aval}",
                f"{aval}",
                f"{aval} {atype}",
            ]
            retrieve_fn = self.retrieve

        all_passages = []
        seen_passages = set()
        for q in queries:
            results = retrieve_fn(q)
            for p in results.passages:
                if p not in seen_passages:
                    seen_passages.add(p)
                    all_passages.append(p)

        # For DOB queries: filter to only pages that contain the exact DOB string,
        # since ColBERT returns semantically similar dates (e.g., 0945-07-12 for query 0945-06-12).
        if is_dob:
            exact_matches = [p for p in all_passages if aval in p]
            if exact_matches:
                header = (
                    f"[find_all_by_attribute: {len(exact_matches)} exact DOB matches found "
                    f"for {attribute_type}={attribute_value!r} (filtered from {len(all_passages)} retrieved pages)]\n\n"
                )
                return header + "\n\n---\n\n".join(exact_matches)
            else:
                header = (
                    f"[find_all_by_attribute: WARNING — 0 exact matches for DOB={attribute_value!r} "
                    f"found in {len(all_passages)} retrieved pages. The pages below have similar but NOT exact DOBs — "
                    f"verify carefully before using any as an anchor.]\n\n"
                )
                return header + "\n\n---\n\n".join(all_passages)

        header = f"[find_all_by_attribute: {len(all_passages)} unique pages found for {attribute_type}={attribute_value!r} across {len(queries)} queries]\n\n"
        return header + "\n\n---\n\n".join(all_passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
