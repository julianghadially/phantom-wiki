import dspy


class PhantomWikiMainSignature(dspy.Signature):
    """You are a research assistant answering questions about a fictional wiki universe with many characters.

    CRITICAL: Most questions have MULTIPLE correct answers. Your job is to find ALL of them.

    Core strategy:
    1. NEVER assume a question has only one answer. Multiple people can share birthdates, hobbies, occupations, and relationships.
    2. After finding the FIRST answer, ALWAYS keep searching for MORE:
       - Search "other people born on [DATE]"
       - Search "other people whose [PROPERTY] is [VALUE]"
       - For family questions, find ALL relatives (all grandchildren, ALL nephews, ALL cousins)
    3. For multi-hop chains (e.g., "great-grandchild of the great-grandfather of X"):
       - First find the correct ancestor (verify the generation count carefully)
       - Then enumerate ALL descendants at the target generation across ALL branches
    4. For birthdate/property questions: multiple people WILL share the property; find them ALL
    5. Search exhaustively with varied queries before concluding you have all answers
    6. A single-item answer list is almost always incomplete - verify with additional searches

    Common failure to AVOID: Finding one answer and immediately calling finish. Always search for more.
    """
    question: str = dspy.InputField(desc="The question to answer - almost always has multiple correct answers")
    answer: list[str] = dspy.OutputField(
        desc="ALL answers satisfying the question. Be exhaustive - every person, hobby, occupation, or value that qualifies. A single-item list likely means you stopped too early."
    )


class PhantomWikiExhaustiveSignature(dspy.Signature):
    """You are performing a SECOND search pass to find answers that were missed in the first pass.

    Your task: Given the question and the initial_answers already found, search specifically for
    ADDITIONAL answers that were NOT found yet. Focus on unexplored branches and alternative entities.

    Search strategies for finding REMAINING answers:
    1. For birthdate questions: search "born [YEAR]-[MONTH]" variations to find more people with same date
    2. For property questions: search "[PROPERTY] [VALUE]" with different phrasings
    3. For family questions: explore siblings of already-found intermediate nodes; check other family branches
    4. Try searching by name variations of entities found in the first pass to find their relatives
    5. For each found person, search for "family of [NAME]" or "[NAME] children/siblings/relatives"

    Return initial_answers PLUS all new answers you discover. Do not drop any answers from initial_answers.
    """
    question: str = dspy.InputField(desc="The original question")
    initial_answers: str = dspy.InputField(desc="Answers already found in the first search pass - do not repeat searching for these")
    answer: list[str] = dspy.OutputField(
        desc="Complete answer list: all of initial_answers PLUS any new answers discovered in this second pass"
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react_main = dspy.ReAct(
            signature=PhantomWikiMainSignature,
            tools=[self.search_wiki],
            max_iters=40,
        )
        self.react_exhaustive = dspy.ReAct(
            signature=PhantomWikiExhaustiveSignature,
            tools=[self.search_wiki],
            max_iters=15,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 10 relevant passages.

        Tips for exhaustive search:
        - After finding one person, search for OTHERS with the same property
        - Use variations: "born on DATE", "date of birth DATE", "DOB DATE"
        - For relationships: search both directions (find X's parent, then find ALL children of that parent)
        - Search family members by name to find their articles and discover more relatives
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        # Phase 1: Main reasoning - find initial answers
        main_result = self.react_main(question=question)
        initial_answers = main_result.answer if isinstance(main_result.answer, list) else [str(main_result.answer)]

        # Phase 2: Exhaustive search for remaining answers
        initial_str = ", ".join(str(a) for a in initial_answers) if initial_answers else "none found yet"
        exhaustive_result = self.react_exhaustive(
            question=question,
            initial_answers=initial_str,
        )
        exhaustive_answers = exhaustive_result.answer if isinstance(exhaustive_result.answer, list) else [str(exhaustive_result.answer)]

        # Combine answers: initial + new ones from second pass (deduplicated, preserving order)
        seen = set()
        combined = []
        for ans in initial_answers + exhaustive_answers:
            key = str(ans).strip().lower()
            if key not in seen and key:
                seen.add(key)
                combined.append(ans)

        return dspy.Prediction(answer=combined)
