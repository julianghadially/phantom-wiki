import dspy


class PhantomWikiQA(dspy.Signature):
    """Answer questions about a fictional wiki universe. Provide accurate and complete answers.

    HOW TO ANSWER:

    1. READ ALL SEARCH RESULTS: When you receive search results, read every passage carefully.
       Multiple passages may contain different people matching the same condition (birthdate, hobby, etc.).
       Extract ALL matching entities from the returned passages, not just the first one you notice.

    2. MULTIPLE CORRECT ANSWERS: Many questions have multiple correct answers because:
       - Multiple people can share the same birthdate, hobby, or occupation
       - A person may have many relatives (many nephews, grandchildren, etc.)
       After reviewing initial search results, do 1-2 additional targeted searches to find further matches.
       Use queries like "other people born on [DATE]" or "[PERSON] siblings" to enumerate more.

    3. FAMILY TREE TRAVERSAL: For relationship questions (nephew, cousin, grandchild, etc.):
       - Find ALL intermediate entities at each hop, not just one
       - For each intermediate person, find ALL their relevant relatives
       - Explore every branch of the family tree systematically

    4. ANSWER FORMAT: Return only the requested values (names, dates, occupations, or numbers).
       Do NOT use "Person: value" or "Name: X" format. Just return the bare values.
       Example: return ["0", "2", "5"] not ["Alice: 0", "Bob: 2", "Carol: 5"]

    5. ONLY VERIFIED ANSWERS: Include only answers explicitly supported by search results.
       Do not guess or include entities you cannot directly verify from the wiki passages.
    """
    question: str = dspy.InputField(desc="The question to answer about the wiki universe")
    answer: list[str] = dspy.OutputField(
        desc="All verified answers found in search results. Return bare values only (no 'Person: value' format)."
    )


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=PhantomWikiQA,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns up to 10 relevant passages.

        Read ALL returned passages carefully - multiple passages may contain different
        people or entities that match the search condition.
        """
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
