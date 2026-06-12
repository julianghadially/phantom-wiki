import re

import dspy

_HOW_MANY_RE = re.compile(r'\bhow\s+many\b', re.I)
_PROPER_NAME_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_SKIP_NAME_PARTS = {
    'What', 'Who', 'Where', 'When', 'Why', 'How', 'The', 'Is', 'Are', 'Was',
    'Were', 'Does', 'Did', 'Has', 'Have', 'Can', 'Could', 'Should', 'Would',
    'May', 'Might', 'Will', 'Shall', 'PhantomWiki', 'Wikipedia', 'And', 'Or',
    'Of', 'In', 'On', 'At', 'To', 'For', 'With', 'By', 'From',
}


class PhantomWikiSignature(dspy.Signature):
    """You are a thorough research assistant for a people database (PhantomWiki).

    CRITICAL: Questions like "What is X of the person whose Y is Z?" may have MULTIPLE correct
    answers because several people in the database can share the same property Z. You MUST find ALL of them.

    Follow this process every time:
    1. Identify the filter condition in the question (e.g., "occupation is video editor", "born on date X", "hobby is Y")
    2. Search for ALL people matching that condition — not just the first result returned
    3. For each matching person, follow the full reasoning chain to find their specific answer
    4. Collect ALL answers across all matching people
    5. Return every answer found as a list

    Critical rules:
    - After finding one matching entity, ALWAYS search for additional people who also satisfy the same condition
    - When search results show multiple candidates matching the condition, investigate EACH one separately
    - Never assume the "closest" result is correct when an exact match isn't found — try different search queries instead
    - Return only the answer values (names, numbers, dates, hobbies, occupations), never explanations
    - If genuinely no information can be found after thorough searching, return an empty list
    """
    question: str = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="All valid answers — one entry per matching entity found")


class PhantomWikiEntitySignature(dspy.Signature):
    """You are a focused research assistant for a people database (PhantomWiki).
    You are investigating ONE SPECIFIC ENTITY to answer a question about them.

    Your job:
    1. Search for the given entity to retrieve their profile
    2. Determine if the entity satisfies the filter condition in the question
       (e.g., does their birth date match? does their hobby match?)
    3. If YES: follow the chain of relationships to find the final answer value for that entity
    4. If NO (entity does not satisfy the filter): return an empty list immediately

    Rules:
    - Always start by searching for the entity by name
    - Check the entity's attributes carefully against the question's filter condition
    - For "how many" questions: COUNT and return the NUMBER as a string (e.g., "3"), NOT the entity names
    - Return ONLY the final answer value(s) for this one entity
    - Return an empty list if the entity does not satisfy the filter condition
    - Do not mix answers from different entities
    """
    question: str = dspy.InputField()
    entity: str = dspy.InputField(desc="The specific entity to investigate")
    answer: list[str] = dspy.OutputField(desc="Answer(s) for this entity only, or empty list if entity does not match the filter")


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature=PhantomWikiSignature,
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)


class PhantomWikiEntityAgent(dspy.Module):
    """Lightweight per-entity ReAct agent — investigates a single candidate entity."""
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=7)
        self.react = dspy.ReAct(
            signature=PhantomWikiEntitySignature,
            tools=[self.search_wiki],
            max_iters=15,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question: str, entity: str):
        result = self.react(question=question, entity=entity)
        return dspy.Prediction(answer=result.answer or [])
