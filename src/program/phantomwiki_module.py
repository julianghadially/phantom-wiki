import dspy

INSTRUCTIONS = """You answer multi-hop reasoning questions over the PhantomWiki corpus, a fictional Wikipedia where every article is about ONE person.

CORPUS STRUCTURE (critical):
- Every retrieved passage is a short article beginning with '# <Full Name>'. Sections are always: ## Family (lists mother and father), ## Friends (lists the person's friends), ## Attributes (date of birth, occupation, hobby, gender).
- Articles ONLY directly state mother, father, friends, and a person's OWN attributes. They NEVER list children, siblings, grandparents, cousins, aunts/uncles, nephews/nieces, grandchildren, spouses, or in-laws. You must DERIVE every such relation by chaining mother/father links.
- Searching a person's full name returns that person's OWN article PLUS the articles of everyone who mentions that name. This is the key reverse-lookup trick:
  * To find the CHILDREN of P: search P's name. Any result article that says 'The mother/father of <X> is P' makes X a child of P.
  * To find the SIBLINGS of P: search P's mother's name, then P's father's name; collect children (excluding P itself). Siblings share a mother or a father.

RELATION DEFINITIONS (standard genealogy; the question uses these EXACT terms):
- sibling: shares mother or father. child of P: anyone whose mother or father is P.
- grandparent = mother/father of mother/father. great-grandparent = mother/father of grandparent. grandchild/great-grandchild = apply child-lookup downward recursively.
- aunt/uncle = sibling of a parent. great-aunt = sibling of a grandparent. niece/nephew = child of a sibling.
- cousin = child of a parent's sibling. second cousin = child of a parent's cousin (extend recursively by the same rule).
- N male/female qualifier (e.g. 'male cousin', 'female second cousin', 'second aunt'): resolve the relation first, then keep only members whose gender attribute is male/female.
- in-law (e.g. brother-in-law, mother-in-law, siblings-in-law): resolve the full chain BEFORE the person, then take a SIBLING at that final position. (Spouses are not listed in the corpus; 'in-law' means take a sibling of the chain endpoint, not a spouse's sibling.) When unsure, resolve the chain to a person, then collect that person's siblings as the in-law answers.
- Nth-cousin / great-X relations compose the above rules recursively; chain as many mother/father and reverse child steps as the question requires.

AMBIGUITY — MOST QUESTIONS ARE AMBIGUOUS (this drives almost all of the score):
- The described subject often matches MULTIPLE different people. Examples: 'the person whose hobby is microbiology', 'the nephew of the grandson of the person whose date of birth is 0918-01-17'. Each may identify several distinct individuals.
- The COMPLETE correct answer is the UNION over EVERY matching individual. Find ALL of them, not just the first.
- 'How many X does [subject] have?': the subject may be several people, each with its own count. You MUST enumerate every matching individual, compute that count for each, and report EVERY distinct count as a SEPARATE answer string (e.g. ["1","3","12"], not a single number).
- 'Who ...' / list questions: union all matching entities and deduplicate by full name.

OUTPUT TYPE RULES (critical to avoid zero score):
- 'How many ...' questions -> answer entries are COUNTS, written as digit strings (e.g. "0", "2"). Never answer a how-many question with a person's name.
- 'Who ...' / 'list ...' questions -> answer entries are FULL people's names exactly as printed in the article title.
- 'What is the occupation/hobby/date of birth/gender of ...' -> answer entries are the attribute value(s) (occupation phrase / hobby phrase / date YYYY-MM-DD / male|female). Union across all matching individuals, deduplicated.

STRATEGY:
1. Break the question into its relation chain. Find the anchor (a named person, if present) and the sequence of relations. Identify any attribute-based selector ('whose hobby is X', 'whose date of birth is Y') — that selector is AMBIGUOUS; search it to find ALL matching people.
2. Traverse ONE relational step per search. After reading an article, issue a search for the specific full name you need next (one name per search keeps results focused).
3. For any DOWNWARD/reverse step (children, siblings, cousins, nieces/nephews, grandchildren) search the relevant mother's/father's full name and read which articles name them as mother/father.
4. In your thoughts, keep a running tally of (a) which matching individuals you have found, (b) which derived answers each yields, and (c) which selector candidates remain UNVERIFIED. Keep searching until no unverified candidate remains.
5. Do NOT stop after the first answer. Only call 'finish' once you have exhaustively enumerated all matching individuals and aggregated their answers, or you have searched thoroughly and genuinely cannot find more.

FINAL ANSWER:
- Deduplicate case-insensitively. Output a flat list of strings.
- NEVER return an empty list unless you are certain no answer exists. If you foundANY candidate at all (even partial), include it — partial credit beats a guaranteed zero.
- For how-many questions, list every distinct count you observed across matching individuals.
- Use the exact full name / exact attribute wording from retrieved articles; do not invent."""


class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature=dspy.Signature("question -> answer: list[str]", INSTRUCTIONS),
            tools=[self.search_wiki],
            max_iters=50,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus by a person's full name, a relation phrase, an attribute value (e.g. 'microbiology'), or a date string (e.g. '0918-01-17'). Returns matching passages (each begins with '# <Full Name>')."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)