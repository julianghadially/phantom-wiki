import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class QuestionDecomposer(dspy.Signature):
    """Analyze the multi-hop question and decompose it into 2–4 independent investigation branches.
    Each branch should be a self-contained sub-question that targets a distinct chain of logic
    (e.g., a different intermediate entity, a different ancestor path, or a different candidate
    starting node). For questions with a single unambiguous answer path, return just 1 branch.
    Always include at least one branch that performs a broad exhaustive search to catch answers
    missed by focused branches."""
    question: str = dspy.InputField()
    branches: list[str] = dspy.OutputField(desc="2–4 focused sub-questions, each covering a distinct investigation path through the knowledge graph")


class AnswerSynthesizer(dspy.Signature):
    """Combine partial answer lists gathered from multiple investigation branches into a single
    deduplicated, correctly formatted answer list. Remove any error messages or 'cannot determine'
    strings. For count-aggregation questions (e.g., 'how many X does each person with property Y
    have?'), return only unique numeric count values as bare strings (e.g. ['0','2','4']), not
    person-keyed pairs. For entity-lookup questions, return bare entity names."""
    question: str = dspy.InputField()
    partial_answers: list[str] = dspy.InputField(desc="All answers collected across all branches, may contain duplicates or noise")
    answer: list[str] = dspy.OutputField(desc="Final deduplicated answer list in correct format")


class FocusedInvestigator(dspy.Module):
    """A focused single-branch ReAct agent with its own retriever and small iteration budget."""
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.react = dspy.ReAct(
            signature="sub_question -> answer: list[str]",
            tools=[self.search_wiki],
            max_iters=15,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, sub_question: str):
        result = self.react(sub_question=sub_question)
        return result


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()           # kept as fallback & first-pass
        self.decomposer = dspy.ChainOfThought(QuestionDecomposer)
        self.investigator = FocusedInvestigator()   # reused sequentially per branch
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesizer)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Phase 1: Decompose question into independent branches (up to 4)
            plan = self.decomposer(question=question)
            branches = plan.branches[:4]  # hard cap at 4 to avoid overload

            # Phase 2: Investigate each branch sequentially, accumulate all answers
            all_partial = []
            for branch_q in branches:
                try:
                    result = self.investigator(sub_question=branch_q)
                    if isinstance(result.answer, list):
                        all_partial.extend(result.answer)
                    else:
                        all_partial.append(str(result.answer))
                except Exception:
                    pass  # skip failed branches; remaining branches may still yield answers

            # Phase 3: Synthesize into final deduplicated answer
            if not all_partial:
                # Fallback to original single-agent if all branches failed
                return self.program(question=question)

            final = self.synthesizer(question=question, partial_answers=all_partial)
            return dspy.Prediction(answer=final.answer)
