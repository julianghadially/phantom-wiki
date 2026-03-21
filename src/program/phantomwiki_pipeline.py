import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


class FindMoreAnswers(dspy.Signature):
    """Search retrieved passages for additional answers to the question beyond what has already been found.
    Focus on completeness - find ALL entities/values satisfying the question criteria."""
    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Retrieved passages to search through")
    answers_so_far: list[str] = dspy.InputField(desc="Answers already collected")
    new_answers: list[str] = dspy.OutputField(desc="New answers found in passages, not already in answers_so_far")
    followup_query: str = dspy.OutputField(desc="Query to search for more answers, or empty string if none needed")


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.find_more = dspy.ChainOfThought(FindMoreAnswers)
        self.retrieve = dspy.Retrieve(k=10)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            initial = self.program(question=question)
            all_answers = list(initial.answer) if initial.answer else []
            query = question
            for _ in range(5):
                passages = "\n\n".join(self.retrieve(query).passages)
                result = self.find_more(
                    question=question,
                    passages=passages,
                    answers_so_far=all_answers,
                )
                new = [a for a in result.new_answers if a not in all_answers]
                all_answers.extend(new)
                if not result.followup_query or not new:
                    break
                query = result.followup_query
            return dspy.Prediction(answer=all_answers)
