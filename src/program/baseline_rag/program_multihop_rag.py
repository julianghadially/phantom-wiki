import dspy


class PhantomWikiMultiHop(dspy.Module):
    def __init__(self):
        self.k = 7
        self.retrieve = dspy.Retrieve(k=self.k)
        self.create_query_hop2 = dspy.ChainOfThought("question, summary_1 -> query")
        self.summarize1 = dspy.ChainOfThought("question, passages -> summary")
        self.summarize2 = dspy.ChainOfThought("question, context, passages -> summary")
        self.generate_answer = dspy.ChainOfThought(
            "question, summary_1, summary_2 -> answer: list[str]"
        )

    def forward(self, question):
        # Hop 1: retrieve on raw question
        hop1_docs = self.retrieve(question).passages
        summary_1 = self.summarize1(question=question, passages=hop1_docs).summary

        # Hop 2: generate refined query, retrieve again
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # Answer
        return dspy.Prediction(
            answer=self.generate_answer(
                question=question, summary_1=summary_1, summary_2=summary_2
            ).answer
        )
