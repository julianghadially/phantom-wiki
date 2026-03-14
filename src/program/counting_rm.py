import dspy


class CountingRM(dspy.Retrieve):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.call_count = 0

    def forward(self, query_or_queries, k=None, **kwargs):
        self.call_count += 1
        return self.rm(query_or_queries, k=k, **kwargs)

    def reset_count(self):
        self.call_count = 0
