# Overview

## PhantomWiki
Paper: https://arxiv.org/pdf/2502.20377
Phantom wiki is an AI system benchmark for question answering with multi-branch reasoning and multi-hop retrieval

Phantom Wiki creates a universe of fictional characters to evaluate reasoning and retrieval capabilities of language models and language model systems.

The benchmark is resilient against leakage because the facts are entirely fictional and random.

## CodeEvolver
CodeEvolver optimizes one program at a time, by starting with the initial program and making changes to the prompts and the code (including context pipeline, tooling, AI modules, AI module graph, etc.). 

In changing the system and code, CodeEvolver fundamentally modifies the resource consumption resulting from changing the number of AI modules called, and the services used. However, CodeEvolver does control for unfair resource additions. For example, the number of hops allowed in the multi hop benchmarks is kept constant. See controls by program, below. 

## CodeEvolver
CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric.

This combines several mechanisms:
- **Optimizer algorithm:** GEPA is a reflective language model algorithm that makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer. 
- **Git branching:** A git process manages evolving code across many git worktrees  
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc. 

CodeEvolver and the optimizer lives in its own separate repository. 
CodeEvolver repository: https://github.com/julianghadially/CodeEvolver
CodeEvolver requirements: github repo with module path, metric path, and dataset. No main function required. 

Users connect their code with the CodeEvolver GitHub app, which allows CodeEvolver to add and run code in new branches.

## Programs

### Baseline 1: Chain of thought, multi-hop RAG 
This program uses a multi-hop RAG agent that is allowed to make five hops, sequentially. 

This baseline program is based on the `HotpotMultiHop` program from LangProbe:

```python
import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")


class HotpotMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Adapted from HoverMultiHop. Hop 3 replaced with answer generation."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
```
### Baseline 2: ReAct agent
This agent has a tool calling capability where it can perform iterative retrieval using dspy.react. It is based entirely on the react agent from PhantomWiki, except re-implemented in dspy. See @phantomwiki_analysis.md   

### CodeEvolver Programs
CodeEvolver will modify the ReAct agent as a baseline.

#### What's Allowed
- The program is allowed to create or remove modules, dynamic prompts, tool calls, reasoning steps, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.RLM for managing large context reasoning, dspy.ChainOfThought, dspy.Predict, etc.)
- There is no limit on the number of search results to display per query or the number of searches to make

#### Constraints:
- Do not change the retriever, as this is outside of the program

#### Ideas
- Try aggregating all relevant context (high page retrieval value and high query count) and processing it with dspy.RLM.
- Try adding reasoning steps and/or structured thinking and/or logic guidance before providing an answer.
- Try formal reasoning / gap analyses / entity mapping in between iterative search steps.
- Try creating a secondary workspace to jot down persistent reasoning logic that agents can add to or remove as they interact with more documents
- Try iterative search methods.
- Try increasing the maximum number of retrieval steps
- Try modifying the number of documents returned per query.