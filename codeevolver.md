PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Two-Pass Sequential ReAct with FollowUpInvestigation and AnswerNormalizer Post-Processing

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. The top-level entry point, `PhantomWikiReActPipeline`, implements a two-pass sequential investigation strategy followed by answer normalization. The first pass runs `PhantomWikiReAct` (up to 30 iterations) to collect initial answers. The second pass runs a `FollowUpInvestigation` ReAct agent (up to 25 iterations) that explores alternative relationship chains and paths not yet investigated. The combined deduplicated answers are then cleaned by an `AnswerNormalizer` post-processing module before returning the final prediction.

`PhantomWikiReAct` implements the core ReAct agent pattern via `dspy.ReAct`, iteratively interleaving reasoning steps with calls to a `search_wiki` tool backed by the ColBERTv2 retriever. The `FollowUpInvestigation` signature directs the second-pass agent to enumerate answers reachable via unexplored paths. The `AnswerNormalizerSignature` via `dspy.ChainOfThought` cleans format artifacts such as 'Name: count' pairs and error/uncertainty strings. The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 with structured feedback for GEPA optimization.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization, it instantiates a `CountingRM` wrapping a `dspy.ColBERTv2` retriever, a `PhantomWikiReAct` module for the first pass, a `dspy.Retrieve(k=7)` for the follow-up search tool, a `dspy.ReAct` agent over the `FollowUpInvestigation` signature for the second pass (max_iters=25), and a `dspy.ChainOfThought(AnswerNormalizerSignature)` for post-processing. The `forward` method sets the active DSPy retrieval model (via `dspy.context(rm=self.rm)`), runs both passes sequentially, merges their answers via `dict.fromkeys` deduplication, then normalizes the combined list before returning.

**AnswerNormalizerSignature** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature used with `dspy.ChainOfThought` that takes `question` and `raw_answers: list[str]` and produces `normalized_answers: list[str]`. It strips 'Name: count' format artifacts (keeping only the unique numeric counts), removes error/uncertainty strings ('Cannot be determined', 'not found', etc.), and preserves clean atomic values unchanged. It falls back to the original list if normalization would produce an empty result.

**FollowUpInvestigation** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature that takes `question` and `already_found: list[str]` as inputs and produces `answer: list[str]`. Its docstring explicitly instructs the agent to explore alternative relationship chains and paths not yet investigated, treating `already_found` as a non-exhaustive partial result to maximize recall.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the primary ReAct agent. It holds a `dspy.Retrieve(k=7)` component and a `dspy.ReAct` agent with the signature `question -> answer: list[str]`, using `search_wiki` as its sole tool and allowing up to 30 agentic iterations.

**CountingRM** (`src/program/counting_rm.py`) is a lightweight `dspy.Retrieve` subclass that wraps any underlying retriever, incrementing a `call_count` on every retrieval invocation.

**Metric — phantomwiki_f1_feedback** (`src/metric/metric.py`) evaluates predictions with set-based F1 and returns `ScoreWithFeedback` for GEPA optimizer compatibility.

**Data flow**: A question enters `PhantomWikiReActPipeline.forward` → Pass 1: `PhantomWikiReAct` iteratively calls `search_wiki` (ColBERTv2, top-7 passages) and produces initial `list[str]` answers → Pass 2: `FollowUpInvestigation` ReAct explores unexplored paths using `_search_wiki`, given the first-pass answers as context → deduplicated combined answer list → `AnswerNormalizer` cleans format artifacts and error strings → evaluated with F1 and structured feedback.

## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response

