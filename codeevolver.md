PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the PhantomWiki corpus. Given a natural-language question, it iteratively retrieves relevant passages and reasons over them to produce one or more answers. It is optimized using the GEPA framework (DSPy) with an F1-based feedback metric.

### Key Modules

**`PhantomWikiReActPipeline`** (entry point, `src/program/phantomwiki_pipeline.py`)
Top-level `dspy.Module` that wires together retrieval and reasoning. On initialization it creates a `CountingRM`-wrapped `ColBERTv2` retriever (pointed at a hosted Modal endpoint), a `PhantomWikiReAct` reasoning module, a `dspy.ChainOfThought(FindMoreAnswers)` sub-module, and a `dspy.Retrieve(k=10)` retriever. The `forward` method sets the wrapped retriever as the active DSPy context retriever, runs the initial ReAct program, then performs up to 5 additional retrieval-and-extraction rounds using `FindMoreAnswers` to accumulate any additional answers. Early stopping occurs when no new answers are found or no follow-up query is suggested. Returns `dspy.Prediction(answer=all_answers)` with the fully accumulated list.

**`FindMoreAnswers`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
A `dspy.Signature` that takes `question`, `passages`, and `answers_so_far` as inputs and outputs `new_answers` (additional answers found in passages) and `followup_query` (the next query to run, or empty string when done). Used by `PhantomWikiReActPipeline` in its iterative answer-accumulation loop.

**`CountingRM`** (`src/program/counting_rm.py`)
A thin instrumentation wrapper around any DSPy retriever. It proxies all retrieval calls to the underlying model while incrementing a `call_count` counter, enabling retrieval-usage diagnostics without altering retrieval behavior.

**`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`)
Core reasoning module built on `dspy.ReAct`. Uses the signature `question -> answer: list[str]` with up to 50 reasoning iterations. Its single registered tool, `search_wiki`, invokes `dspy.Retrieve(k=7)` to fetch the top-7 passages from the PhantomWiki corpus for a given query and returns them as a newline-separated string for the LLM to reason over.

### Data Flow
1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward`.
2. **Context setup**: `CountingRM(ColBERTv2)` is injected as the active retriever via `dspy.context`.
3. **ReAct loop**: `PhantomWikiReAct` runs up to 50 think-act cycles. Each `search_wiki` action queries ColBERTv2 through `CountingRM`, retrieving 7 passages per call.
4. **Answer accumulation**: After the ReAct loop, up to 5 additional rounds retrieve 10 passages per round using `dspy.Retrieve(k=10)`. Each round calls `FindMoreAnswers` (via `dspy.ChainOfThought`) with the current passages and accumulated answers to extract new answers and a follow-up query. Loop exits early if no new answers or no follow-up query is produced.
5. **Output**: The fully accumulated answer list is returned as `dspy.Prediction(answer=all_answers)`.

### Metric Being Optimized
`phantomwiki_f1_feedback` computes token-set F1 between the predicted answer list and the gold answer list (both normalized to lowercase). It returns a `ScoreWithFeedback` object containing the numeric F1 score (0–1) and a detailed textual breakdown of correct, missed, and extra answers, enabling GEPA's feedback-driven prompt optimization.

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

