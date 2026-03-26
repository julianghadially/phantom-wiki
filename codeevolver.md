PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "ReAct with broad retrieval tool + deterministic Name—Value extractor"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs a ReAct (Reasoning + Acting) loop that interleaves language model reasoning steps with targeted retrieval actions, enabling it to gather evidence across multiple hops before producing a final answer. A lightweight deterministic post-processing step handles "Name — Value" formatted answers by stripping the name prefix, while preserving all answers intact in all other cases.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with the signature `question -> answer: list[str]` and two retrieval tools: `search_wiki` (top-7, precise) and `search_wiki_broad` (top-50, for enumerating all entities with a given property). The ReAct agent iterates up to 50 steps. After the ReAct loop, a deterministic `_extract_values` function handles "Name — Value" formatting: if any answer item contains " — ", the value after the separator is extracted from each item; otherwise all items are returned unchanged.

- **`_extract_values`** (`src/program/phantomwiki_module.py`): A pure Python helper function (no LLM calls). If any item in the answer list contains the " — " separator, it splits all items on " — " and returns the part after the separator. Otherwise, returns the items unchanged. This preserves all correct answers while still stripping name prefixes from "Name — Value" pairs.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. `PhantomWikiReAct.forward` feeds the question into `dspy.ReAct`, which calls `search_wiki(query)` or `search_wiki_broad(query)` one or more times (up to 50 iterations).
4. `search_wiki` uses `dspy.Retrieve(k=7)`; `search_wiki_broad` uses `dspy.Retrieve(k=50)`.
5. The ReAct loop reasons over accumulated passages and terminates with a raw `answer: list[str]`.
6. The raw answer is passed through `_extract_values()` to handle "Name — Value" formatting deterministically.
7. A `dspy.Prediction(answer=answer)` is returned upstream.

## ARCHITECTURE DESCRIPTION: PhantomWikiReActPipeline orchestrates multi-hop QA over a private PhantomWiki corpus using DSPy. The pipeline sets a ColBERTv2-backed CountingRM as the active retrieval model, then delegates to PhantomWikiReAct. That module runs a dspy.ReAct loop (up to 50 iterations) equipped with two tools: search_wiki (k=7, for targeted lookups) and search_wiki_broad (k=50, for enumerating all entities sharing a property like occupation or hobby). After the ReAct loop produces a raw list answer, a deterministic Python function _extract_values handles "Name — Value" formatting: if any answer item contains the " — " separator, it strips the name prefix from all items; otherwise all items are returned as-is. This approach preserves all correct answers found by ReAct (avoiding LLM-based extraction regressions that dropped answers) while still correctly handling the "Name — Value" format that ReAct sometimes produces.

### Metric Being Optimized
**`phantomwiki_f1_feedback`** computes token-set F1 between predicted and gold answer lists (after lowercasing/stripping), then returns a `ScoreWithFeedback` object containing the numeric F1 score (0–1) plus a natural-language feedback string detailing correct, missed, and extraneous answers. This feedback-augmented metric is designed for use with DSPy's GEPA optimizer, which leverages textual feedback to guide prompt refinement.

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

