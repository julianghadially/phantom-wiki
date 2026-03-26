PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "ReAct + Targeted Answer-Completion Loop with IdentifyMissingEntityQueries & ExtractTargetAttribute"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs a two-phase strategy: (1) an initial ReAct loop that interleaves reasoning with retrieval to find the primary answer, followed by (2) a targeted answer-completion loop that systematically retrieves missing answers by iterating over unchecked entity paths identified by a lightweight ChainOfThought gap-finder.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model, the core ReAct program, and three new sub-modules for the completion loop (`gap_finder`, `attr_extractor`, `completion_retrieve`). The `forward()` method runs the initial ReAct pass, then drives the targeted completion loop before returning a deduplicated answer list.

- **`IdentifyMissingEntityQueries`** (`src/program/phantomwiki_pipeline.py`): DSPy Signature used by `gap_finder` (a `ChainOfThought` module). Given the question, initial answers, and the ReAct reasoning trace, it determines whether additional entity paths were missed and generates up to 10 targeted search queries for unchecked entities, plus the target attribute to extract.

- **`ExtractTargetAttribute`** (`src/program/phantomwiki_pipeline.py`): DSPy Signature used by `attr_extractor` (a `ChainOfThought` module). Given retrieved passages for a specific entity query, it extracts only new attribute values not already in the existing answer list.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with the signature `question -> answer: list[str]` and a single `search_wiki` tool. The ReAct agent iterates up to 50 steps, issuing search queries and reasoning over retrieved passages (top-7 per query) before emitting a list of answer strings. Now also returns the `reasoning` trace for downstream use.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. `PhantomWikiReAct.forward` feeds the question into `dspy.ReAct`, which calls `search_wiki(query)` one or more times (up to 50 iterations). Returns `answer` and `reasoning`.
4. Each `search_wiki` call uses `dspy.Retrieve(k=7)` to fetch passages from the PhantomWiki ColBERT index.
5. The ReAct loop reasons over accumulated passages and terminates with `answer: list[str]` and a `reasoning` trace.
6. The `gap_finder` ChainOfThought analyzes the question, initial answers, and reasoning trace to identify unchecked entity paths and generate targeted queries.
7. For each of up to 10 targeted queries, `completion_retrieve` (k=7) fetches passages, then `attr_extractor` extracts new answer values not already found.
8. All answers are deduplicated (case-insensitive, order-preserving) and returned as `dspy.Prediction(answer=deduped)`.

## ARCHITECTURE DESCRIPTION:
PhantomWikiReActPipeline implements a two-phase answer-completion strategy for multi-hop QA over the PhantomWiki corpus. In Phase 1, a ReAct agent (up to 50 iterations, top-7 retrieval per query) reasons and searches to produce an initial answer list along with its full reasoning trace. In Phase 2, the pipeline runs a targeted completion loop: the IdentifyMissingEntityQueries signature (via ChainOfThought) analyzes the question, initial answers, and reasoning trace to determine whether additional entity paths remain unchecked — e.g., all siblings, all persons with a given hobby — and generates up to 10 specific search queries. For each query, direct retrieval (k=7) fetches passages, and the ExtractTargetAttribute signature (via ChainOfThought) extracts only new attribute values not already in the answer list. This directly addresses the systematic multi-entity enumeration failure common in ReAct-only approaches, where the agent may find one valid answer chain but miss others. The completion phase uses lightweight direct retrieval + extraction rather than a full second ReAct loop, making it efficient. Final answers are deduplicated while preserving order and returned as a list.

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

