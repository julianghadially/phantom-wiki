PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the PhantomWiki corpus. Given a natural-language question, it uses a two-phase Decompose-by-Reasoning-Type architecture to first exhaustively enumerate matching entities, then resolve any relational hops to produce final answers. It is optimized using the GEPA framework (DSPy) with an F1-based feedback metric.

### Key Modules

**`PhantomWikiReActPipeline`** (entry point, `src/program/phantomwiki_pipeline.py`)
Top-level `dspy.Module` that orchestrates the two-phase pipeline. On initialization it creates a `CountingRM`-wrapped `ColBERTv2` retriever (pointed at a hosted Modal endpoint), an `EntityEnumerator` module, and a `HopResolver` module. The `forward` method sets the wrapped retriever as the active DSPy context retriever, runs Phase 1 then Phase 2 sequentially, and returns a `dspy.Prediction` with the final answers.

**`EntityEnumerator`** (`src/program/phantomwiki_pipeline.py`)
Phase-1 module using `dspy.ReAct` (k=20, max_iters=40) with the `EnumerateAnchorEntities` signature (`question -> anchor_entities: list[str]`). Its sole job is exhaustively enumerating every anchor entity matching the question's filter criteria (e.g., all people whose hobby is auto racing), using varied phrasings to maximize coverage. High breadth retrieval (k=20) ensures wide corpus coverage per search call.

**`HopResolver`** (`src/program/phantomwiki_pipeline.py`)
Phase-2 module using `dspy.ReAct` (k=10, max_iters=20) with the `ResolveRelationalHops` signature (`question, anchor_entities -> answer: list[str]`). Receives the full accumulated anchor-entity list from Phase 1 and resolves any remaining relational hops (e.g., "find their cousin") to produce final answers. If no hop is needed it returns the anchor entities directly.

**`CountingRM`** (`src/program/counting_rm.py`)
A thin instrumentation wrapper around any DSPy retriever. It proxies all retrieval calls to the underlying model while incrementing a `call_count` counter, enabling retrieval-usage diagnostics without altering retrieval behavior.

### Data Flow
1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward`.
2. **Context setup**: `CountingRM(ColBERTv2)` is injected as the active retriever via `dspy.context`.
3. **Phase 1 – Entity Enumeration**: `EntityEnumerator` runs up to 40 ReAct iterations, retrieving k=20 passages per `search_wiki` call, to produce an exhaustive `anchor_entities` list.
4. **Phase 2 – Hop Resolution**: `HopResolver` receives the `anchor_entities` list and runs up to 20 ReAct iterations with k=10 passages per call to resolve relational hops and produce final answers.
5. **Output**: Final `answer: list[str]` wrapped in a `dspy.Prediction`.

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

