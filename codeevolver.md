PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the PhantomWiki corpus. Given a natural-language question, it iteratively retrieves relevant passages and reasons over them to produce one or more answers. It is optimized using the GEPA framework (DSPy) with an F1-based feedback metric.

### Key Modules

**`PhantomWikiReActPipeline`** (entry point, `src/program/phantomwiki_pipeline.py`)
Top-level `dspy.Module` implementing a dual-path architecture: an aggregation path for per-entity counting questions and a multi-strategy fan-out path for all other questions. On initialization it creates a `CountingRM`-wrapped `ColBERTv2` retriever, a `PhantomWikiReAct` reasoning module, a `ChainOfThought(GenerateSearchStrategies)` for generating diverse search strategies, a `ChainOfThought(MergeAnswers)` for deduplicating results, and a `ChainOfThought(DecomposeAggregationQuestion)` for detecting and decomposing aggregation questions. In `forward`, it first uses `decomposer` to classify the question; if aggregation, it enumerates qualifying entities then runs a per-entity count query for each (up to 25); otherwise it falls back to the 4-strategy fan-out path.

**`DecomposeAggregationQuestion`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature that classifies whether a question requires per-entity aggregation (e.g., "How many friends does each person born in 1990 have?"). Outputs `is_aggregation` (bool), `enumeration_query` (query to find all qualifying entities), and `count_query_template` (query with `{name}` placeholder for per-entity counting).

**`GenerateSearchStrategies`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for generating 3 diverse, independent search strategies/rephrased questions from the original question to ensure exhaustive answer coverage across different starting points in the knowledge graph.

**`MergeAnswers`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for merging and deduplicating candidate answers collected across multiple search strategies into a single comprehensive list of valid, distinct answers.

**`CountingRM`** (`src/program/counting_rm.py`)
A thin instrumentation wrapper around any DSPy retriever. It proxies all retrieval calls to the underlying model while incrementing a `call_count` counter, enabling retrieval-usage diagnostics without altering retrieval behavior.

**`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`)
Core reasoning module built on `dspy.ReAct`. Uses the signature `question -> answer: list[str]` with up to 50 reasoning iterations. Its single registered tool, `search_wiki`, invokes `dspy.Retrieve(k=7)` to fetch the top-7 passages from the PhantomWiki corpus for a given query and returns them as a newline-separated string for the LLM to reason over.

### Data Flow
1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward`.
2. **Aggregation detection**: `DecomposeAggregationQuestion` (via `ChainOfThought`) classifies the question. If `is_aggregation=True`:
   a. **Enumeration**: `PhantomWikiReAct` runs on `enumeration_query` to retrieve all qualifying entities.
   b. **Per-entity counting**: For each entity (up to 25), the `count_query_template` is instantiated with the entity name and `PhantomWikiReAct` runs to get that entity's count.
   c. All counts are merged via `MergeAnswers`.
3. **Standard path** (non-aggregation): `GenerateSearchStrategies` produces 3 diverse strategies; `PhantomWikiReAct` runs for each strategy plus the original question (4 total) inside a `dspy.context(rm=self.rm)` block; all answers are concatenated and merged via `MergeAnswers`.
4. **Output**: A `dspy.Prediction(answer=...)` with the merged, deduplicated answer list.

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

