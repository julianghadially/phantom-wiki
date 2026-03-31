PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Three-Stage Decompose→Enumerate→Synthesize Pipeline with EntityEnumeratorReAct, ParallelAttributeFetcher, and AnswerSynthesizer

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. `PhantomWikiReActPipeline` implements a three-stage Decompose→Enumerate→Synthesize architecture to reliably answer multi-answer questions. Stage 1 uses `EntityEnumeratorReAct` (a `dspy.ReAct` with k=10 retrieval, max_iters=20) to exhaustively enumerate all target entities. Stage 2 uses `ParallelAttributeFetcher` to fetch targeted passages per entity (k=5) and extract bare attribute values via `dspy.ChainOfThought(AttributeExtractor)` in parallel. Stage 3 uses `dspy.ChainOfThought(AnswerSynthesizer)` to deduplicate and normalize entity-attribute pairs into a final `list[str]` answer.

If Stage 1 returns no entities (question types that don't fit the enumeration pattern), the pipeline falls back to the original `PhantomWikiReAct` single ReAct agent for full coverage.

The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 between predicted and gold answer lists, and additionally returns structured textual feedback compatible with the GEPA optimizer via DSPy's `ScoreWithFeedback`.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization it instantiates a `CountingRM` wrapping a `dspy.ColBERTv2` retriever, a `fallback_react` (`PhantomWikiReAct`), an `entity_enumerator` (`EntityEnumeratorReAct`), an `attribute_fetcher` (`ParallelAttributeFetcher`), and an `answer_synthesizer` (`dspy.ChainOfThought(AnswerSynthesizer)`). The `forward` method orchestrates the three stages within `dspy.context(rm=self.rm)`.

**Stage 1 — EntityEnumeratorReAct** (`src/program/phantomwiki_module.py`): A `dspy.ReAct` agent with signature `question -> all_target_entities: list[str], attribute_to_collect: str`, using `dspy.Retrieve(k=10)` and up to 20 iterations. Its sole purpose is exhaustive entity identification (e.g., ALL siblings, ALL friends matching a criterion) without prematurely collecting attribute values.

**Stage 2 — ParallelAttributeFetcher** (`src/program/phantomwiki_module.py`): A `dspy.Module` that caps entities at 4, then for each runs a targeted `dspy.Retrieve(k=5)` on the entity name and calls `dspy.ChainOfThought(AttributeExtractor)` to extract the bare attribute value. Fetches are dispatched via `ThreadPoolExecutor(max_workers=4)` for parallelism. `AttributeExtractor` is a `dspy.Signature` with `question`, `entity_name`, `passages` as inputs and `attribute_value` (bare value, no prefix) as output.

**Stage 3 — AnswerSynthesizer** (`src/program/phantomwiki_module.py`): A `dspy.ChainOfThought` on the `AnswerSynthesizer` signature, which receives the question and a JSON list of `{entity, value}` dicts and emits a deduplicated `list[str]` of bare values.

**Fallback — PhantomWikiReAct**: The original `dspy.ReAct` agent (k=7, max_iters=50) is retained and invoked when Stage 1 yields an empty entity list.

**CountingRM** tracks retrieval call counts for monitoring. **Metric — phantomwiki_f1_feedback** evaluates with set-based F1 and returns structured feedback for GEPA optimization.

**Data flow**: question → Stage 1 enumerates entities → Stage 2 fetches attribute per entity in parallel → Stage 3 synthesizes final `list[str]` answer (or fallback to single ReAct if no entities found) → F1 evaluation.

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

