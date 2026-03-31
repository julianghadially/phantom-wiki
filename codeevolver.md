PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Planner → Multi-Branch Sequential Investigation → Synthesizer with QuestionDecomposer, FocusedInvestigator, AnswerSynthesizer

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark, now structured as a three-phase Planner → Multi-Branch Sequential Investigation → Synthesizer architecture. `PhantomWikiReActPipeline` orchestrates a `QuestionDecomposer` (planner), a `FocusedInvestigator` (per-branch ReAct agent), and an `AnswerSynthesizer` (deduplicating aggregator), with `PhantomWikiReAct` retained as a fallback.

The `QuestionDecomposer` (via `dspy.ChainOfThought`) breaks a multi-hop question into 2–4 independent investigation branches, each targeting a distinct logical chain. A shared `FocusedInvestigator` module — a lightweight `dspy.ReAct` with `k=10` retrieval and 15 max iterations — is invoked sequentially for each branch, accumulating partial answers. The `AnswerSynthesizer` (via `dspy.ChainOfThought`) deduplicates and format-normalizes all partial answers into a final `list[str]` response.

The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 with structured textual feedback compatible with GEPA optimization.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. It instantiates a `CountingRM`-wrapped `dspy.ColBERTv2` retriever, the original `PhantomWikiReAct` as fallback, a `dspy.ChainOfThought(QuestionDecomposer)` planner, a `FocusedInvestigator` sub-module, and a `dspy.ChainOfThought(AnswerSynthesizer)`. The `forward` method runs three sequential phases inside `dspy.context(rm=self.rm)`.

**Phase 1 — Decompose**: `QuestionDecomposer` (signature class) receives the original question and emits 2–4 independent sub-questions (branches), each targeting a distinct path through the knowledge graph (different intermediate entity, ancestor path, or candidate starting node). Capped at 4 branches to control compute.

**Phase 2 — Investigate**: `FocusedInvestigator` (a `dspy.Module`) wraps a `dspy.Retrieve(k=10)` and a `dspy.ReAct` with signature `sub_question -> answer: list[str]` and `max_iters=15`. It is invoked sequentially for each branch; exceptions are caught and silently skipped. All answers from all branches are accumulated into `all_partial`.

**Phase 3 — Synthesize**: `AnswerSynthesizer` (signature class) receives the original question plus all accumulated partial answers and produces a final deduplicated, correctly-formatted `list[str]`. Handles count-aggregation (bare numeric strings) and entity-lookup (bare names) normalization. If all branches failed (empty `all_partial`), the pipeline falls back to `PhantomWikiReAct`.

**CountingRM** (`src/program/counting_rm.py`) wraps the ColBERTv2 retriever with a call counter for monitoring. **PhantomWikiReAct** (`src/program/phantomwiki_module.py`) is the original single-agent ReAct (k=7, max_iters=50), preserved as fallback. **Metric** (`src/metric/metric.py`) uses token-level set F1 with GEPA-compatible `ScoreWithFeedback`.

**Data flow**: question → `QuestionDecomposer` → [branch_1, …, branch_N] → sequential `FocusedInvestigator` per branch → `all_partial` answers → `AnswerSynthesizer` → final `dspy.Prediction(answer=list[str])`.

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

