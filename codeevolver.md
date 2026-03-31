PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Workspace-Driven Entity Gap-Filling ReAct with GapAnalyzer and EntityTargetedFocusSignature

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. The top-level entry point, `PhantomWikiReActPipeline`, orchestrates a two-pass workspace-driven architecture: a broad initial ReAct pass followed by targeted entity gap-filling. On the first pass, the inner `PhantomWikiReAct` agent (up to 30 iterations) explores the question space and produces initial answers along with a trajectory of thoughts and observations. The pipeline then uses `GapAnalyzer` (ChainOfThought) to identify named entities that were encountered but whose final attributes were never retrieved. For each such gap entity (up to 4), a focused `entity_react` (ReAct with `EntityTargetedFocusSignature`, up to 12 iterations) performs a targeted investigation. All answers are merged and deduplicated preserving order.

`PhantomWikiReActPipeline` owns a shared `search_wiki` tool (backed by `dspy.Retrieve(k=7)` and a remote ColBERTv2 index) used by both the entity_react and the inner PhantomWikiReAct module. The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 and returns GEPA-compatible structured feedback.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module` implementing the workspace-driven gap-filling architecture. It initializes: a `CountingRM`-wrapped `dspy.ColBERTv2` retriever; a `dspy.Retrieve(k=7)` used by its own `search_wiki` method; the inner `PhantomWikiReAct` program; a `dspy.ChainOfThought(GapAnalyzer)` for gap analysis; and a `dspy.ReAct(EntityTargetedFocusSignature, tools=[self.search_wiki], max_iters=12)` for entity-focused follow-up. The `forward` method (1) calls the inner react directly to capture the full trajectory prediction, (2) extracts a 2000-char trajectory summary from thought/observation fields, (3) invokes GapAnalyzer to identify unfilled entity gaps and determine if the investigation is complete, (4) if incomplete, loops over up to 4 pending entities running targeted ReAct passes, and (5) merges and deduplicates all answers before returning.

**GapAnalyzer** (defined in `phantomwiki_pipeline.py`) is a `dspy.Signature` used with `ChainOfThought`. It takes the original question, initial answers, and trajectory summary, and outputs `pending_entities` (named entities mentioned as intermediate nodes whose attributes were never finally retrieved) and `investigation_complete` (bool).

**EntityTargetedFocusSignature** (defined in `phantomwiki_pipeline.py`) is a `dspy.Signature` for the entity-focused ReAct sub-agent. It takes the question, a specific `target_entity`, and `context_from_prior_search`, and outputs `partial_answers` — the answers found for that entity.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the core first-pass ReAct agent with `dspy.ReAct(question -> answer: list[str], max_iters=30)` and its own `search_wiki` tool backed by ColBERTv2.

**CountingRM** (`src/program/counting_rm.py`) is a lightweight `dspy.Retrieve` subclass that counts retrieval invocations for monitoring.

**Metric — phantomwiki_f1_feedback** (`src/metric/metric.py`) evaluates predictions using token-level set F1 and returns a `ScoreWithFeedback` object for GEPA-based optimizer-driven prompt refinement.

**Data flow**: question → PhantomWikiReActPipeline.forward → [Pass 1] PhantomWikiReAct (30-iter ReAct, ColBERTv2) → initial_answers + trajectory → GapAnalyzer (ChainOfThought) → pending_entities → [Pass 2, per entity] EntityTargetedFocusSignature ReAct (12-iter) → partial_answers → merge + deduplicate → final answer list → F1 evaluation.

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

