PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Answer Completeness + Targeted Search + Synthesis Pipeline (ReAct + AnswerCompletenessChecker + AnswerSynthesizer)

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. `PhantomWikiReActPipeline` implements a four-pass architecture designed to combat under-enumeration: an initial ReAct investigation, a dedicated completeness-check module that identifies missing answer branches, targeted direct retrieval for gaps, and a final synthesis module that normalizes format and deduplicates all findings.

The pipeline composes `PhantomWikiReAct` (ReAct agent, max 50 iters, k=7), `AnswerCompletenessChecker` (ChainOfThought module detecting incomplete enumerations and generating follow-up queries), and `AnswerSynthesizer` (ChainOfThought module merging initial and supplemental results into a clean deduplicated answer list).

The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 between predicted and gold answer lists, and additionally returns structured textual feedback (correct, missed, and extra answers) compatible with the GEPA optimizer via DSPy's `ScoreWithFeedback`.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization, it instantiates a `CountingRM` wrapping a `dspy.ColBERTv2` retriever pointed at a remotely hosted PhantomWiki corpus (served via Modal), a `PhantomWikiReAct` inner reasoning program, a `dspy.ChainOfThought(AnswerCompletenessChecker)` module, and a `dspy.ChainOfThought(AnswerSynthesizer)` module.

**Four-pass forward()**: (1) ReAct pass — `PhantomWikiReAct` iterates up to 50 reasoning/action steps using `search_wiki` (ColBERTv2, k=7) to produce an initial `list[str]` answer. (2) Completeness check — `AnswerCompletenessChecker` receives the question and initial answers, reasons about whether multi-answer questions (siblings, hobbies, aggregations, multi-branch relationships) are fully covered, and outputs up to 4 targeted follow-up search queries (empty list if complete). (3) Targeted retrieval — for each follow-up query, `self.rm` is called directly (no ReAct overhead) and results are collected as supplemental passages. (4) Synthesis — `AnswerSynthesizer` merges initial answers and supplemental results, extracts only requested values (stripping "Person — N" format pollution), removes error strings, deduplicates, and returns the final `list[str]` answer. The synthesizer is always called — even when no follow-up queries were needed — to normalize format.

**AnswerCompletenessChecker** (`src/program/phantomwiki_pipeline.py`) is a `dspy.Signature` that explicitly handles PhantomWiki's multi-answer patterns: aggregation questions requiring enumeration of all matching people, and multi-branch relationship traversals. **AnswerSynthesizer** is a `dspy.Signature` that extracts clean values (counts as strings, names, attributes), removes non-answer strings, and deduplicates — directly fixing the "Person — N friends" format pollution that causes 0.0 F1 on aggregation questions.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the core ReAct agent (unchanged): `dspy.ReAct` with signature `question -> answer: list[str]`, `search_wiki` tool, max 50 iters, k=7.

**CountingRM** (`src/program/counting_rm.py`) wraps any retriever and tracks call counts.

**Metric — phantomwiki_f1_feedback** (`src/metric/metric.py`) scores via set-based token F1 with structured feedback for GEPA optimization.

**Data flow**: question → ReAct (ColBERTv2, ≤50 iters) → initial answers → AnswerCompletenessChecker → follow-up queries → direct rm calls → supplemental passages → AnswerSynthesizer → final deduplicated answer list → F1 evaluation.

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

