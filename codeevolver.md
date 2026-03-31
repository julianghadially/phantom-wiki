PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Three-Pass ReAct with PathDecomposer + Sequential Micro-Investigation (Pass 1.5), FollowUpInvestigation, and AnswerNormalizer

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. `PhantomWikiReActPipeline` implements a three-stage investigation strategy: Pass 1 (primary ReAct, up to 30 iters), Pass 1.5 (PathDecomposer + Sequential Micro-Investigations), and Pass 2 (FollowUpInvestigation ReAct, up to 25 iters), followed by AnswerNormalizer post-processing.

Pass 1.5 is the key addition: after Pass 1 collects initial answers, a `PathDecomposerSignature` ChainOfThought module reasons over the partial answers to identify up to 4 unexplored entity/relationship paths. Each path is then handed to a `TargetedInvestigation` ReAct agent (max_iters=8) which focuses exclusively on that specific path. The resulting micro-answers are deduplicated and merged with Pass 1 answers to form an enriched context that Pass 2 receives as `already_found`, enabling FollowUpInvestigation to explore genuinely novel paths rather than re-discovering already-explored ones. The combined deduplicated answers are normalized by `AnswerNormalizerSignature` before returning the final prediction.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization it instantiates: a `CountingRM`-wrapped `dspy.ColBERTv2` retriever, `PhantomWikiReAct` for Pass 1, `dspy.Retrieve(k=7)` for the shared `_search_wiki` tool, `dspy.ReAct(FollowUpInvestigation, max_iters=25)` for Pass 2, `dspy.ChainOfThought(PathDecomposerSignature)` for path planning, `dspy.ReAct(TargetedInvestigation, max_iters=8)` for micro-investigations, and `dspy.ChainOfThought(AnswerNormalizerSignature)` for post-processing.

**Data flow**: question → Pass 1 (`PhantomWikiReAct`, ≤30 iters, ColBERTv2 top-7) → partial `list[str]` answers → Pass 1.5: `PathDecomposerSignature` (ChainOfThought) identifies ≤4 unexplored entity/relationship paths → sequential `TargetedInvestigation` ReAct (≤8 iters each) per path → micro_answers merged/deduped with Pass 1 answers → enriched `already_found` → Pass 2: `FollowUpInvestigation` ReAct (≤25 iters) explores remaining paths → full combined deduplicated list → `AnswerNormalizerSignature` (ChainOfThought) strips 'Name: count' artifacts and error strings → final `list[str]` prediction.

**PathDecomposerSignature** takes `question` and `partial_answers: list[str]`, outputs `unexplored_paths: list[str]` (1–4 actionable entity/relationship path descriptions). Used via ChainOfThought for deliberate path planning between passes.

**TargetedInvestigation** takes `question` and `investigation_target: str` (a single specific path), outputs `answer: list[str]`. Used via `dspy.ReAct` with max_iters=8 — a lightweight focused agent per path rather than a single broad sweep.

**FollowUpInvestigation** takes `question` and `already_found: list[str]`, outputs `answer: list[str]`. Now receives enriched context from Pass 1 + Pass 1.5 combined, enabling it to target truly unexplored territory.

**AnswerNormalizerSignature** (`dspy.ChainOfThought`) cleans 'Name: count' artifacts, removes error/uncertainty strings, and falls back to raw answers if normalization would yield empty results.

**CountingRM** wraps `dspy.ColBERTv2` and increments a `call_count` on every retrieval. **Metric** (`phantomwiki_f1_feedback`) scores with token-level set F1 and returns `ScoreWithFeedback` for GEPA optimization.

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

