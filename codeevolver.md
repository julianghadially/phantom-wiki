PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Three-Pass ReAct with Seed Entity Fan-Out, FollowUpInvestigation, and AnswerNormalizer Post-Processing

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. The top-level entry point, `PhantomWikiReActPipeline`, implements a three-pass sequential investigation strategy followed by answer normalization. The first pass runs `PhantomWikiReAct` (up to 30 iterations) to collect initial answers. The second pass runs a `FollowUpInvestigation` ReAct agent (up to 25 iterations) that explores alternative relationship chains and paths not yet investigated. A third "Seed Entity Fan-Out" pass explicitly addresses multi-entity under-enumeration by running a `SeedEnumeratorReAct` (up to 20 iterations, k=10 retrieval) to exhaustively list all intermediate entities, then retrieving passages for each seed and extracting additional answers via `MultiSeedAnswerExtractor`. The fully merged deduplicated answers are then cleaned by an `AnswerNormalizer` post-processing module before returning the final prediction.

`PhantomWikiReAct` implements the core ReAct agent pattern via `dspy.ReAct`, iteratively interleaving reasoning steps with calls to a `search_wiki` tool backed by the ColBERTv2 retriever. The `FollowUpInvestigation` signature directs the second-pass agent to enumerate answers reachable via unexplored paths. The `SeedEnumerationSignature` and `MultiSeedAnswerExtractorSignature` modules force exhaustive enumeration of ALL intermediate entities (friends of X, all people with occupation Y, etc.) before attribute extraction. The `AnswerNormalizerSignature` via `dspy.ChainOfThought` cleans format artifacts. The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 with structured feedback for GEPA optimization.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization, it instantiates a `CountingRM` wrapping a `dspy.ColBERTv2` retriever, a `PhantomWikiReAct` module for the first pass, a `dspy.Retrieve(k=7)` for the follow-up search tool, a `dspy.ReAct` agent over the `FollowUpInvestigation` signature for the second pass (max_iters=25), a `dspy.Retrieve(k=10)` for the fan-out pass, a `dspy.ReAct` agent over `SeedEnumerationSignature` (max_iters=20) for the third pass, a `dspy.ChainOfThought(MultiSeedAnswerExtractorSignature)` for answer extraction from seed passages, and a `dspy.ChainOfThought(AnswerNormalizerSignature)` for post-processing. The `forward` method runs all three passes sequentially, merges answers with deduplication after each pass, then normalizes the combined list before returning.

**SeedEnumerationSignature** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature used with `dspy.ReAct` (max_iters=20, k=10 retriever) that takes `question` and `already_found: list[str]` and produces `seed_entities: list[str]`. It explicitly instructs the agent to exhaustively enumerate ALL intermediate entities (e.g., all friends of X, all people with occupation Y) — targeting the primary failure mode of single-path exploration.

**MultiSeedAnswerExtractorSignature** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature used with `dspy.ChainOfThought` that takes `question`, `retrieved_passages` (concatenated passages for up to 6 seed entities, truncated to 4000 chars), and `already_found: list[str]`, and produces `additional_answers: list[str]` — new atomic values not already found.

**AnswerNormalizerSignature** (`src/program/phantomwiki_pipeline.py`) strips 'Name: count' format artifacts, removes error/uncertainty strings, and preserves clean atomic values. Falls back to original list if normalization would produce an empty result.

**FollowUpInvestigation** (`src/program/phantomwiki_pipeline.py`) directs a ReAct agent to explore alternative relationship chains and paths not yet investigated, treating `already_found` as non-exhaustive.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the primary ReAct agent with `dspy.Retrieve(k=7)` and up to 30 agentic iterations.

**CountingRM** (`src/program/counting_rm.py`) wraps any underlying retriever, incrementing a `call_count` on every retrieval invocation.

**Data flow**: A question enters `PhantomWikiReActPipeline.forward` → Pass 1: `PhantomWikiReAct` (ColBERTv2, top-7) produces initial `list[str]` answers → Pass 2: `FollowUpInvestigation` ReAct explores unexplored paths → Pass 3 (Fan-Out): `SeedEnumeratorReAct` (k=10) enumerates all intermediate entities (capped at 6), passages retrieved per seed and concatenated, `MultiSeedAnswerExtractor` extracts additional answers → fully merged deduplicated combined list → `AnswerNormalizer` cleans format artifacts → evaluated with F1 and structured feedback.

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

