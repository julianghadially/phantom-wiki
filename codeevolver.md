PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "Adaptive two-pass ReAct with CompletenessCheck gating and deterministic answer normalizer"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs an adaptive two-pass ReAct architecture: a first ReAct pass gathers initial answers, a `ChainOfThought` completeness checker gates whether a second targeted search round is needed, and a deterministic normalizer strips "Name — Value" format contamination from all answers.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module implementing the two-pass adaptive search pipeline. Contains `dspy.ReAct` (question → answer: list[str], up to 50 iters), `dspy.ChainOfThought(CompletenessCheck)` for mid-pipeline gating, and `_normalize_answers` for deterministic format cleanup.

- **`CompletenessCheck`** (`src/program/phantomwiki_module.py`): DSPy Signature with inputs `question` and `current_answers` and outputs `appears_complete: bool` and `follow_up_hint: str`. Used by `ChainOfThought` to decide whether a second ReAct pass is warranted.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. **Pass 1**: `PhantomWikiReAct.forward` runs `dspy.ReAct`, calling `search_wiki(query)` up to 50 times. Results are normalized via `_normalize_answers` (strips "Name — Value" contamination).
4. **Completeness check**: `ChainOfThought(CompletenessCheck)` evaluates pass-1 answers. If `appears_complete` is True, answers are returned immediately (protects single-answer questions from regression).
5. **Pass 2** (if incomplete): A follow-up question appending the already-found answers is fed into a second `dspy.ReAct` call to find additional missing entities via different search angles. Results are normalized.
6. **Merge**: Pass-1 and pass-2 answers are merged with `dict.fromkeys` (order-preserving deduplication) and returned as `dspy.Prediction(answer=...)`.

## ARCHITECTURE DESCRIPTION: The PhantomWikiReActPipeline implements an adaptive two-pass ReAct architecture to address the dominant failure mode where the agent finds 1–5 answers and halts when questions have 8–360+ correct answers. The core module PhantomWikiReAct runs an initial ReAct search pass (up to 50 iterations, top-7 ColBERT passages per query), then applies a deterministic _normalize_answers method to strip "Name — Value" format contamination that causes 0.0 F1 even when correct entities are found. A ChainOfThought(CompletenessCheck) module then gates whether a second pass is needed: if appears_complete is True, the pipeline returns immediately to protect already-correct single-answer questions from regression. If incomplete, a follow-up question string is constructed that appends the already-found answers as context and instructs the ReAct agent to try different search angles to find missing entities. Pass-2 answers are normalized and merged with pass-1 answers using order-preserving deduplication (dict.fromkeys). The CompletenessCheck DSPy Signature takes question and current_answers as inputs and produces appears_complete (bool) and follow_up_hint (reformulated search hint) as outputs. This pipeline-level two-pass approach forces a second targeted search round without adding new tools the agent might ignore, leveraging the LM's chain-of-thought reasoning to intelligently decide when additional search effort is warranted.

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

