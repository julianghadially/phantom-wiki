PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "Anchor-entity pre-enumeration + adaptive two-pass ReAct with CompletenessCheck gating, AnchorEntityExtraction, and deterministic answer normalizer"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs an anchor-entity pre-enumeration step followed by an adaptive two-pass ReAct architecture: before Pass 1, a broad k=25 retrieval identifies all entities matching the anchor attribute in the question via `ChainOfThought(AnchorEntityExtraction)`, which enhances the question with an explicit entity list. A first ReAct pass gathers initial answers, a `ChainOfThought` completeness checker gates whether a second targeted search round is needed, and a deterministic normalizer strips "Name — Value" format contamination from all answers.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module implementing the pre-enumeration + two-pass adaptive search pipeline. Contains `dspy.Retrieve(k=7)` for the `search_wiki` tool, `dspy.Retrieve(k=25)` for broad anchor retrieval, `dspy.ChainOfThought(AnchorEntityExtraction)` for pre-enumeration, `dspy.ReAct` (question → answer: list[str], up to 50 iters), `dspy.ChainOfThought(CompletenessCheck)` for mid-pipeline gating, and `_normalize_answers` for deterministic format cleanup.

- **`AnchorEntityExtraction`** (`src/program/phantomwiki_module.py`): DSPy Signature with inputs `question` and `search_passages` (broad k=25 retrieval results) and outputs `anchor_attribute: str` and `matching_entities: list[str]`. Used by `ChainOfThought` to identify all entities satisfying the anchor attribute before Pass 1.

- **`CompletenessCheck`** (`src/program/phantomwiki_module.py`): DSPy Signature with inputs `question` and `current_answers` and outputs `appears_complete: bool` and `follow_up_hint: str`. Used by `ChainOfThought` to decide whether a second ReAct pass is warranted.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. **Pre-enumeration**: `self.retrieve_broad(question)` fetches k=25 passages. `ChainOfThought(AnchorEntityExtraction)` identifies the anchor attribute and all matching entity names. If entities are found, the question is enhanced with an explicit entity list and a directive to trace each one.
4. **Pass 1**: `PhantomWikiReAct.forward` runs `dspy.ReAct` with the (potentially enhanced) question, calling `search_wiki(query)` up to 50 times (k=7 passages each). Results are normalized via `_normalize_answers` (strips "Name — Value" contamination).
5. **Completeness check**: `ChainOfThought(CompletenessCheck)` evaluates pass-1 answers. If `appears_complete` is True, answers are returned immediately (protects single-answer questions from regression).
6. **Pass 2** (if incomplete): A follow-up question appending the already-found answers and the anchor entity list is fed into a second `dspy.ReAct` call to find additional missing entities via different search angles. Results are normalized.
7. **Merge**: Pass-1 and pass-2 answers are merged with `dict.fromkeys` (order-preserving deduplication) and returned as `dspy.Prediction(answer=...)`.

## ARCHITECTURE DESCRIPTION: The PhantomWikiReActPipeline implements an anchor-entity pre-enumeration step followed by an adaptive two-pass ReAct architecture. Before Pass 1, a broad dspy.Retrieve(k=25) fetches 25 passages for the question, and ChainOfThought(AnchorEntityExtraction) identifies the anchor attribute (e.g. "occupation: farm manager") and extracts all matching entity names from those passages. If matching_entities is non-empty, the question is enhanced with an explicit entity list and a directive instructing the agent to trace each entity individually. This pre-enumeration targets the core failure mode where the agent stops after finding a few answers when questions have dozens of correct ones. The core module PhantomWikiReAct also runs the initial ReAct search pass (up to 50 iterations, top-7 ColBERT passages per query via dspy.Retrieve(k=7)), then applies _normalize_answers to strip "Name — Value" format contamination. A ChainOfThought(CompletenessCheck) module gates whether a second pass is needed: if appears_complete is True, the pipeline returns immediately to protect single-answer questions from regression. If incomplete, a follow-up question appending already-found answers and anchor entity context is constructed and fed into a second ReAct call. Pass-2 answers are normalized and merged using order-preserving deduplication (dict.fromkeys). The AnchorEntityExtraction signature takes question and search_passages as inputs and produces anchor_attribute (str) and matching_entities (list[str]) as outputs. The CompletenessCheck signature produces appears_complete (bool) and follow_up_hint (str).

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

