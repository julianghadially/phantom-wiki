PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "CompletenessCheck-gated 2-pass ReAct with enhanced answer normalization (iteration 10 all-time best, valset 0.6083)"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs a **CompletenessCheck-gated 2-pass ReAct** architecture: after an initial ReAct pass, a `ChainOfThought(CompletenessCheck)` module assesses whether the current answers fully satisfy the question. If the answers appear incomplete, a targeted follow-up ReAct pass is run using the already-found answers and a reformulated search hint. An enhanced deterministic normalizer strips multiple forms of format contamination from all answers.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module implementing the CompletenessCheck-gated 2-pass search pipeline. Contains `dspy.ReAct` (question → answer: list[str], up to 50 iters), `dspy.ChainOfThought(CompletenessCheck)` for completeness-gating between passes, and an enhanced `_normalize_answers` for deterministic format cleanup.

- **`CompletenessCheck`** (`src/program/phantomwiki_module.py`): DSPy Signature with inputs `question` and `current_answers`, and outputs `appears_complete: bool` and `follow_up_hint: str`. Used by `ChainOfThought` to assess whether a second ReAct pass is needed and to generate a targeted hint for finding missing answers.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. **Pass 1**: `PhantomWikiReAct.forward` runs `dspy.ReAct`, calling `search_wiki(query)` up to 50 times. Results are normalized via `_normalize_answers` (strips "Name — Value", "Name: value", and "relationship: Name" format contamination).
4. **Completeness check**: `ChainOfThought(CompletenessCheck)` assesses pass-1 answers. If `appears_complete` is True, answers are returned immediately.
5. **Pass 2** (if incomplete): A targeted follow-up question embedding already-found answers and the `follow_up_hint` is fed into a second `dspy.ReAct` call. Results are normalized and merged with pass-1 answers using order-preserving deduplication (`dict.fromkeys`).

## ARCHITECTURE DESCRIPTION: The PhantomWikiReActPipeline implements the CompletenessCheck-gated architecture (iteration 10 all-time best, valset 0.6083) for multi-hop factual QA over a private PhantomWiki corpus. The pipeline runs up to 2 ReAct passes (up to 50 iterations each, top-7 ColBERT passages per query). After Pass 1, a ChainOfThought(CompletenessCheck) module evaluates whether the current answers fully satisfy the question and, if not, produces a follow_up_hint — a reformulated search directive targeting the missing entities. Pass 2 embeds both the already-found answers and this hint into the follow-up question, allowing the ReAct agent to focus its search on gaps. The appears_complete gate protects single-answer or already-complete questions from unnecessary extra inference. The enhanced _normalize_answers method strips three forms of format contamination: (1) "Name — Value" em-dash patterns (keep right side), (2) "Firstname Lastname: value" person-name colon patterns (detected by ≥2 consecutive capitalized words before a colon, keep right side), and (3) "relationship-word: Name" lowercase prefix patterns (e.g., "great-grandfather: Name", keep right side). All passes are merged with order-preserving deduplication via dict.fromkeys.

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

