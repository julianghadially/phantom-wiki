PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "Entity Frontier Tracker: adaptive 3-pass ReAct with named-entity frontier extraction and targeted reverse-link queries"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs an adaptive up-to-3-pass ReAct architecture driven by an **Entity Frontier Tracker**: after each ReAct pass, a `ChainOfThought(EntityFrontierExtractor)` module identifies intermediate named entities that were discovered but not fully explored, generates concrete reverse-link search queries (e.g., "son of [name]", "children of [name]") to target those entities in subsequent passes, and gates whether further search is warranted. A deterministic normalizer strips "Name — Value" format contamination from all answers.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module implementing the adaptive multi-pass search pipeline. Contains `dspy.ReAct` (question → answer: list[str], up to 50 iters), `dspy.ChainOfThought(EntityFrontierExtractor)` for mid-pipeline frontier extraction and gating, and `_normalize_answers` for deterministic format cleanup.

- **`EntityFrontierExtractor`** (`src/program/phantomwiki_module.py`): DSPy Signature with inputs `question`, `current_answers`, and `pass_reasoning`, and outputs `frontier_entities: list[str]` (named entities needing further exploration), `targeted_search_hints: list[str]` (concrete queries like "son of [name]"), and `needs_more_exploration: bool`. Used by `ChainOfThought` to produce specific entity names and search queries for subsequent passes, directly addressing the failure mode where the agent finds intermediate entities but does not issue reverse-link queries to exhaust their relatives.

- **`CompletenessCheck`** (`src/program/phantomwiki_module.py`): Legacy DSPy Signature retained for reference; no longer used in the active pipeline.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. **Pass 1**: `PhantomWikiReAct.forward` runs `dspy.ReAct`, calling `search_wiki(query)` up to 50 times. Results are normalized via `_normalize_answers` (strips "Name — Value" contamination).
4. **Frontier extraction 1**: `ChainOfThought(EntityFrontierExtractor)` inspects pass-1 answers and reasoning. If `needs_more_exploration` is False, answers are returned immediately.
5. **Pass 2** (if frontier found): A targeted follow-up question embedding the frontier entities and concrete search hints is fed into a second `dspy.ReAct` call. Results are normalized and merged with pass-1 answers.
6. **Frontier extraction 2**: A second `EntityFrontierExtractor` call on the merged answers. If `needs_more_exploration` is False, merged answers are returned.
7. **Pass 3** (if still incomplete): A third targeted `dspy.ReAct` call using the updated frontier. Results are normalized and merged with all prior answers using order-preserving deduplication (`dict.fromkeys`).

## ARCHITECTURE DESCRIPTION: The PhantomWikiReActPipeline implements an Entity Frontier Tracker architecture to address the core failure mode where the agent discovers intermediate entities (e.g., Lea Fassett → Ty Fassett, Derick Lafave, Joey Lafave) but concludes "no great-grandchildren found" instead of issuing reverse-link queries like "son of Derick Lafave" to find the next generation. The pipeline runs up to 3 ReAct passes (up to 50 iterations each, top-7 ColBERT passages per query). After each pass, a ChainOfThought(EntityFrontierExtractor) module inspects the reasoning trace and current answers to produce: (1) frontier_entities — specific named entities whose relatives have not yet been searched, and (2) targeted_search_hints — concrete ColBERT-friendly queries such as "son of [name]" or "[name] family". These named-entity queries exploit ColBERT's strength at named-entity retrieval. The needs_more_exploration boolean gates whether additional passes are run, protecting single-answer questions from unnecessary extra search. Each follow-up question explicitly embeds the frontier entities and search hints as context for the next ReAct pass. After every pass, _normalize_answers strips "Name — Value" format contamination, and all passes are merged with order-preserving deduplication via dict.fromkeys. The CompletenessCheck signature is retained in the file but is no longer used in the active forward() path.

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

