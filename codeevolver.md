PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "ReAct with persistent in-process scratchpad (write_note / read_notes tools)"

## Architecture Summary

### High-Level Purpose
PhantomWikiReActPipeline is a DSPy-based question-answering system designed to answer multi-hop factual questions by iteratively searching a private knowledge corpus called PhantomWiki. It employs a ReAct (Reasoning + Acting) loop that interleaves language model reasoning steps with targeted retrieval actions and scratchpad operations, enabling it to exhaustively track and process all discovered entities across multiple hops before producing a final answer.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level `dspy.Module` pipeline. Instantiates the retrieval model and the core ReAct program, then injects the retrieval model into the DSPy context on every `forward(question)` call. Acts as the main entry point for evaluation and optimization.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin `dspy.Retrieve` wrapper around a ColBERTv2 retriever (hosted remotely via Modal). Tracks the number of retrieval calls made during inference via an internal counter, useful for efficiency monitoring.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with the signature `question -> answer: list[str]` and three tools: `search_wiki`, `write_note`, and `read_notes`. The ReAct agent iterates up to 50 steps, issuing search queries, writing notes about discovered entities to a persistent in-process scratchpad, reading those notes back to ensure exhaustive processing, and finally emitting a list of answer strings. Notes are reset at the start of each `forward()` call to prevent state leakage between questions.

### Data Flow
1. A `question` string enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` wraps the ColBERTv2 remote retriever and is set as the active DSPy retrieval model.
3. `PhantomWikiReAct.forward` resets `self.notes = []`, then feeds the question into `dspy.ReAct`.
4. The ReAct loop can call `search_wiki(query)` to fetch passages, `write_note(note)` to persist intermediate findings (e.g., "Brothers of X: Carmine, Damon, Sal — need occupations for all three"), and `read_notes()` to review previously saved context before deciding next steps.
5. Each `search_wiki` call uses `dspy.Retrieve(k=7)` to fetch passages from the PhantomWiki ColBERT index.
6. The scratchpad tools address the "satisfice on first answer" failure mode by giving the model explicit external memory to track discovered-but-unprocessed entities.
7. The ReAct loop reasons over accumulated passages and notes, then terminates with `answer: list[str]`.
8. A `dspy.Prediction(answer=...)` is returned upstream.

## ARCHITECTURE DESCRIPTION: PhantomWikiReActPipeline uses a DSPy ReAct loop with three tools to answer multi-hop factual questions against a PhantomWiki ColBERT corpus. The core innovation is a persistent in-process scratchpad (write_note / read_notes tools) alongside the existing search_wiki tool, giving the LM lightweight external memory to record all discovered entities and intermediate facts during reasoning. This directly addresses the dominant failure mode where the agent finds multiple intermediate entities but only follows through on the first before finishing. The agent can now write structured to-do notes (e.g., "Brothers of X: Carmine, Damon, Sal — need occupations for all three"), then read them back to ensure exhaustive traversal before emitting the final answer list. Notes are cleared at the start of each forward() call to prevent cross-question state leakage. The ReAct loop runs up to 50 iterations, retrieving top-7 passages per search query via a remote ColBERTv2 index hosted on Modal.

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

