PARENT_MODULE_PATH: src.program.baseline_rlm.rlm_pipeline.RLMPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This program is a **Retrieval-augmented Language Model (RLM) pipeline** designed to answer questions about a fictional universe (PhantomWiki). It performs multi-hop question answering by iteratively searching a corpus of fictional character articles and synthesizing answers from the retrieved passages.

### Key Modules & Responsibilities

- **`RLMPipeline`** (`rlm_pipeline.py`): Top-level DSPy `Module` and entry point. Initializes a `CountingRM`-wrapped `ColBERTv2` retrieval model pointed at a remote PhantomWiki ColBERT server, and delegates all question-answering to `PhantomWikiRLM` within a retrieval model context.

- **`PhantomWikiRLM`** (`rlm_module.py`): Core reasoning module. Wraps DSPy's `RLM` (Reasoning Language Model) with a `search_wiki` tool. Configured with `k=7` passages per retrieval, up to 15 reasoning iterations, and up to 50 LLM calls. Returns `answer: list[str]` predictions.

- **`CountingRM`** (`counting_rm.py`): A thin instrumentation wrapper around a `dspy.Retrieve` instance that tracks the number of retrieval calls made, useful for monitoring and optimization.

- **`phantomwiki_f1_feedback`** (`metric/metric.py`): Evaluation metric returning a `ScoreWithFeedback` object. Computes token-set F1 between predicted and gold answer lists, plus structured textual feedback detailing correct, missed, and extra answers — suitable for GEPA-style optimizer feedback.

### Data Flow

1. A `question` string is passed to `RLMPipeline.forward()`.
2. The pipeline sets `CountingRM` as the active DSPy retrieval model and calls `PhantomWikiRLM`.
3. `PhantomWikiRLM` invokes DSPy's `RLM`, which iteratively calls `search_wiki(query)` as a tool, retrieving top-7 passages from ColBERTv2 and joining them with separators.
4. After up to 15 iterations / 50 LLM calls, `RLM` produces an `answer: list[str]`.
5. The pipeline returns a `dspy.Prediction(answer=...)`.
6. At evaluation time, `phantomwiki_f1_feedback` computes the F1 score between predicted and gold answers and returns detailed feedback for optimizer-guided improvement.

### Metric Being Optimized
**`phantomwiki_f1_feedback`** — Answer-level F1 score (harmonic mean of precision and recall over normalized answer sets), augmented with human-readable feedback (correct/missed/extra answers) for use with DSPy's GEPA optimizer.

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

