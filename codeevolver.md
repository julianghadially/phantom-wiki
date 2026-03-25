PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
`PhantomWikiReActPipeline` is a DSPy-based question-answering pipeline designed to answer multi-hop, knowledge-intensive questions over the **PhantomWiki** corpus — a large synthetic Wikipedia-style dataset. The pipeline uses a **ReAct** (Reasoning + Acting) agentic loop to iteratively retrieve passages and reason over them to produce a final list of answers.

---

### Key Modules and Responsibilities

| Module | Responsibility |
|---|---|
| `PhantomWikiReActPipeline` (`phantomwiki_pipeline.py`) | Top-level DSPy `Module`. Wires together the retrieval model and the ReAct program, injecting the retriever via `dspy.context`. |
| `PhantomWikiReAct` (`phantomwiki_module.py`) | Core agent module. Runs `dspy.ReAct` with up to 50 iterations, using `search_wiki` as its tool. Returns a `dspy.Prediction` with `answer: list[str]`. |
| `CountingRM` (`counting_rm.py`) | A thin wrapper around a `dspy.ColBERTv2` retrieval model. Delegates all retrieval calls and maintains a running `call_count` for diagnostic/efficiency tracking. |
| `dspy.ColBERTv2` (external) | Remote ColBERT retriever hosted on Modal, queried via HTTP for top-k passage retrieval against the PhantomWiki corpus. |
| `src.metric.metric` | Defines `phantomwiki_f1_feedback` — the optimization metric. |

---

### Data Flow

1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward()`.
2. **Context Injection**: The `CountingRM`-wrapped `ColBERTv2` retriever is set as the active retrieval model via `dspy.context(rm=self.rm)`.
3. **ReAct Loop**: `PhantomWikiReAct` runs a ReAct agent that, at each step, may call `search_wiki(query)` → `dspy.Retrieve(k=7)` → returns top-7 passages joined as a string.
4. **Output**: After up to 50 iterations, the agent produces `answer: list[str]` wrapped in a `dspy.Prediction`.

---

### Metric Being Optimized
`phantomwiki_f1_feedback` computes **token-level F1** between predicted and gold answer sets (case-insensitive). It returns a `ScoreWithFeedback` object (score ∈ [0,1] + human-readable feedback string) compatible with **GEPA**-style prompt optimization, detailing correct, missed, and extra answers to guide optimizer updates.

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

