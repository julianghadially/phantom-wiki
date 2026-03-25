PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the **PhantomWiki** corpus. Given a natural language question, it uses a ReAct (Reason + Act) agent to iteratively retrieve evidence from a remote ColBERTv2 index and reason over it to produce a final answer — which may be a list of entities or values.

### Key Modules & Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level DSPy `Module` and entry point. Instantiates the retrieval model and the ReAct sub-program, then injects the retriever into the DSPy context via `dspy.context(rm=...)` before delegating to the inner program.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Wraps `dspy.ReAct` with the signature `question -> answer: list[str]` and exposes a `search_wiki` tool. The agent may iterate up to 50 times, issuing retrieval queries and incorporating retrieved passages into its chain of thought before emitting a final answer list.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin instrumentation wrapper around `dspy.ColBERTv2`. It delegates retrieval calls to the underlying ColBERTv2 endpoint (hosted remotely via Modal) while maintaining a `call_count` for diagnostics.

### Data Flow
1. **Input**: A `question` string enters `PhantomWikiReActPipeline.forward()`.
2. **Retrieval setup**: `CountingRM` wraps a `ColBERTv2` instance pointed at a hosted Modal endpoint; top-7 passages are fetched per query.
3. **ReAct loop**: `PhantomWikiReAct` runs up to 50 think/act iterations, calling `search_wiki(query)` to retrieve and concatenate relevant passages, then reasoning over them.
4. **Output**: A `dspy.Prediction` with `answer: list[str]` is returned.

### Metric Being Optimized
`phantomwiki_f1_feedback` computes **token-set F1** between predicted and gold answer lists (after lowercasing and stripping), then returns a `ScoreWithFeedback` object containing the numeric F1 score plus a natural-language breakdown of correct, missed, and extra answers — designed for use with DSPy's GEPA optimizer which leverages textual feedback to improve prompt quality.

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

