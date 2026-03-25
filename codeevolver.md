PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Program Overview

**PhantomWikiReActPipeline** is a DSPy-based multi-hop question answering system that answers questions over the PhantomWiki corpus — a synthetic, large-scale knowledge base. It uses a ReAct (Reasoning + Acting) loop to iteratively retrieve evidence and reason toward a final answer expressed as a list of strings.

---

## Key Modules

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level DSPy `Module` and entry point. Initializes a `CountingRM`-wrapped `ColBERTv2` retriever pointed at a remote Modal endpoint, then injects it as the active retrieval model via `dspy.context` before delegating to the inner agent.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with signature `question -> answer: list[str]` and up to 50 iterations. Exposes a `search_wiki(query)` tool that calls `dspy.Retrieve(k=7)` and joins the top-7 passages for the LLM to read.

- **`CountingRM`** (`src/program/counting_rm.py`): Thin instrumentation wrapper around any DSPy retriever. Proxies all retrieval calls while incrementing a `call_count` counter for telemetry reported at evaluation time.

- **`metric.py`** (`src/metric/metric.py`): Defines both `phantomwiki_f1` (plain float, for `Evaluate`) and `phantomwiki_f1_feedback` (for GEPA optimization). Both compute a set-based token F1 between normalized predicted and gold answer lists. The feedback variant returns a `ScoreWithFeedback` object with a detailed breakdown of correct, missed, and extra answers alongside the F1 score.

---

## Data Flow

1. A `question: str` enters `PhantomWikiReActPipeline.forward`.
2. `CountingRM` (wrapping ColBERTv2) is set as the DSPy retrieval context.
3. The question is forwarded to `PhantomWikiReAct.forward`.
4. The `ReAct` agent iteratively calls `search_wiki(query)`, retrieving top-7 passages from the remote ColBERT index (up to 50 tool-call iterations).
5. The agent produces a final `answer: list[str]`, returned as a `dspy.Prediction`.
6. During optimization/evaluation, `phantomwiki_f1_feedback` scores the prediction against gold answers, returning an F1 score and natural-language feedback used by the GEPA optimizer to improve prompts.

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

