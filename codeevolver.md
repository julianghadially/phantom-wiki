PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
`PhantomWikiReActPipeline` is a multi-hop question-answering system built on the DSPy framework. It answers natural-language questions by iteratively searching a PhantomWiki corpus (a synthetic Wikipedia-style knowledge base) and reasoning over retrieved passages using a ReAct (Reason + Act) agent loop.

### Key Modules

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): The top-level DSPy `Module` and entry point. Initialises the retrieval model and the core reasoning program, then executes them within a shared DSPy retrieval context on each `forward(question)` call.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin instrumentation wrapper around a `ColBERTv2` retrieval model. It proxies all retrieval calls while maintaining a `call_count` counter, enabling downstream telemetry on search frequency.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): The core reasoning agent. Uses `dspy.ReAct` with the signature `question -> answer: list[str]`, up to 50 iterations, and a single `search_wiki` tool. On each iteration the agent may invoke `search_wiki(query)`, which calls `dspy.Retrieve(k=7)` to fetch the top-7 passages from the PhantomWiki ColBERT index.

### Data Flow

1. A natural-language `question` enters `PhantomWikiReActPipeline.forward`.
2. The DSPy context is set with the `CountingRM`-wrapped ColBERT retriever.
3. `PhantomWikiReAct.react` runs iteratively: the LLM reasons about the question, decides to call `search_wiki`, retrieves up to 7 passages from the remote ColBERT server, and incorporates them into the next reasoning step.
4. After convergence (or 50 iterations), a `dspy.Prediction(answer: list[str])` is returned.

### Metric Being Optimised

`phantomwiki_f1_feedback` computes a token-set **F1 score** between predicted and gold answer lists (after lowercasing/stripping). It returns a `ScoreWithFeedback` object containing the numeric F1 (0–1) plus a human-readable string detailing correct, missed, and spurious answers — enabling GEPA-style optimisers to use both the score signal and the textual feedback for prompt refinement.

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

