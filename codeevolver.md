PARENT_MODULE_PATH: src.program.baseline_rlm.rlm_pipeline.RLMPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This program is an agentic, multi-hop question-answering system over the **PhantomWiki** corpus — a fictional knowledge base of characters, relationships, occupations, and attributes. It uses a Retrieval Language Model (RLM) that iteratively searches for relevant passages and reasons over them to produce a list of answers.

### Key Modules and Responsibilities

- **`RLMPipeline`** (`src.program.baseline_rlm.rlm_pipeline`): Top-level `dspy.Module` and program entry point. Initializes the retrieval model (`CountingRM` wrapping `ColBERTv2`) and the core reasoning program (`PhantomWikiRLM`). Sets the DSPy retrieval context via `dspy.context(rm=self.rm)` before forwarding the question to the inner program.

- **`PhantomWikiRLM`** (`src.program.baseline_rlm.rlm_module`): Core reasoning module using `dspy.RLM` (an agentic LLM loop). Configured with `k=7` passages per retrieval call, up to 15 reasoning iterations, and a cap of 50 LLM calls. Exposes a `search_wiki(query)` tool that the RLM calls autonomously during reasoning to retrieve relevant passages. Produces `answer: list[str]`.

- **`CountingRM`** (`src.program.counting_rm`): A thin instrumentation wrapper around the `ColBERTv2` retriever that tracks the total number of retrieval calls via `call_count`, enabling monitoring of retrieval usage.

### Data Flow
1. **Input**: A natural-language `question` string is passed to `RLMPipeline.forward`.
2. **Context Setup**: The pipeline installs `CountingRM(ColBERTv2)` as the active retriever.
3. **Agentic Retrieval Loop**: `PhantomWikiRLM` hands the question to `dspy.RLM`, which iteratively calls `search_wiki(query)` — firing ColBERTv2 to fetch top-k passages — and reasons over the returned text until it converges on an answer or hits iteration/LLM-call limits.
4. **Output**: A `dspy.Prediction(answer=list[str])` is returned.

### Metric Being Optimized
`phantomwiki_f1_feedback` computes token-set **F1** between the predicted and gold answer lists (after lowercasing and stripping). It returns a `ScoreWithFeedback` object containing the numeric F1 score plus a detailed natural-language feedback string listing correct, missed, and extra answers — enabling gradient-free GEPA-style prompt optimization that uses textual critique signals.

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

