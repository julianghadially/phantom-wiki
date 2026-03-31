PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: PhantomWiki ReAct Pipeline with ColBERTv2 Retrieval and F1 Feedback Metric

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki corpus. The top-level entry point, `PhantomWikiReActPipeline` (`src.program.phantomwiki_pipeline`), composes a retrieval module (`CountingRM` wrapping a remote `ColBERTv2` server) with a ReAct reasoning agent (`PhantomWikiReAct`) to answer open-domain, potentially multi-hop questions whose answers are lists of strings.

The reasoning core lives in `PhantomWikiReAct` (`src.program.phantomwiki_module`), which uses DSPy's `ReAct` with a `search_wiki` tool backed by `dspy.Retrieve` (k=7). The retriever is wrapped in `CountingRM` (`src.program.counting_rm`) to instrument call counts. Evaluation is driven by `phantomwiki_f1_feedback` (`src.metric.metric`), which computes token-set F1 between predicted and gold answer lists and attaches structured textual feedback for GEPA-based optimization.

## ARCHITECTURE DESCRIPTION:
**Program Flow:**
`PhantomWikiReActPipeline.forward(question)` sets a DSPy context with the `CountingRM` retriever, then delegates to `PhantomWikiReAct.forward(question)`. The ReAct agent iterates (up to 50 steps) between thinking and calling `search_wiki`, which invokes `dspy.Retrieve(k=7)` to fetch the top-7 passages from the remote ColBERTv2 index hosted at a Modal endpoint. The ReAct signature is `question -> answer: list[str]`, so the model is expected to produce a list of answer strings. The final `dspy.Prediction(answer=...)` is returned to the caller.

**Key Modules:**
- `PhantomWikiReActPipeline` (pipeline wrapper): sets retrieval context and delegates to the core module.
- `PhantomWikiReAct` (reasoning core): DSPy `ReAct` agent with `search_wiki` as its only tool and up to 50 reasoning iterations.
- `CountingRM` (instrumented retriever): thin wrapper over `dspy.ColBERTv2` that counts retrieval calls, useful for profiling and optimization auditing.
- `ColBERTv2` (retrieval backend): remote dense retrieval over the PhantomWiki corpus via a Modal-hosted ColBERT server.

**Metric — `phantomwiki_f1_feedback`:**
The metric normalizes (lowercase + strip) both predicted and gold answer lists, then computes token-set precision, recall, and harmonic-mean F1. It returns a `ScoreWithFeedback` object (from `dspy.teleprompt.gepa`) containing the numeric F1 score (0.0–1.0) and a human-readable string enumerating the gold answers, predicted answers, correct matches, missed answers, and spurious predictions. This structured feedback is consumed by GEPA (Gradient-free Evolutionary Prompt Adaptation) to guide prompt optimization without gradient information.

**Optimization Target:**
The pipeline is optimized to maximize answer-set F1 on PhantomWiki questions, balancing precision (avoiding extra wrong answers) and recall (finding all correct answers) across potentially multi-hop queries that require iterative retrieval and reasoning.

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

