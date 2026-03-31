PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: DSPy ReAct Agent with ColBERTv2 Retrieval over PhantomWiki Corpus

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. The top-level entry point, `PhantomWikiReActPipeline`, composes two core components: a `CountingRM` retrieval wrapper around a remote ColBERTv2 index, and a `PhantomWikiReAct` reasoning module. The pipeline sets the DSPy retrieval context and delegates all question-answering logic to the inner ReAct agent.

`PhantomWikiReAct` implements the ReAct (Reasoning + Acting) agent pattern via `dspy.ReAct`, iteratively interleaving language model reasoning steps with calls to a `search_wiki` tool backed by the ColBERTv2 retriever. The agent is configured to produce a `list[str]` answer and may take up to 50 reasoning/action iterations per question.

The optimization metric, `phantomwiki_f1_feedback`, scores predictions using token-level set F1 between predicted and gold answer lists, and additionally returns structured textual feedback (correct, missed, and extra answers) compatible with the GEPA optimizer via DSPy's `ScoreWithFeedback`.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization, it instantiates a `CountingRM` wrapping a `dspy.ColBERTv2` retriever pointed at a remotely hosted PhantomWiki corpus (served via Modal). It also instantiates `PhantomWikiReAct` as the inner reasoning program. The `forward` method sets the active DSPy retrieval model (via `dspy.context(rm=self.rm)`) and calls the inner program with the incoming question.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the core ReAct agent. It holds a `dspy.Retrieve(k=7)` component and a `dspy.ReAct` agent with the signature `question -> answer: list[str]`, using `search_wiki` as its sole tool and allowing up to 50 agentic iterations. The `search_wiki` tool invokes the retriever and joins the top-7 retrieved passages into a single string for the LM to reason over. The `forward` method wraps the ReAct output into a `dspy.Prediction`.

**CountingRM** (`src/program/counting_rm.py`) is a lightweight `dspy.Retrieve` subclass that wraps any underlying retriever, incrementing a `call_count` on every retrieval invocation. This enables downstream monitoring or budget-aware analysis of retrieval usage.

**Metric — phantomwiki_f1_feedback** (`src/metric/metric.py`) evaluates model predictions with set-based F1: both gold and predicted answers are normalized (lowercased, stripped), converted to sets, and scored via precision/recall harmonic mean. The `phantomwiki_f1_feedback` variant also returns a `ScoreWithFeedback` object containing a human-readable summary of correct matches, missed answers, and spurious predictions, making it directly compatible with GEPA (Generative Evaluator Prompt Amplification) for optimizer-driven prompt refinement.

**Data flow**: A question string enters `PhantomWikiReActPipeline.forward` → the ReAct agent iteratively calls `search_wiki` (→ ColBERTv2 → top-7 passages) and reasons over retrieved text → produces a `list[str]` answer → evaluated against gold answers using F1 with structured feedback for optimization.

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

