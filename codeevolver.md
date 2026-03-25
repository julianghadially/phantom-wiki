PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "Exhaustive Multi-Hop ReAct with ExhaustiveMultiHopQA Signature"

## ARCHITECTURE SUMMARY:
`PhantomWikiReActPipeline` is a DSPy-based question-answering pipeline that answers factoid (multi-hop) questions by iteratively searching a PhantomWiki corpus using a ReAct (Reasoning + Acting) loop. The pipeline is optimized to maximize answer F1 score against gold answer sets.

The core improvement in this architecture is the introduction of the `ExhaustiveMultiHopQA` class-based DSPy Signature, which replaces the inline `"question -> answer: list[str]"` string. This signature provides explicit instructions to the agent to enumerate ALL entities matching an anchor attribute (not just the first), issue multiple search phrasings until no new results appear, never return "Cannot be determined", process all people sharing an attribute without asking for clarification, and return plain integer strings for "How many" questions.

The `answer` output field carries a description reinforcing these formatting rules. Together, these changes guide the LLM to be more exhaustive and precise in its multi-hop reasoning, improving recall over multi-entity questions.

## ARCHITECTURE DESCRIPTION:
`PhantomWikiReActPipeline` is the top-level entry point. It instantiates a `CountingRM`-wrapped ColBERT retriever and delegates all question-answering to `PhantomWikiReAct`. The `PhantomWikiReAct` module uses `dspy.ReAct` with the new `ExhaustiveMultiHopQA` signature (up to 50 iterations, k=7 retrieval). `ExhaustiveMultiHopQA` is a class-based DSPy Signature whose docstring instructs the agent to: (1) identify the anchor attribute in the question, (2) search exhaustively with multiple phrasings until two consecutive searches return no new entities, (3) never return "Cannot be determined" — always try alternate queries, (4) process all people sharing an attribute without asking for clarification, and (5) for "How many" questions return only a plain integer string (e.g., ["3"]). The `answer` output field description reinforces these rules. The `search_wiki` tool retrieves the top-7 passages from the remote ColBERT index and returns them as newline-separated text for the agent to reason over. The evaluation metric `phantomwiki_f1_feedback` computes token-set F1 between predicted and gold answer lists.

## High-Level Purpose
`PhantomWikiReActPipeline` is a DSPy-based question-answering pipeline that answers factoid (multi-hop) questions by iteratively searching a PhantomWiki corpus using a ReAct (Reasoning + Acting) loop. The pipeline is optimized to maximize answer F1 score against gold answer sets.

## Key Modules and Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level DSPy `Module` and entry point. Instantiates a `CountingRM`-wrapped retriever and a `PhantomWikiReAct` sub-program. Sets the retriever in DSPy context before delegating to the sub-program.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin instrumentation wrapper around any DSPy retriever. Tracks how many retrieval calls are made during a forward pass via `call_count`, enabling observability without altering retrieval behavior. Wraps a `dspy.ColBERTv2` instance pointed at a hosted ColBERT server.

- **`ExhaustiveMultiHopQA`** (`src/program/phantomwiki_module.py`): A class-based DSPy Signature replacing the old inline string signature. Its docstring instructs the agent to exhaustively enumerate all entities matching an anchor attribute, never give up with "Cannot be determined", handle multiple people sharing an attribute, and format "How many" answers as plain integer strings.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with `signature=ExhaustiveMultiHopQA` and up to 50 iterations. Exposes a `search_wiki(query)` tool that retrieves the top-7 passages from the corpus and returns them as newline-separated text, allowing the LLM to iteratively search and reason.

- **`phantomwiki_f1_feedback`** (`src/metric/metric.py`): Evaluation metric used for optimization. Computes token-set F1 between predicted and gold answer lists after case/whitespace normalization. Returns a `ScoreWithFeedback` object (score ∈ [0,1] + human-readable feedback detailing correct, missed, and extra answers) for use with the GEPA optimizer.

## Data Flow
1. A `question` string is passed into `PhantomWikiReActPipeline.forward`.
2. The pipeline sets `CountingRM` as the active retriever via `dspy.context`.
3. `PhantomWikiReAct.forward` invokes `dspy.ReAct`, which iteratively calls `search_wiki` to retrieve passages from the remote ColBERT index, reasoning over them step-by-step.
4. After up to 50 iterations, ReAct produces a final `answer: list[str]` prediction.
5. The prediction is wrapped in a `dspy.Prediction` and returned.
6. `phantomwiki_f1_feedback` scores the prediction against gold answers, returning an F1 score and textual feedback to guide prompt optimization.

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

