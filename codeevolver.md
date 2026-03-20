PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the PhantomWiki corpus. Given a natural-language question, it iteratively retrieves relevant passages and reasons over them to produce one or more answers. It is optimized using the GEPA framework (DSPy) with an F1-based feedback metric.

### Key Modules

**`PhantomWikiReActPipeline`** (entry point, `src/program/phantomwiki_pipeline.py`)
Top-level `dspy.Module` implementing a multi-strategy fan-out plus sequential decomposition architecture. On initialization it creates a `CountingRM`-wrapped `ColBERTv2` retriever, a `PhantomWikiReAct` reasoning module, a `ChainOfThought(GenerateSearchStrategies)` module for generating diverse search strategies, a `ChainOfThought(MergeAnswers)` module for deduplicating results, and a `ChainOfThought(DecomposeToSteps)` module for sequential multi-hop decomposition. The `forward` method first generates 3 diverse search strategies for the question, then runs `PhantomWikiReAct` for each strategy plus the original question (4 total runs) within the retrieval context. Next, inside the same retrieval context, it decomposes the question into ordered sub-questions via `DecomposeToSteps` and answers each step sequentially, augmenting each step with prior answers as context and accumulating results into `all_answers`. Finally, all answers are merged/deduplicated via `MergeAnswers`.

**`GenerateSearchStrategies`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for generating 3 diverse, independent search strategies/rephrased questions from the original question to ensure exhaustive answer coverage across different starting points in the knowledge graph.

**`MergeAnswers`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for merging and deduplicating candidate answers collected across multiple search strategies into a single comprehensive list of valid, distinct answers.

**`DecomposeToSteps`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for decomposing a complex multi-hop question into an ordered list of simpler sub-questions that must be answered sequentially, where each step can build on answers from prior steps.

**`CountingRM`** (`src/program/counting_rm.py`)
A thin instrumentation wrapper around any DSPy retriever. It proxies all retrieval calls to the underlying model while incrementing a `call_count` counter, enabling retrieval-usage diagnostics without altering retrieval behavior.

**`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`)
Core reasoning module built on `dspy.ReAct`. Uses the signature `question -> answer: list[str]` with up to 50 reasoning iterations. Its single registered tool, `search_wiki`, invokes `dspy.Retrieve(k=20)` to fetch the top-20 passages from the PhantomWiki corpus for a given query and returns them as a newline-separated string for the LLM to reason over.

### Data Flow
1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward`.
2. **Strategy generation**: `GenerateSearchStrategies` (via `ChainOfThought`) produces 3 diverse search strategies/rephrased questions.
3. **Fan-out execution**: For each of the 3 strategies plus the original question (4 total), `PhantomWikiReAct` runs inside a `dspy.context(rm=self.rm)` block. Each run performs up to 50 think-act cycles, with each `search_wiki` action querying ColBERTv2 through `CountingRM` for 20 passages.
4. **Sequential decomposition**: Still inside `dspy.context(rm=self.rm)`, `DecomposeToSteps` decomposes the question into an ordered list of sub-questions. Each sub-question is answered sequentially by `PhantomWikiReAct`, with prior answers accumulated and prepended as context (`"(Prior answers: ...)"`) to each subsequent step. Results extend `all_answers`.
5. **Answer collection**: All `answer` lists from the fan-out runs and sequential steps are concatenated into `all_answers`.
6. **Merge & deduplicate**: `MergeAnswers` (via `ChainOfThought`) deduplicates and filters `all_answers` into a final answer list.
7. **Output**: A `dspy.Prediction(answer=...)` with the merged, deduplicated answer list.

### Metric Being Optimized
`phantomwiki_f1_feedback` computes token-set F1 between the predicted answer list and the gold answer list (both normalized to lowercase). It returns a `ScoreWithFeedback` object containing the numeric F1 score (0–1) and a detailed textual breakdown of correct, missed, and extra answers, enabling GEPA's feedback-driven prompt optimization.

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

