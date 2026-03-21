PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## Architecture Summary

### High-Level Purpose
This system is a multi-hop question-answering pipeline over the PhantomWiki corpus. Given a natural-language question, it sequentially decomposes it into simpler sub-questions, resolves each one in order (with each answer feeding into the next), and then runs a final enriched pass to produce a comprehensive answer. It is optimized using the GEPA framework (DSPy) with an F1-based feedback metric.

### Key Modules

**`PhantomWikiReActPipeline`** (entry point, `src/program/phantomwiki_pipeline.py`)
Top-level `dspy.Module` implementing a sequential question decomposition pipeline. On initialization it creates a `CountingRM`-wrapped `ColBERTv2` retriever, a `PhantomWikiReAct` reasoning module, a `ChainOfThought(DecomposeQuestion)` module for breaking the question into ordered sub-questions, a `ChainOfThought(ResolveWithContext)` module for context-aware resolution, a `ChainOfThought(MergeAnswers)` module for deduplicating results, a `ChainOfThought(ExtractKnowledge)` module for extracting entity-relationship facts, and a `ChainOfThought(EnrichQuestion)` module for augmenting the final query with accumulated facts. The `forward` method first decomposes the question into 1–5 ordered single-hop sub-questions, resolves each sequentially (substituting "result from step N" placeholders with actual answers), then runs a final enriched pass on the original question, and merges all answers.

**`DecomposeQuestion`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for decomposing a complex multi-hop question into an ordered list of 1–5 simpler single-hop sub-questions. Later steps may reference "the result from step N" as a placeholder for the answer obtained at step N, enabling chained resolution.

**`ResolveWithContext`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for answering a sub-question given previously resolved context entities. Takes `sub_question: str`, `context_entities: list[str]`, and `original_question: str` as inputs and outputs `answers: list[str]`.

**`MergeAnswers`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for merging and deduplicating candidate answers collected across multiple steps into a single comprehensive list of valid, distinct answers.

**`ExtractKnowledge`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for extracting structured entity-relationship facts from a ReAct run's results. Takes `question`, `strategy`, and `react_answer: list[str]` as inputs and outputs `entity_facts: list[str]`, a structured list of discovered facts about named entities and their relationships.

**`EnrichQuestion`** (DSPy Signature, `src/program/phantomwiki_pipeline.py`)
Signature for augmenting the original question with accumulated facts/entities from prior steps. Takes `question: str` and `accumulated_facts: list[str]` as inputs and outputs `enriched_question: str`, enabling the final ReAct run to leverage all intermediate discoveries.

**`CountingRM`** (`src/program/counting_rm.py`)
A thin instrumentation wrapper around any DSPy retriever. It proxies all retrieval calls to the underlying model while incrementing a `call_count` counter, enabling retrieval-usage diagnostics without altering retrieval behavior.

**`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`)
Core reasoning module built on `dspy.ReAct`. Uses the signature `question -> answer: list[str]` with up to 50 reasoning iterations. Its single registered tool, `search_wiki`, invokes `dspy.Retrieve(k=7)` to fetch the top-7 passages from the PhantomWiki corpus for a given query and returns them as a newline-separated string for the LLM to reason over.

### Data Flow
1. **Input**: A `question` string is passed to `PhantomWikiReActPipeline.forward`.
2. **Decomposition**: `DecomposeQuestion` (via `ChainOfThought`) produces an ordered list of 1–5 simpler single-hop sub-questions, where later steps may reference "the result from step N" as a placeholder.
3. **Sequential resolution**: For each sub-question, any "result from step N" placeholders are substituted with actual resolved entities from that step; then `PhantomWikiReAct` runs inside a `dspy.context(rm=self.rm)` block. Answers are accumulated as `context_entities` for the next step.
4. **Final enriched pass**: After all sub-questions are resolved, `EnrichQuestion` enriches the original question with all discovered entities; `PhantomWikiReAct` runs once more on the enriched question.
5. **Answer collection**: All answers from sub-question steps and the final pass are concatenated into `all_answers`.
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

