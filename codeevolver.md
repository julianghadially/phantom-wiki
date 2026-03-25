PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "3-stage pipeline: QuestionDecomposer → retry-resilient PhantomWikiReAct → AnswerValidator"

## ARCHITECTURE SUMMARY:
`PhantomWikiReActPipeline` is a three-stage DSPy pipeline for factoid multi-hop question answering over a PhantomWiki corpus. The first stage (`QuestionDecomposer`) decomposes the input question into 2–3 diverse seed queries and extracts metadata (`is_multi_answer`, `answer_format_hint`) to guide downstream reasoning. The second stage (`PhantomWikiReAct`) runs a ReAct loop (up to 50 iterations) using a retry-resilient `search_wiki` tool that automatically retries up to 4 times with `time.sleep(2)` on transient ColBERT server failures. The third stage (`AnswerValidator`) cleans, de-duplicates, and validates the raw answer list against the format hint before returning the final prediction.

This design directly addresses three dominant failure modes: ~40% timeout-cascade failures (via Python-level retry in `search_wiki`), ~20% partial-answer failures (via `is_multi_answer` awareness passed into ReAct), and ~5% format failures (via `AnswerValidator` stripping context artifacts like "Person — attribute").

## ARCHITECTURE DESCRIPTION:
The pipeline is orchestrated by `PhantomWikiReActPipeline.forward()` which calls three sub-modules in sequence. First, `QuestionDecomposer` (a `dspy.ChainOfThought` on `QuestionDecomposerSignature`) receives the raw question and outputs: (1) `seed_queries` — 2–3 diverse phrasings of the root entity to improve retrieval coverage; (2) `is_multi_answer` — a boolean flag indicating whether the answer likely requires enumerating multiple entities; (3) `answer_format_hint` — a natural-language description of the expected answer format. The first seed query is used as the `question` passed to `PhantomWikiReAct`, together with `is_multi_answer` and `answer_format_hint` as additional input fields of `PhantomWikiReActSignature`. Inside `PhantomWikiReAct`, the `search_wiki(query)` tool wraps `dspy.Retrieve(k=7)` with a retry loop (up to 4 attempts, 2-second sleep between) so transient ColBERT timeouts are recovered at the Python level before the LLM sees a failure message. After ReAct produces a raw `answer: list[str]`, `AnswerValidator` (a `dspy.ChainOfThought` on `AnswerValidatorSignature`) strips accidentally included context from each answer item and outputs a corrected, de-duplicated answer list. The final `dspy.Prediction(answer=...)` is returned from `forward()`.

## Key Modules and Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level DSPy `Module` and entry point. Orchestrates the three-stage flow: QuestionDecomposer → PhantomWikiReAct → AnswerValidator.

- **`QuestionDecomposer`** (`src/program/phantomwiki_module.py`): `dspy.ChainOfThought` on `QuestionDecomposerSignature`. Decomposes the question into seed queries and metadata (`is_multi_answer`, `answer_format_hint`) to guide retrieval and answer validation.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with `PhantomWikiReActSignature` (`question, is_multi_answer: bool, answer_format_hint: str -> answer: list[str]`) and up to 50 iterations. The `search_wiki` tool retries up to 4 times with 2-second delays on failures or empty results.

- **`AnswerValidator`** (`src/program/phantomwiki_module.py`): `dspy.ChainOfThought` on `AnswerValidatorSignature`. Strips context artifacts from raw answers, de-duplicates, and validates completeness against the format hint.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin instrumentation wrapper around any DSPy retriever. Tracks how many retrieval calls are made during a forward pass via `call_count`, enabling observability without altering retrieval behavior. Wraps a `dspy.ColBERTv2` instance pointed at a hosted ColBERT server.

- **`phantomwiki_f1_feedback`** (`src/metric/metric.py`): Evaluation metric used for optimization. Computes token-set F1 between predicted and gold answer lists after case/whitespace normalization. Returns a `ScoreWithFeedback` object (score ∈ [0,1] + human-readable feedback detailing correct, missed, and extra answers) for use with the GEPA optimizer.

## Data Flow
1. A `question` string is passed into `PhantomWikiReActPipeline.forward`.
2. `QuestionDecomposer` decomposes the question into `seed_queries`, `is_multi_answer`, and `answer_format_hint`.
3. The first seed query and metadata are passed to `PhantomWikiReAct` inside a `dspy.context(rm=self.rm)` block.
4. `PhantomWikiReAct.forward` invokes `dspy.ReAct`, which iteratively calls `search_wiki` (with retry logic) to retrieve passages from the remote ColBERT index, reasoning over them step-by-step.
5. After up to 50 iterations, ReAct produces a raw `answer: list[str]`.
6. `AnswerValidator` cleans and de-duplicates the raw answer list.
7. The validated prediction is wrapped in a `dspy.Prediction` and returned.
8. `phantomwiki_f1_feedback` scores the prediction against gold answers, returning an F1 score and textual feedback to guide prompt optimization.

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

