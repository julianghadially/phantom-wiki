PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "Iterative Expansion Architecture: Four-Phase ReAct with AnswerExpansionQuery and AnswerMerger"

## ARCHITECTURE SUMMARY:
`PhantomWikiReActPipeline` is a DSPy-based question-answering pipeline that answers factoid (multi-hop) questions by iteratively searching a PhantomWiki corpus using a four-phase Iterative Expansion Architecture. The pipeline is optimized to maximize answer F1 score against gold answer sets, with a particular focus on improving recall for multi-entity questions where a single ReAct pass may miss answers.

The architecture extends the single-chain approach with three additional phases: a completeness check (`AnswerExpansionQuery` via `dspy.ChainOfThought`) that decides whether initial answers look incomplete or failed, a targeted expansion pass using an independent `PhantomWikiReAct` instance (`program_expand`) that explicitly names already-found answers and searches from a different angle, and a final answer merge (`AnswerMerger` via `dspy.ChainOfThought`) that unions both answer sets, discards failure phrases, and deduplicates. Counting questions skip expansion entirely and directly return the initial integer answer.

The `ExhaustiveMultiHopQA` signature (in `PhantomWikiReAct`) continues to guide each ReAct chain to be exhaustive and precise — searching with multiple phrasings, never returning "Cannot be determined", and formatting counts as plain integers. The expansion layer adds a second chance to recover missed answers or retry failed attribute lookups from a rephrased retrieval angle.

## ARCHITECTURE DESCRIPTION:
`PhantomWikiReActPipeline` is the top-level entry point implementing a four-phase Iterative Expansion Architecture. Phase 1 runs the primary `PhantomWikiReAct` chain (`self.program`) on the original question via `dspy.ReAct` with `ExhaustiveMultiHopQA` (up to 50 iterations, k=7 retrieval). Phase 2 runs `self.expander` (`dspy.ChainOfThought(AnswerExpansionQuery)`) to inspect the initial answers and decide whether expansion is needed — triggered when answers look like a partial subset, contain failure phrases ("Unknown", "Cannot be determined", etc.), or the question asks for all members of a category; skipped for counting questions. Phase 3 conditionally runs `self.program_expand` (a second independent `PhantomWikiReAct` instance) on the follow-up question crafted by `AnswerExpansionQuery`, which explicitly names already-found answers and rephrases the retrieval angle. Phase 4 runs `self.merger` (`dspy.ChainOfThought(AnswerMerger)`) to union both answer lists, discard failure phrases unless both sets failed, deduplicate, and for counting questions prefer the larger integer. The `CountingRM`-wrapped ColBERT retriever is shared across both ReAct chains via `dspy.context`. The metric `phantomwiki_f1_feedback` computes token-set F1 with textual feedback for the GEPA optimizer.

## High-Level Purpose
`PhantomWikiReActPipeline` is a DSPy-based question-answering pipeline that answers factoid (multi-hop) questions by iteratively searching a PhantomWiki corpus. The four-phase Iterative Expansion Architecture improves recall by running a second targeted retrieval pass when the initial answers appear incomplete or failed, then merging both result sets into a comprehensive deduplicated answer.

## Key Modules and Responsibilities

- **`PhantomWikiReActPipeline`** (`src/program/phantomwiki_pipeline.py`): Top-level DSPy `Module` and entry point. Instantiates a `CountingRM`-wrapped retriever, two `PhantomWikiReAct` sub-programs (`program` and `program_expand`), and two `dspy.ChainOfThought` modules (`expander` and `merger`). Orchestrates the four-phase forward pass.

- **`AnswerExpansionQuery`** (`src/program/phantomwiki_pipeline.py`): DSPy Signature for the completeness check. Takes the original question and partial answers, outputs `needs_expansion: bool` and `followup_question: str`. Decides expansion is needed for partial subsets, failure phrases, or category enumeration questions; skips counting questions.

- **`AnswerMerger`** (`src/program/phantomwiki_pipeline.py`): DSPy Signature for merging two answer lists. Takes the question and two string-repr answer lists, outputs a merged `answer: list[str]`. Unions distinct valid values, discards failure phrases (unless both lists fail), deduplicates, and for counting questions prefers the larger integer.

- **`CountingRM`** (`src/program/counting_rm.py`): A thin instrumentation wrapper around any DSPy retriever. Tracks how many retrieval calls are made during a forward pass via `call_count`, enabling observability without altering retrieval behavior. Wraps a `dspy.ColBERTv2` instance pointed at a hosted ColBERT server.

- **`ExhaustiveMultiHopQA`** (`src/program/phantomwiki_module.py`): A class-based DSPy Signature whose docstring instructs the agent to exhaustively enumerate all entities matching an anchor attribute, never give up with "Cannot be determined", handle multiple people sharing an attribute, and format "How many" answers as plain integer strings.

- **`PhantomWikiReAct`** (`src/program/phantomwiki_module.py`): Core reasoning module. Uses `dspy.ReAct` with `signature=ExhaustiveMultiHopQA` and up to 50 iterations. Exposes a `search_wiki(query)` tool that retrieves the top-7 passages from the corpus and returns them as newline-separated text, allowing the LLM to iteratively search and reason.

- **`phantomwiki_f1_feedback`** (`src/metric/metric.py`): Evaluation metric used for optimization. Computes token-set F1 between predicted and gold answer lists after case/whitespace normalization. Returns a `ScoreWithFeedback` object (score ∈ [0,1] + human-readable feedback detailing correct, missed, and extra answers) for use with the GEPA optimizer.

## Data Flow
1. A `question` string is passed into `PhantomWikiReActPipeline.forward`.
2. **Phase 1**: The pipeline sets `CountingRM` as the active retriever via `dspy.context` and calls `self.program(question=question)`. `PhantomWikiReAct` runs up to 50 ReAct iterations, calling `search_wiki` to retrieve passages from the remote ColBERT index, producing an initial `answer: list[str]`.
3. **Phase 2**: `self.expander` (`dspy.ChainOfThought(AnswerExpansionQuery)`) inspects the initial answers. If `needs_expansion` is False (e.g., counting question or fully exhaustive answer), the result is returned immediately.
4. **Phase 3**: If `needs_expansion` is True, `self.program_expand` (a second independent `PhantomWikiReAct`) runs on `expansion.followup_question` — a rephrased query that names already-found answers and searches from a different angle.
5. **Phase 4**: `self.merger` (`dspy.ChainOfThought(AnswerMerger)`) merges both answer lists into a comprehensive, deduplicated final answer wrapped in a `dspy.Prediction`.
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

