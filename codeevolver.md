PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: "3-Stage Pipeline: QuestionDecomposer → ReAct (dual-tool, k=15/5) → AnswerGapFinder with exhaustive gap-filling"

## ARCHITECTURE SUMMARY:

PhantomWikiReActPipeline implements a three-stage pipeline designed for exhaustive answer accumulation on multi-hop and enumeration questions over the PhantomWiki corpus. Stage 1 (QuestionDecomposer, dspy.ChainOfThought) classifies the question type ("enumeration", "multi_hop_traversal", or "single_entity"), identifies seed anchor entities, and generates a structured step-by-step search plan. Stage 2 (PhantomWikiReAct, dspy.ReAct with max_iters=40) uses the decomposed plan and anchor entities as additional context and iterates with two complementary retrieval tools: search_wiki (dspy.Retrieve k=15 for broad enumeration) and search_entity (dspy.Retrieve k=5 for targeted entity lookup). It outputs a candidate answer list and an exploration summary. Stage 3 (AnswerGapFinder, dspy.ChainOfThought) reviews the candidates, emits up to 5 missing_searches, and produces a final_answer list.

If missing_searches is non-empty, a gap-filling loop executes each query with dspy.Retrieve(k=15), appends newly discovered unique entities to the candidate list, augments the exploration summary, then re-runs AnswerGapFinder once more to finalise the answer. The entire pipeline runs inside a dspy.context(rm=self.rm) block so that CountingRM (wrapping ColBERTv2 via Modal) intercepts all retrieval calls for efficiency tracking.

## ARCHITECTURE DESCRIPTION:

PhantomWikiReActPipeline is the top-level DSPy pipeline for answering multi-hop and enumeration questions over PhantomWiki. It orchestrates three collaborating sub-modules, all wired inside forward().

**Stage 1 – QuestionDecomposer** (dspy.ChainOfThought over the QuestionDecomposer signature): Receives the raw question and outputs (a) question_type: one of "enumeration", "multi_hop_traversal", "single_entity"; (b) anchor_entities: the seed entities/attributes to search first; (c) search_plan: a natural-language step-by-step retrieval strategy.

**Stage 2 – PhantomWikiReAct** (dspy.ReAct, max_iters=40): Receives question, search_plan, and anchor_entities. Equipped with two tools — search_wiki(query) using dspy.Retrieve(k=15) for broad corpus sweep, and search_entity(entity_name) using dspy.Retrieve(k=5) for precise entity lookup. Outputs candidate_answers (list[str]) and exploration_summary (str).

**Stage 3 – AnswerGapFinder** (dspy.ChainOfThought over the AnswerGapFinder signature): Receives question, candidate_answers, and exploration_summary. Outputs missing_searches (list[str], up to 5 queries) and final_answer (list[str]). If missing_searches is non-empty, a gap-filling loop runs each query through dspy.Retrieve(k=15), appends unique entity candidates, then re-runs AnswerGapFinder to produce the definitive answer.

**CountingRM** wraps ColBERTv2 (hosted on Modal) and tracks retrieval call counts. The pipeline returns dspy.Prediction(answer=final_answer).

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
