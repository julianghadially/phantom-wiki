PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Two-phase ReAct (EntityFinder + AnswerComputer) with DOB exact index and broad search

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline that uses a two-phase ReAct architecture to answer multi-hop questions over PhantomWiki — a benchmark corpus of entirely fictional characters and facts. The pipeline separates entity traversal (Phase 1: `EntityFinderSig`) from answer computation (Phase 2: `AnswerComputerSig`), directly addressing multi-count fan-out and attribute contamination bugs in the prior single-agent design.

Phase 1 (`entity_finder`, `EntityFinderSig`) is a traversal specialist that identifies ALL person names at the end of the relationship chain in a question using up to 35 ReAct iterations. It has access to three tools: `search_wiki` (k=10), `search_wiki_broad` (k=30), and `search_by_date_exact` (exact DOB index lookup). Phase 2 (`answer_computer`, `AnswerComputerSig`) receives the full list of traversed entities and computes final answers (attributes, counts, or entity names) for every entity independently, using up to 20 ReAct iterations with `search_wiki` and `search_wiki_broad`. Infrastructure (ColBERTv2 retriever, LM, CountingRM) is owned by `PhantomWikiReActPipeline`.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a natural language question about fictional PhantomWiki entities, the system runs two sequential ReAct agents to produce a `list[str]` answer. Questions can require multi-hop, multi-branch reasoning (e.g., "how many siblings does the child of X have?").

**Key modules and responsibilities:**

- `src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline` (the evaluator's entry point). Constructs the `CountingRM` retriever (wraps `dspy.ColBERTv2`), the LM (`gpt-5.4-nano`, `reasoning_effort="low"`), and the `PhantomWikiReAct` module. Its `forward(question)` sets the LM and RM into DSPy's thread-local context and delegates to the module.

- `src/program/phantomwiki_module.py` — `PhantomWikiReAct` (the file CodeEvolver mutates). Contains two signature classes and two ReAct sub-modules: `EntityFinderSig`/`entity_finder` traverses the relationship chain to find ALL intermediate person names (max 35 iters, 3 tools including DOB exact index); `AnswerComputerSig`/`answer_computer` receives those entities and computes final attributes, counts, or entity lists for each one independently (max 20 iters, 2 tools). A module-level `_load_date_index()` function caches a JSON map from YYYY-MM-DD date strings to article texts for exact DOB lookup.

- `src/program/counting_rm.py` — `CountingRM`. A `dspy.Retrieve` subclass that wraps any retriever, increments a call counter, enforces a 60-second timeout, and retries up to 2 times on errors.

- `src/metric/metric.py` — `phantomwiki_f1` (float F1) and `phantomwiki_f1_feedback` (ScoreWithFeedback for GEPA). Set-based precision/recall/F1 over normalized answer lists.

**Data flow:** `evaluate.py` loads JSON question splits, calls `PhantomWikiReActPipeline.forward(question)` per row. The pipeline delegates to `PhantomWikiReAct.forward()`, which runs Phase 1 (entity traversal) then Phase 2 (answer computation), returning `dspy.Prediction(answer=list[str])`. The answer is scored against the gold using `phantomwiki_f1`.

**Metric being optimized:** Token-level F1 between the predicted answer set and the gold answer set (both normalized to lowercase). A score of 1.0 requires exactly matching the complete set of correct answers — no more, no less.
