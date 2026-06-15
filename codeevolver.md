PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Single-phase ReAct (PhantomWikiQA) with DOB exact index, broad search, ATTRIBUTE FAN-OUT, MULTI-ENTITY ENUMERATION, and HOW MANY FORMAT rule

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline that uses a unified single-phase ReAct architecture to answer multi-hop questions over PhantomWiki — a benchmark corpus of entirely fictional characters and facts. The pipeline reverts from the two-phase design (which caused counting regressions) to a single `PhantomWikiQA` ReAct agent with up to 50 iterations, incorporating all proven structural wins from iterations 5–8.

The single `react` module (`PhantomWikiQA`) handles both entity traversal and answer computation in one ReAct loop, using three tools: `search_wiki` (k=10), `search_wiki_broad` (k=30), and `search_by_date_exact` (exact DOB index). The signature carries comprehensive reasoning instructions: ATTRIBUTE FAN-OUT (≥5 query phrasings for attribute-anchored questions), MULTI-ENTITY ENUMERATION (bilateral ancestor search), HOW MANY FORMAT RULE (numeric count strings), IMPLICIT RELATIONSHIPS traversal, NON-STANDARD KINSHIP TERMS definitions, and a tautological DOB short-circuit. Infrastructure (ColBERTv2 retriever, LM, CountingRM) is owned by `PhantomWikiReActPipeline`.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a natural language question about fictional PhantomWiki entities, the system runs a single ReAct agent (up to 50 iterations) to produce a `list[str]` answer. Questions can require multi-hop, multi-branch reasoning (e.g., "how many siblings does the child of X have?").

**Key modules and responsibilities:**

- `src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline` (the evaluator's entry point). Constructs the `CountingRM` retriever (wraps `dspy.ColBERTv2`), the LM (`gpt-5.4-nano`, `reasoning_effort="low"`), and the `PhantomWikiReAct` module. Its `forward(question)` sets the LM and RM into DSPy's thread-local context and delegates to the module.

- `src/program/phantomwiki_module.py` — `PhantomWikiReAct` (the file CodeEvolver mutates). Contains one signature class (`PhantomWikiQA`) and one ReAct sub-module (`react`, max 50 iters, 3 tools). The signature encodes all proven prompt wins: HOW MANY FORMAT RULE (return numeric count strings, not names), MULTI-ENTITY ENUMERATION (bilateral ancestor search), ATTRIBUTE FAN-OUT (≥5 phrasings for occupation/hobby anchors), IMPLICIT RELATIONSHIPS (step-by-step traversal for cousin/nephew/uncle), NON-STANDARD KINSHIP TERMS definitions, DATE-OF-BIRTH ANCHOR (always call search_by_date_exact first for DOB-anchored questions), and a tautological DOB short-circuit. A module-level `_load_date_index()` caches a JSON map from YYYY-MM-DD date strings to article texts for exact DOB lookup.

- `src/program/counting_rm.py` — `CountingRM`. A `dspy.Retrieve` subclass that wraps any retriever, increments a call counter, enforces a 60-second timeout, and retries up to 2 times on errors.

- `src/metric/metric.py` — `phantomwiki_f1` (float F1) and `phantomwiki_f1_feedback` (ScoreWithFeedback for GEPA). Set-based precision/recall/F1 over normalized answer lists.

**Data flow:** `evaluate.py` loads JSON question splits, calls `PhantomWikiReActPipeline.forward(question)` per row. The pipeline delegates to `PhantomWikiReAct.forward()`, which runs the single `react` module and returns `dspy.Prediction(answer=list[str])`. The answer is scored against the gold using `phantomwiki_f1`.

**Metric being optimized:** Token-level F1 between the predicted answer set and the gold answer set (both normalized to lowercase). A score of 1.0 requires exactly matching the complete set of correct answers — no more, no less.
