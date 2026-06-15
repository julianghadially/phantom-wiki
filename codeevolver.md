PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Two-phase ReAct with Python-determined question_type (entity/attribute/count_answer/count_pivot)

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline that uses a two-phase ReAct architecture to answer multi-hop questions over PhantomWiki. The pipeline is organized into three layers: a pipeline wrapper (`PhantomWikiReActPipeline`) that owns infrastructure, a core reasoning module (`PhantomWikiReAct`) that CodeEvolver optimizes, and a metric module (`phantomwiki_f1_feedback`) that scores predictions against gold answers.

The key innovation is deterministic Python-based question classification (`_classify_question`) that eliminates the LLM misclassification failure mode from prior two-phase attempts. Questions are classified into four types — `entity`, `attribute`, `count_answer`, `count_pivot` — using pure lexical rules. Phase 1 (`EntityFinderSig` via `dspy.ReAct`, max_iters=40) finds all terminal entities via multi-hop traversal. Phase 2 branches by type: for `attribute` questions, `AttributeComputerSig` (max_iters=35) looks up attributes for the named entities; for `count_pivot` questions, `CountComputerSig` (max_iters=15) is called once per pivot entity to count independently, preventing fan-out collapse. The LM is `openai/gpt-5.4-nano` with `reasoning_effort="low"`.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a natural language question about fictional PhantomWiki entities, the system classifies it deterministically, then runs a two-phase ReAct pipeline to produce a `list[str]` answer. Phase 1 isolates the entity-finding traversal from the answer computation, preventing intermediate-entity contamination. Phase 2 handles per-entity attribute lookup or per-pivot counting in separate calls, preventing multi-count fan-out collapse.

**Key modules and responsibilities:**

- `src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline` (the evaluator's entry point). Constructs the `CountingRM` retriever (wraps `dspy.ColBERTv2` pointed at `https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search`), the LM (`gpt-5.4-nano`, `reasoning_effort="low"`), and the `PhantomWikiReAct` module. Its `forward(question)` sets the LM and RM into DSPy's thread-local context and delegates to the module.

- `src/program/phantomwiki_module.py` — `PhantomWikiReAct` (the file CodeEvolver mutates). Contains three DSPy Signatures and the main module: `EntityFinderSig` (Phase 1, finds all terminal entities), `AttributeComputerSig` (Phase 2a, looks up attributes for named entities), and `CountComputerSig` (Phase 2b, counts X for a single pivot entity). The module-level `_classify_question(question)` function deterministically routes questions using lexical rules: `how many` + `does/did/do the ...` → `count_pivot`; `how many` otherwise → `count_answer`; `what is/are/was/were` → `attribute`; everything else → `entity`. Three tools are exposed: `search_wiki` (k=10), `search_wiki_broad` (k=50), and `search_by_date_exact` (pre-built JSON index for DOB-anchored questions). `forward()` dispatches to the correct phase-2 handler or short-circuits for `entity` (return entities directly) and `count_answer` (return `len(entities)`).

- `src/program/counting_rm.py` — `CountingRM`. A `dspy.Retrieve` subclass that wraps any retriever, increments a call counter on each retrieval, monkey-patches the ColBERTv2 HTTP client to enforce a 60-second timeout, and retries up to 2 times on timeout/connection errors. Used to track retrieval cost per question.

- `src/metric/metric.py` — Two metrics: `phantomwiki_f1` (plain float F1 for the evaluation harness) and `phantomwiki_f1_feedback` (returns `dspy.teleprompt.gepa.ScoreWithFeedback` with score + human-readable breakdown of correct/missed/extra answers). Normalization is lowercase + strip; scoring is set-based precision/recall/F1 over answer lists.

**Data flow:** `evaluate.py` loads JSON question splits from `output/depth_10_size_1000000/`, calls `PhantomWikiReActPipeline.forward(question)` per row, and scores the returned `dspy.Prediction.answer` against the gold `answer` field using `phantomwiki_f1`. CodeEvolver uses `phantomwiki_f1_feedback` during optimization for richer gradient signal.

**Metric being optimized:** Token-level F1 between the predicted answer set and the gold answer set (both normalized to lowercase). A score of 1.0 requires exactly matching the complete set of correct answers — no more, no less.
