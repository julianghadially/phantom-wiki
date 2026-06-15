PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: ReAct with PhantomWikiQA Signature (multi-answer guidance, k=10)

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline that uses a ReAct (Reasoning + Acting) agent to answer multi-hop questions over PhantomWiki — a benchmark corpus of entirely fictional characters and facts designed to resist data leakage. The pipeline is organized into three layers: a pipeline wrapper (`PhantomWikiReActPipeline`) that owns infrastructure, a core reasoning module (`PhantomWikiReAct`) that CodeEvolver optimizes, and a metric module (`phantomwiki_f1_feedback`) that scores predictions against gold answers with textual feedback for the GEPA optimizer.

The `PhantomWikiReAct` module now uses a structured `PhantomWikiQA(dspy.Signature)` class instead of a bare string signature. The signature includes a detailed system docstring that instructs the LM to exhaustively search for multiple answers, with field-level descriptions reinforcing the multi-answer objective. Retrieval k is increased from 7 to 10 for broader passage coverage per query. Retrieval is handled by `CountingRM`, a wrapper around a remote ColBERTv2 endpoint hosted on Modal. The language model is `openai/gpt-5.4-nano` configured with `reasoning_effort="low"`.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a natural language question about fictional PhantomWiki entities, the system iteratively searches a corpus and reasons over retrieved passages to produce a `list[str]` answer. Questions can require multi-hop, multi-branch reasoning (e.g., "list all siblings of X who are also friends of Y").

**Key modules and responsibilities:**

- `src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline` (the evaluator's entry point). Constructs the `CountingRM` retriever (wraps `dspy.ColBERTv2` pointed at `https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search`), the LM (`gpt-5.4-nano`, `reasoning_effort="low"`), and the `PhantomWikiReAct` module. Its `forward(question)` sets the LM and RM into DSPy's thread-local context and delegates to the module.

- `src/program/phantomwiki_module.py` — `PhantomWikiReAct` (the file CodeEvolver mutates). Contains `PhantomWikiQA(dspy.Signature)`, a structured signature with a detailed docstring guiding exhaustive multi-answer search, plus annotated `question` (InputField) and `answer: list[str]` (OutputField) fields. `PhantomWikiReAct` wraps `dspy.ReAct` with this signature class, a single `search_wiki` tool (calls `dspy.Retrieve(k=10)`), and `max_iters=50`. The `search_wiki` docstring encourages varied query angles. Each iteration the LM decides whether to call `search_wiki(query)` or emit a final answer list.

- `src/program/counting_rm.py` — `CountingRM`. A `dspy.Retrieve` subclass that wraps any retriever, increments a call counter on each retrieval, monkey-patches the ColBERTv2 HTTP client to enforce a 60-second timeout, and retries up to 2 times on timeout/connection errors. Used to track retrieval cost per question.

- `src/metric/metric.py` — Two metrics: `phantomwiki_f1` (plain float F1 for the evaluation harness) and `phantomwiki_f1_feedback` (returns `dspy.teleprompt.gepa.ScoreWithFeedback` with score + human-readable breakdown of correct/missed/extra answers). Normalization is lowercase + strip; scoring is set-based precision/recall/F1 over answer lists.

**Data flow:** `evaluate.py` loads JSON question splits from `output/depth_10_size_1000000/`, calls `PhantomWikiReActPipeline.forward(question)` per row, and scores the returned `dspy.Prediction.answer` against the gold `answer` field using `phantomwiki_f1`. CodeEvolver uses `phantomwiki_f1_feedback` during optimization for richer gradient signal.

**Metric being optimized:** Token-level F1 between the predicted answer set and the gold answer set (both normalized to lowercase). A score of 1.0 requires exactly matching the complete set of correct answers — no more, no less.
