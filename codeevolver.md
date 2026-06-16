PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: DSPy ReAct Agent with Generation-Depth Counting and Second-Aunt/Uncle Fix — Multi-Hop QA Pipeline

## ARCHITECTURE SUMMARY:
The system is a multi-hop question-answering pipeline built on the DSPy framework, targeting the PhantomWiki benchmark — a synthetic fictional-universe QA dataset designed to resist leakage. The top-level entry point is `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`), which wires together a remote ColBERT retriever and the core reasoning module `PhantomWikiReAct` (`src/program/phantomwiki_module.py`). The pipeline delegates all forward passes to the reasoning module while setting the retriever and LM into DSPy's thread-local context.

The core module `PhantomWikiReAct` exposes a single `search_wiki` tool (k=20). The signature prompt adds a GENERATION DEPTH rule (counting "great-" prefixes to avoid stopping one generation short), expands the second-aunt/uncle definition to clarify it means TWO generations above X, and instructs the agent to try multiple date-of-birth phrasings via `search_wiki` rather than a dedicated DOB tool.

## ARCHITECTURE DESCRIPTION:
**What this program does**: Given a natural-language question about a fictional universe (PhantomWiki), the pipeline retrieves relevant passages from a large corpus via a remote ColBERT index and uses an LM to iteratively reason and search before returning a list of answer strings. Questions require multi-hop reasoning (e.g., "list all siblings of character X who also know character Y"), and answers are sets of strings compared with F1.

**Key modules and responsibilities**:
- `src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline`: Top-level `dspy.Module`. Owns the `CountingRM` retriever (wrapping `dspy.ColBERTv2` pointed at a remote Modal endpoint) and the `PhantomWikiReAct` module. On `forward(question)`, sets `lm` and `rm` into `dspy.context` then delegates. LM is `openai/gpt-5.4-nano` with `reasoning_effort="low"`.
- `src/program/phantomwiki_module.py` — `PhantomWikiSignature`: A class-based `dspy.Signature` with a 5-step system prompt instructing the agent to (1) decompose questions with a DIRECTION RULE for ancestor vs. descendant traversal and a GENERATION DEPTH rule for counting "great-" prefixes, (2) enumerate per-entity values without aggregating, (3) decompose derived relationships including an expanded second-aunt/uncle definition clarifying it means TWO generations above X, (4) search exhaustively using multiple phrasings including date-of-birth variants, and (5) match the answer type to the question.
- `src/program/phantomwiki_module.py` — `PhantomWikiReAct`: The **evolvable core**. Uses `dspy.ReAct` with `PhantomWikiSignature`, up to 50 tool-calling iterations, and a single tool: `search_wiki` (calls `dspy.Retrieve(k=20)`) for all queries including dates of birth.
- `src/program/counting_rm.py` — `CountingRM`: A `dspy.Retrieve` wrapper that counts retrieval calls, patches the ColBERT HTTP client to use a 60s timeout with 2 retries, and surfaces call counts in evaluation output.
- `src/metric/metric.py` — Two metrics: `phantomwiki_f1` (plain float, used by `evaluate.py`) and `phantomwiki_f1_feedback` (returns a `dspy.teleprompt.gepa.ScoreWithFeedback` with score + detailed textual breakdown of correct/missed/extra answers, used by CodeEvolver's GEPA optimizer). Both normalize answers via lowercase+strip and compute set-based precision/recall/F1.
- `src/evaluate.py` — Standalone evaluation harness. Loads JSON splits from `data/`, runs 4 parallel threads, accumulates per-question scores, and reports mean F1 broken down by difficulty level and total retrieval calls.

**Data flow**: Dataset row (`question`, `answer` list, `difficulty`) → `PhantomWikiReActPipeline.forward(question)` → `dspy.context(lm, rm)` → `PhantomWikiReAct.forward` → `dspy.ReAct` loop (guided by `PhantomWikiSignature`): LM generates thought+action, calls `search_wiki(query)` (top-20 passages), LM incorporates results, repeats up to 50× → `dspy.Prediction(answer=list[str])` → `phantomwiki_f1_feedback(output, answer)` → `ScoreWithFeedback(score, feedback)`.

**Metric**: Token-level set F1 between normalized predicted and gold answer lists. `phantomwiki_f1_feedback` additionally produces a human-readable string enumerating correct, missed, and extra answers for use in GEPA prompt optimization feedback loops.
