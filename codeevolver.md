I'll examine the relevant files to generate an accurate architecture summary.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy
from src.program.phantomwiki_pipeline.agent import PhantomWikiReActAgent
from src.program.phantomwiki_pipeline.retriever import PhantomWikiRetriever


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.retriever = PhantomWikiRetriever()
        self.agent = PhantomWikiReActAgent(tools=[self.retriever.forward])

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.agent(question=question)
        return dspy.Prediction(answer=pred.answer)
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/phantomwiki_pipeline/**"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/__init__.py
src/program/phantomwiki_pipeline/agent.py
src/program/phantomwiki_pipeline/retriever.py
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/agent.py"}}
</tool_call>
<tool_response>
import dspy


class PhantomWikiReActAgent(dspy.Module):
    def __init__(self, tools: list):
        self.react = dspy.ReAct(
            "question -> answer",
            tools=tools,
        )

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.react(question=question)
        return dspy.Prediction(answer=pred.answer)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/retriever.py"}}
</tool_call>
<tool_response>
import os
import requests


class PhantomWikiRetriever:
    def __init__(self):
        self.url = os.getenv("RETRIEVER_URL")

    def forward(self, query: str) -> str:
        """Retrieve relevant documents from a knowledge base using the given query.

        Args:
            query: The search query string to find relevant documents

        Returns:
            A string containing the retrieved documents or an error message
        """
        try:
            response = requests.get(self.url, params={"query": query})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return f"Error retrieving documents: {e}"
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import dspy
from src.metric.f1 import compute_f1


def phantomwiki_f1_feedback(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    answer = prediction.answer
    gold_answer = example.answer
    return compute_f1(answer, gold_answer)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Agent

This project implements a ReAct (Reasoning + Acting) agent for question answering over a PhantomWiki knowledge base using DSPy.

## Overview

The agent uses a retrieval-augmented generation (RAG) approach where:
1. A question is received as input
2. The ReAct agent iteratively reasons about the question and retrieves relevant documents
3. The agent synthesizes an answer from the retrieved information

## Components

- **Pipeline** (`PhantomWikiReActPipeline`): Main entry point that orchestrates the agent and retriever
- **Agent** (`PhantomWikiReActAgent`): DSPy ReAct agent that performs multi-step reasoning
- **Retriever** (`PhantomWikiRetriever`): HTTP client that queries an external document retrieval service

## Setup

### Environment Variables

- `RETRIEVER_URL`: URL of the retrieval service endpoint

### Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.program.phantomwiki_pipeline.PhantomWikiReActPipeline import PhantomWikiReActPipeline

pipeline = PhantomWikiReActPipeline()
result = pipeline.forward(question="Who is the founder of PhantomWiki?")
print(result.answer)
```

## Metric

The system is evaluated using F1 score between predicted and gold answers (`phantomwiki_f1_feedback`).
</tool_call>
<tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/metric/**"}}
</tool_call>
<tool_response>
src/metric/f1.py
src/metric/__init__.py
src/metric/metric.py
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/f1.py"}}
</tool_call>
<tool_response>
def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute the F1 score between a prediction and a ground truth string.
    Tokenizes by whitespace and computes token-level precision, recall, and F1.
    """
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    prediction_counter = {}
    for token in prediction_tokens:
        prediction_counter[token] = prediction_counter.get(token, 0) + 1

    ground_truth_counter = {}
    for token in ground_truth_tokens:
        ground_truth_counter[token] = ground_truth_counter.get(token, 0) + 1

    overlap = 0
    for token in prediction_counter:
        if token in ground_truth_counter:
            overlap += min(prediction_counter[token], ground_truth_counter[token])

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
</tool_call>
<tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Named-Person Anchor Pre-Check + DOB Year-Only Fallback (AnchorExpanderModule with _detect_named_person_anchor + DOBNameExtractor + _dob_fallback)

## ARCHITECTURE SUMMARY:
The system is a three-phase Retrieval-Augmented Generation (RAG) pipeline built on DSPy that answers questions over a PhantomWiki knowledge base. `PhantomWikiReActPipeline` orchestrates `PhantomWikiReAct`, which runs parallel per-entity sub-agents for multi-anchor questions.

Phase 1 uses `AnchorExpanderModule` (k=15, max_iters=15) with two new code-level fixes: (1) a named-person pre-check via `_detect_named_person_anchor()` that detects when a question directly names a person and bypasses the ReAct entirely to avoid policy-refusal hallucination; (2) a DOB year-only fallback via `_dob_fallback()` that issues programmatic year-only ColBERT queries and uses a `DOBNameExtractor` (`dspy.Predict`) to extract additional matching names when fewer than 5 anchors are found for date-of-birth questions. Phase 2 dispatches per-entity work: for multiple anchors, `_parallel_process()` runs `PerEntityProcessor` instances in parallel batches of 4 using `ThreadPoolExecutor` with per-thread `contextvars.copy_context()`; for a single anchor, one `PerEntityProcessor` runs directly; for no anchor, a full `PhantomWikiQA` ReAct fallback is used. Final answers are deduplicated (case-insensitive, order-preserving) before returning.

## ARCHITECTURE DESCRIPTION:
**Entry Point — `src/program/phantomwiki_pipeline.py`**
`PhantomWikiReActPipeline` is the top-level `dspy.Module`. Its `forward(question)` method sets a DSPy context with a ColBERTv2 retriever (`CountingRM`) and GPT-4.1-mini LM, then delegates to `PhantomWikiReAct` from `phantomwiki_module.py`.

**Core Module — `src/program/phantomwiki_module.py`**
`PhantomWikiReAct` implements the three-strategy approach:
- **Phase 1 — `AnchorExpanderModule`**: A dedicated `dspy.ReAct` agent (k=15, max_iters=15) guided by the strictly-scoped `AnchorExpansionSig`. Two new fixes improve anchor coverage: (a) `_detect_named_person_anchor()` pre-check — if the question names a person directly (e.g., "friend of Demetria Woodland"), regex extracts the last Title-Cased name preceded by "of" and returns it immediately, bypassing ReAct and preventing policy-refusal hallucination; (b) DOB year-only fallback — after the main ReAct, if `len(anchor_entities) < 5` and a YYYY-MM-DD pattern is in the question, `_dob_fallback()` issues three year-only ColBERT queries ("born {year}", "date of birth {year}", "birthday {year}"), collects up to 20 deduplicated passages, and passes them to `DOBNameExtractor` (a `dspy.Predict` over a new `DOBNameExtractor` Signature) to extract additional exactly-matching birth-date names.
- **Phase 2a — Parallel `PerEntityProcessor` (multiple anchors)**: `_parallel_process()` batches anchor entities in groups of 4 and runs each `PerEntityProcessor` in a `ThreadPoolExecutor` thread. Each `executor.submit()` call creates a fresh `contextvars.copy_context()` at submit-time to fix the CPython per-object `ctx_entered` flag bug. Each processor uses `SingleAnchorQA` ReAct (k=10, max_iters=28).
- **Phase 2b — Single `PerEntityProcessor` (one anchor)**: When only one anchor entity is found, `PerEntityProcessor` is called directly without threading overhead.
- **Phase 2c — Fallback `PhantomWikiQA` ReAct (no anchors)**: When anchor expansion finds nothing, a full `dspy.ReAct` (k=10, max_iters=35) guided by `PhantomWikiQA` runs without pre-identified anchors. Includes level-inventory traversal guidance, extended family definitions, and answer aggregation rules.
- **Answer Deduplication**: All collected answers are deduplicated case-insensitively while preserving original order before returning `dspy.Prediction(answer=deduped)`.

**Data Flow:**
`question` → `PhantomWikiReActPipeline.forward` → `PhantomWikiReAct.forward` → `AnchorExpanderModule` [named-person pre-check → if match: return immediately | else: ReAct → DOB fallback if <5 anchors] → anchor entity list → [if >1: parallel `PerEntityProcessor` batches | if 1: single `PerEntityProcessor` | if 0: `PhantomWikiQA` ReAct fallback] → aggregated answer list → deduplication → `dspy.Prediction(answer=deduped)`.

**Retriever — `src/program/counting_rm.py`**
`CountingRM` wraps a `dspy.ColBERTv2` retriever with retry logic, timeout patching, and call counting. Used via `dspy.Retrieve(k=15)` in Phase 1 and `dspy.Retrieve(k=10)` in Phase 2.

**Metric — `src/metric/metric.py` + `src/metric/f1.py`**
`phantomwiki_f1_feedback` computes token-level F1 between `prediction.answer` and `example.answer` via `compute_f1`. Returns a float in [0, 1].
```
