I'll analyze the codebase by examining the entry point file, related files, and README.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.phantomwiki_pipeline.react_agent import ReactAgent
from src.program.phantomwiki_pipeline.retrieval_tools import (
    get_article,
    search_articles,
)


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.agent = ReactAgent(tools=[get_article, search_articles])

    def forward(self, question: str) -> dspy.Prediction:
        result = self.agent(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/phantomwiki_pipeline/**/*"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/react_agent.py
src/program/phantomwiki_pipeline/retrieval_tools.py
src/program/phantomwiki_pipeline/__init__.py
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/react_agent.py"}}
</tool_call>
<tool_response>
import dspy


class ReactAgent(dspy.Module):
    def __init__(self, tools: list):
        self.react = dspy.ReAct(
            dspy.Signature(
                "question -> answer",
                "You are a helpful AI research assistant. You are given a question and need to answer it. You have access to tools to help you answer the question.",
            ),
            tools=tools,
            max_iters=10,
        )

    def forward(self, question: str) -> dspy.Prediction:
        result = self.react(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/retrieval_tools.py"}}
</tool_call>
<tool_response>
import os

import requests


def get_article(title: str) -> str:
    """Retrieve the full article content for a given title.

    Args:
        title: The exact title of the article to retrieve.

    Returns:
        The full text content of the article.
    """
    base_url = os.environ.get("PHANTOMWIKI_URL", "http://localhost:7000")
    response = requests.get(f"{base_url}/article", params={"title": title})
    return response.text


def search_articles(query: str) -> str:
    """Search for articles matching a query string.

    Args:
        query: A search query string to find relevant articles.

    Returns:
        A list of article titles and snippets matching the query.
    """
    base_url = os.environ.get("PHANTOMWIKI_URL", "http://localhost:7000")
    response = requests.get(f"{base_url}/search", params={"query": query})
    return response.text
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import dspy


def phantomwiki_f1_feedback(example, pred, trace=None):
    gold_answer = example.answer
    predicted_answer = pred.answer

    em = dspy.evaluate.answer_exact_match(example, pred)

    # compute f1 score
    gold_tokens = set(gold_answer.lower().split())
    pred_tokens = set(predicted_answer.lower().split())
    common_tokens = gold_tokens & pred_tokens
    if len(common_tokens) == 0:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

    if trace is not None:
        return em
    return f1
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Pipeline

This is a DSPy-based pipeline for answering questions using PhantomWiki, a synthetic knowledge base.

## Overview

The pipeline uses a ReAct (Reasoning + Acting) agent to answer questions by iteratively searching and retrieving articles from a PhantomWiki instance.

## Architecture

```
PhantomWikiReActPipeline (dspy.Module)
└── ReactAgent (dspy.Module)
    └── dspy.ReAct
        ├── get_article (tool)
        └── search_articles (tool)
```

## Components

- **PhantomWikiReActPipeline**: Top-level DSPy module that serves as the entry point
- **ReactAgent**: Wraps dspy.ReAct with a question->answer signature and up to 10 iterations
- **retrieval_tools**: HTTP client functions for the PhantomWiki REST API
  - `get_article(title)`: Fetches full article by exact title
  - `search_articles(query)`: Searches articles by query string

## Configuration

Set `PHANTOMWIKI_URL` environment variable to point to your PhantomWiki instance (default: `http://localhost:7000`).

## Metric

The pipeline is evaluated using `phantomwiki_f1_feedback`:
- Returns token-level F1 score during training/evaluation
- Returns exact match (EM) during tracing/optimization
</tool_call>
<tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Reflexive Shortcut, Passage-Relevance Candidate Ordering, Mini-Pattern G per Entity, Breadth Cap 6, and Fixed Discovery Queries

## ARCHITECTURE SUMMARY:
The pipeline routes questions through three tiers. First, a zero-cost reflexive shortcut catches self-referential questions ("What is X of the person whose X is Y?") and returns the embedded answer immediately. Named-entity questions (Path A) go directly to the main `PhantomWikiReAct` agent. Filter-condition questions (Path B) now use improved entity discovery: up to 5 ColBERT queries (fixed attr_match regex stops before trailing verbs, adds standalone value query), candidate names extracted in passage-relevance order (article subjects first, not alphabetically), and up to 6 per-entity `PhantomWikiEntityAgent` calls. Each entity agent result is individually post-processed with mini-Pattern G before aggregation, correctly handling "how many" count-vs-name mismatches per entity.

## ARCHITECTURE DESCRIPTION:
**Entry point** (`src/program/phantomwiki_pipeline.py` — `PhantomWikiReActPipeline`): Three-tier routing. (1) `_reflexive_shortcut()` detects "What is X of the person whose X is Y?" and returns [Y] with no LLM call. (2) `_has_named_entity()` checks for a proper First-Last name → Path A: delegates to `PhantomWikiReAct`. (3) Otherwise Path B: programmatic discovery, per-entity sub-agents, mini-Pattern G per entity, aggregation, and global `_postprocess()`. Both LLM paths return a deduplicated `dspy.Prediction(answer=list[str])`.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`): Unchanged. Uses `PhantomWikiSignature` with rich instructions. Backed by `dspy.Retrieve(k=7)` via ColBERT.

**PhantomWikiEntityAgent** (`src/program/phantomwiki_module.py`): Lightweight per-entity ReAct agent (15 iterations). Checks if the entity satisfies the filter condition and returns the answer value(s) or empty list.

**Entity discovery** (`_generate_discovery_queries` + `_extract_entity_names`): Generates up to 5 queries — original question, date variants ("born DATE", "date of birth DATE"), "attr value" phrase, and standalone value. The attr_match regex now uses a lookahead to stop before trailing verbs (have/has/do/does/etc.) preventing value over-capture. `_extract_entity_names` now orders candidates by passage relevance: the first proper name in each passage (likely the article subject) becomes a "primary" candidate; remaining names are "secondary". Primary names appear first, ensuring `candidates[:6]` picks the most ColBERT-relevant entities rather than alphabetically-first ones.

**Mini-Pattern G per entity**: Before aggregating each per-entity agent's output, the pipeline checks if it's a "how many" question. If the agent returned non-numeric names, they are counted and replaced with the numeric string. If it returned numeric answers, those are kept and stray names discarded. This prevents cross-entity miscounting.

**Post-processing** (`_postprocess`): Global deduplication. Pattern G applied at the Path A/fallback level; disabled for the per-entity path (mini-Pattern G already handled per entity).

**Metric** (`src/metric/metric.py` — `phantomwiki_f1_feedback`): Token-level F1 during evaluation; exact-match during DSPy optimization tracing.
```
