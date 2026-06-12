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

## ARCHITECTURE TITLE: PhantomWiki ReAct QA Pipeline with DSPy ReAct Agent and HTTP Retrieval Tools

## ARCHITECTURE SUMMARY:
The pipeline is a DSPy-based question-answering system that uses a ReAct (Reasoning + Acting) agent to answer questions by iteratively querying a PhantomWiki knowledge base over HTTP. The top-level entry point (`PhantomWikiReActPipeline`) delegates to a `ReactAgent`, which wraps DSPy's built-in `dspy.ReAct` module equipped with two retrieval tools. The system is evaluated with a token-level F1 metric that falls back to exact match during optimization tracing.

All pipeline logic lives in `src/program/phantomwiki_pipeline/`: the orchestrator in `PhantomWikiReActPipeline.py`, the agent wrapper in `react_agent.py`, and the HTTP tool functions in `retrieval_tools.py`. The metric is defined in `src/metric/metric.py`.

## ARCHITECTURE DESCRIPTION:
**Entry point** (`src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`): A thin `dspy.Module` subclass. Its `forward(question: str)` method instantiates and delegates entirely to `ReactAgent`, passing the two retrieval tool functions at construction time.

**ReactAgent** (`src/program/phantomwiki_pipeline/react_agent.py`): Wraps `dspy.ReAct` with a simple `question -> answer` signature and an instructional system prompt describing the assistant role. The agent is allowed up to 10 reasoning/action iterations, enabling multi-hop retrieval over the knowledge base.

**Retrieval tools** (`src/program/phantomwiki_pipeline/retrieval_tools.py`): Two plain Python functions exposed as tools to the ReAct agent.
- `get_article(title: str)`: Issues a GET request to `{PHANTOMWIKI_URL}/article?title=<title>` and returns the full article text.
- `search_articles(query: str)`: Issues a GET request to `{PHANTOMWIKI_URL}/search?query=<query>` and returns matching article titles and snippets.
The base URL is read from the `PHANTOMWIKI_URL` environment variable (default: `http://localhost:7000`).

**Data flow**: An input `question` string enters `PhantomWikiReActPipeline.forward`, is passed to `ReactAgent.forward`, which drives `dspy.ReAct` through iterative thought–action–observation loops. In each loop iteration the LLM may call `search_articles` to discover relevant articles or `get_article` to fetch full content. After up to 10 iterations a final `answer` string is produced and returned as a `dspy.Prediction`.

**Metric** (`src/metric/metric.py` — `phantomwiki_f1_feedback`): Compares `example.answer` (gold) against `pred.answer` (predicted) using token-level F1 (set intersection over union of lowercased word tokens). During DSPy optimization tracing (`trace is not None`) it returns binary exact-match instead, guiding the optimizer toward fully correct answers while the continuous F1 signal drives evaluation scoring.
```
