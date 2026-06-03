I'll analyze the codebase by reading the relevant files.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy
from src.program.react_agent.ReActAgent import ReActAgent
from src.tools.search_tools import SearchTools

class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.search_tools = SearchTools()
        self.agent = ReActAgent(tools=self.search_tools.get_tools())
    
    def forward(self, question: str) -> dspy.Prediction:
        result = self.agent(question=question)
        return dspy.Prediction(answer=result.answer, trajectory=result.trajectory if hasattr(result, 'trajectory') else None)
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/phantomwiki_pipeline/**"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
src/program/phantomwiki_pipeline/__init__.py
</tool_call>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/**"}}
</tool_call>
<tool_response>
src/program/react_agent/ReActAgent.py
src/program/react_agent/__init__.py
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
src/program/phantomwiki_pipeline/__init__.py
src/program/__init__.py
</tool_call>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/react_agent/ReActAgent.py"}}
</tool_call>
<tool_response>
import dspy

class ReActAgent(dspy.Module):
    def __init__(self, tools: list):
        self.agent = dspy.ReAct(
            dspy.Signature(
                "question -> answer",
                "Answer the question using the available tools. Be concise and direct in your final answer."
            ),
            tools=tools,
            max_iters=10
        )
    
    def forward(self, question: str) -> dspy.Prediction:
        result = self.agent(question=question)
        return dspy.Prediction(answer=result.answer, trajectory=result.trajectory if hasattr(result, 'trajectory') else None)
</tool_call>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/tools/**"}}
</tool_call>
<tool_response>
src/tools/search_tools.py
src/tools/__init__.py
</tool_call>

<tool_call>
{"name": "Read", "arguments": {"path": "src/tools/search_tools.py"}}
</tool_call>
<tool_response>
import dspy
import os
import requests
from typing import Optional

class SearchTools:
    def __init__(self):
        self.base_url = os.getenv("SEARCH_API_URL", "http://localhost:8080")
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia articles for relevant information."""
        try:
            response = requests.get(f"{self.base_url}/search", params={"q": query, "limit": 3})
            response.raise_for_status()
            results = response.json()
            if not results:
                return "No results found."
            formatted = []
            for r in results:
                title = r.get("title", "Unknown")
                content = r.get("content", "No content available")[:500]
                formatted.append(f"Title: {title}\nContent: {content}")
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def lookup_article(self, title: str) -> str:
        """Look up a specific Wikipedia article by title."""
        try:
            response = requests.get(f"{self.base_url}/article", params={"title": title})
            response.raise_for_status()
            result = response.json()
            content = result.get("content", "Article not found")
            return content[:2000]
        except Exception as e:
            return f"Lookup failed: {str(e)}"
    
    def get_tools(self) -> list:
        return [self.search_wikipedia, self.lookup_article]
</tool_call>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import re
import string
from typing import Union
import dspy

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> list:
    """Get tokens from a string."""
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = sum(min(prediction_tokens.count(t), ground_truth_tokens.count(t)) for t in common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def phantomwiki_f1_feedback(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> Union[float, bool]:
    """
    Compute F1 feedback for PhantomWiki question answering.
    
    Args:
        example: DSPy example with 'answer' field as ground truth
        prediction: DSPy prediction with 'answer' field
        trace: Optional trace for optimization
    
    Returns:
        Float F1 score (0.0 to 1.0) during evaluation, bool during optimization
    """
    ground_truth = example.answer if hasattr(example, 'answer') else ""
    predicted = prediction.answer if hasattr(prediction, 'answer') else ""
    
    if not ground_truth or not predicted:
        return 0.0 if trace is None else False
    
    f1_score = compute_f1(str(predicted), str(ground_truth))
    
    if trace is not None:
        return f1_score >= 0.5
    
    return f1_score
</tool_call>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Pipeline

A DSPy-based question answering system that uses the ReAct (Reasoning + Acting) paradigm to answer questions about a synthetic Wikipedia-like knowledge base (PhantomWiki).

## Overview

This pipeline uses DSPy's ReAct module to iteratively reason and search through a PhantomWiki knowledge base to answer complex questions. The system can perform multiple search iterations to find relevant information before producing a final answer.

## Architecture

- **PhantomWikiReActPipeline**: Main pipeline entry point that orchestrates the ReAct agent and search tools
- **ReActAgent**: Wraps DSPy's built-in ReAct module with a question → answer signature, allowing up to 10 iterations
- **SearchTools**: Provides two tools for interacting with a search API:
  - `search_wikipedia`: Performs keyword search returning top-3 results (truncated to 500 chars each)
  - `lookup_article`: Fetches a full article by title (truncated to 2000 chars)

## Metric

The system is optimized using token-level F1 score (`phantomwiki_f1_feedback`):
- Answers are normalized (lowercased, punctuation removed, articles removed)
- Token overlap between predicted and ground truth answers is computed
- During optimization: binary signal (F1 ≥ 0.5 = success)
- During evaluation: continuous F1 score (0.0–1.0)

## Setup

Set `SEARCH_API_URL` environment variable to point to your PhantomWiki search API (default: `http://localhost:8080`).
</tool_call>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: PhantomWiki ReAct QA Pipeline with DSPy ReAct Agent and Search Tool Integration

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question answering pipeline targeting a synthetic Wikipedia-like knowledge base (PhantomWiki). It chains three modules: `PhantomWikiReActPipeline` (entry point, `src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`), `ReActAgent` (`src/program/react_agent/ReActAgent.py`), and `SearchTools` (`src/tools/search_tools.py`). Evaluation is driven by a token-level F1 metric defined in `src/metric/metric.py`.

The pipeline delegates all reasoning to a DSPy `ReAct` loop that iteratively decides whether to invoke search tools or produce a final answer. Search tools communicate with a configurable external REST API (`SEARCH_API_URL`) that exposes a PhantomWiki knowledge base.

Optimization uses a binary threshold on F1 (≥ 0.5) during training and a continuous F1 score during evaluation, both computed via normalized token overlap in `phantomwiki_f1_feedback`.

## ARCHITECTURE DESCRIPTION:
**Entry point** — `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`): A `dspy.Module` that instantiates `SearchTools` and `ReActAgent` on construction. Its `forward(question)` method calls the agent and returns a `dspy.Prediction` containing `answer` and an optional `trajectory`.

**ReActAgent** (`src/program/react_agent/ReActAgent.py`): Wraps DSPy's built-in `dspy.ReAct` with the signature `question -> answer` and a conciseness instruction. It runs up to 10 reasoning/acting iterations, each step allowing the LLM to either call a tool or emit a final answer. The trajectory of intermediate steps is forwarded through to the pipeline output.

**SearchTools** (`src/tools/search_tools.py`): Provides two callable tools registered with the ReAct agent:
- `search_wikipedia(query)`: GET `/search` on the configured API, returns the top-3 results with titles and up to 500 characters of content each.
- `lookup_article(title)`: GET `/article` on the configured API, returns up to 2000 characters of a specific article's content.
Both methods handle HTTP errors gracefully, returning error strings rather than raising exceptions. The API base URL defaults to `http://localhost:8080` and is overridable via the `SEARCH_API_URL` environment variable.

**Metric** (`src/metric/metric.py`) — `phantomwiki_f1_feedback`: Compares `prediction.answer` against `example.answer` using token-level F1. Normalization strips punctuation, articles (a/an/the), and lowercases both strings before splitting into tokens. Token overlap (accounting for duplicates) yields precision and recall, combined into an F1 score. When a DSPy `trace` is present (optimization mode), the function returns a boolean (`F1 ≥ 0.5`); otherwise it returns the raw float for evaluation metrics.

**Data flow**: A question string enters `PhantomWikiReActPipeline.forward` → passed to `ReActAgent.forward` → fed into `dspy.ReAct`, which iteratively calls `search_wikipedia` or `lookup_article` against the external API until it produces a final answer (up to 10 iterations) → `answer` (and trajectory) bubble back up through the agent and pipeline as a `dspy.Prediction` → `phantomwiki_f1_feedback` scores the answer against the ground truth label.
```
