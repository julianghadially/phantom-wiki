I'll examine the relevant files to generate an accurate architecture summary.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.phantomwiki_pipeline.utils import create_react_agent, convert_answers_to_string
from src.tools.phantomwiki_tools import search_wikipedia, lookup_wikipedia


class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.agent = create_react_agent()

    def forward(self, question: str) -> dspy.Prediction:
        result = self.agent(question=question)
        answer = convert_answers_to_string(result.answer)
        return dspy.Prediction(answer=answer)
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/phantomwiki_pipeline/**"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/__init__.py
src/program/phantomwiki_pipeline/utils.py
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/utils.py"}}
</tool_call>
<tool_response>
import dspy

from src.tools.phantomwiki_tools import search_wikipedia, lookup_wikipedia


def create_react_agent() -> dspy.ReAct:
    return dspy.ReAct(
        signature="question -> answer",
        tools=[search_wikipedia, lookup_wikipedia],
        max_iters=10,
    )


def convert_answers_to_string(answer: str | list[str]) -> str:
    if isinstance(answer, list):
        return ", ".join(answer)
    return answer
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/tools/phantomwiki_tools.py"}}
</tool_call>
<tool_response>
import requests

PHANTOMWIKI_URL = "http://localhost:8899"


def search_wikipedia(query: str) -> str:
    """
    Searches a PhantomWiki (Wikipedia-like) knowledge base for articles that match the given query.

    :param query: A search query string used to find relevant articles.
    :return: A newline-separated string of article titles that match the search query.
    """
    params = {"query": query}
    response = requests.get(f"{PHANTOMWIKI_URL}/search", params=params)
    return response.text


def lookup_wikipedia(title: str) -> str:
    """
    Retrieves the full content of a specific article from the PhantomWiki (Wikipedia-like) knowledge base.

    :param query: The exact title of the article to look up.
    :return: The full text content of the article with the given title.
    """
    params = {"title": title}
    response = requests.get(f"{PHANTOMWIKI_URL}/lookup", params=params)
    return response.text
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import dspy

from src.metric.utils import extract_answer, f1_score


def phantomwiki_f1_feedback(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    gold_answer = extract_answer(example.answer)
    predicted_answer = extract_answer(prediction.answer)
    return f1_score(gold_answer, predicted_answer)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/utils.py"}}
</tool_call>
<tool_response>
def extract_answer(answer: str | list[str]) -> list[str]:
    if isinstance(answer, list):
        return [a.strip().lower() for a in answer]
    return [a.strip().lower() for a in answer.split(",")]


def f1_score(gold: list[str], predicted: list[str]) -> float:
    gold_set = set(gold)
    predicted_set = set(predicted)
    
    if not gold_set or not predicted_set:
        return 0.0
    
    tp = len(gold_set & predicted_set)
    precision = tp / len(predicted_set)
    recall = tp / len(gold_set)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki DSPy Agent

This repository demonstrates an implementation of a [DSPy](https://dspy.ai/) ReAct agent that answers questions using a PhantomWiki knowledge base.

PhantomWiki is a tool for generating unique, factual knowledge bases with associated question-answer datasets. It is useful for evaluating RAG and agentic systems.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the PhantomWiki server

Follow instructions at [PhantomWiki](https://github.com/kilian-group/phantom-wiki) to start the server locally.

By default, the server should be running at `http://localhost:8899`.

### 3. Run the tests
```bash
python -m pytest tests/
```

### 4. Run the optimizer
```bash
python -m src.optimize
```

## Architecture

The agent uses:
- **DSPy ReAct** for reasoning and action loops
- **PhantomWiki tools** (`search_wikipedia`, `lookup_wikipedia`) for knowledge retrieval
- **F1 score** as the optimization metric

The agent iteratively searches and looks up articles in the PhantomWiki knowledge base to answer multi-hop questions.
</tool_call>
<tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: DSPy ReAct Agent over PhantomWiki with Token-Level F1 Metric

## ARCHITECTURE SUMMARY:
`PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`) is a DSPy module that wraps a ReAct agent constructed in `src/program/phantomwiki_pipeline/utils.py`. The agent iteratively reasons and acts using two retrieval tools defined in `src/tools/phantomwiki_tools.py` to answer questions against a locally-running PhantomWiki knowledge base server.

The pipeline is evaluated with `phantomwiki_f1_feedback` (`src/metric/metric.py`), which computes a token-level F1 score between the predicted and gold answers via helpers in `src/metric/utils.py`.

## ARCHITECTURE DESCRIPTION:
**Entry point — `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`):**
On initialization, `__init__` calls `create_react_agent()` from `utils.py` to build a `dspy.ReAct` agent with the signature `question -> answer` and up to 10 reasoning iterations. The `forward` method accepts a `question` string, runs the ReAct agent, then normalizes the raw answer (which may be a list or a string) into a comma-separated string via `convert_answers_to_string` before returning a `dspy.Prediction`.

**Agent construction — `src/program/phantomwiki_pipeline/utils.py`:**
`create_react_agent()` instantiates `dspy.ReAct` with two tools and `max_iters=10`. `convert_answers_to_string` joins list answers with `", "` or passes a plain string through unchanged.

**Retrieval tools — `src/tools/phantomwiki_tools.py`:**
Two HTTP-based tools hit a PhantomWiki server at `http://localhost:8899`:
- `search_wikipedia(query)` — calls `/search` and returns a newline-separated list of matching article titles.
- `lookup_wikipedia(title)` — calls `/lookup` and returns the full text of a specific article.
The ReAct agent uses these tools in a multi-hop loop to gather evidence before producing a final answer.

**Metric — `src/metric/metric.py` + `src/metric/utils.py`:**
`phantomwiki_f1_feedback(example, prediction, trace)` extracts the gold and predicted answers via `extract_answer`, which lowercases and splits comma-separated values into token lists. It then computes a set-based F1 score: precision is true-positives / predicted tokens, recall is true-positives / gold tokens, and the harmonic mean is returned as a float in [0, 1]. This metric drives DSPy optimizer feedback.

**Data flow:** `question` (str) → `PhantomWikiReActPipeline.forward` → `dspy.ReAct` loop (search/lookup HTTP calls, up to 10 iterations) → raw `answer` → `convert_answers_to_string` → `dspy.Prediction(answer=str)` → `phantomwiki_f1_feedback` compares against `example.answer` via token F1.
```
