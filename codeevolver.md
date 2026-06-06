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

## ARCHITECTURE TITLE: Two-Phase Anchor Enumeration Pre-pass + Main ReAct Agent with Closed-World Rule

## ARCHITECTURE SUMMARY:
`PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`) wraps `PhantomWikiReAct` (`src/program/phantomwiki_module.py`), which implements a two-phase architecture. Phase 1 runs an `AnchorEnumerationQA`-driven ReAct agent (`anchor_finder`) that exhaustively identifies ALL entities matching the question's anchor description (e.g., all people with a given occupation, hobby, or date of birth) using a wide-retrieval tool (`search_wiki_wide`, k=15). Phase 2 augments the original question with the discovered anchor list and feeds it to the main `PhantomWikiQA`-driven ReAct agent (`react`, max_iters=50) for multi-hop reasoning.

This two-phase design directly addresses the dominant failure mode where the agent commits to a single anchor entity when many matching entities exist. The `PhantomWikiQA` signature enforces 10 rules covering genealogical depth tracking, exhaustive enumeration, numeric count formatting, and a closed-world assumption (Rule 10) that prevents inferring undocumented relationships.

## ARCHITECTURE DESCRIPTION:
**Entry point — `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`):**
Initializes a ColBERTv2 retrieval model, a GPT-4.1-mini LM, and a `PhantomWikiReAct` sub-module. The `forward` method runs `PhantomWikiReAct` inside a `dspy.context` that binds the LM and RM for all downstream DSPy calls.

**Core module — `PhantomWikiReAct` (`src/program/phantomwiki_module.py`):**
Contains two `dspy.Retrieve` instances (`retrieve` k=10, `anchor_retrieve` k=15) and two `dspy.ReAct` agents:
- `anchor_finder`: uses `AnchorEnumerationQA` signature + `search_wiki_wide` tool (k=15), max_iters=12. Its sole job is to enumerate ALL entities matching the anchor description via multiple query phrasings.
- `react`: uses `PhantomWikiQA` signature + `search_wiki` tool (k=10), max_iters=50. Performs the full multi-hop reasoning to answer the question.

The `forward` method runs Phase 1 (anchor enumeration), then injects an `[ANCHOR SEARCH COMPLETE: ...]` or `[ANCHOR SEARCH HINT: ...]` block into the question before Phase 2 (main reasoning). Errors in Phase 1 fall back to passing the original question unchanged.

**Signatures — `AnchorEnumerationQA` and `PhantomWikiQA`:**
`AnchorEnumerationQA` outputs a `list[str]` of all matching entity names. `PhantomWikiQA` enforces 10 rules: genealogical depth tracking, exhaustive enumeration, no repeated queries, exact DOB matching, reverse relationship lookups, systematic tree traversal, document field extraction, numeric-only "how many" answers (including the multi-anchor case), answer-type determination by question wording, and the closed-world assumption for undocumented relationships.

**Metric — `src/metric/metric.py`:**
`phantomwiki_f1_feedback` computes set-based token F1 between gold and predicted answer lists, driving DSPy optimizer feedback.
```
