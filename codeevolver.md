I'll analyze the codebase by examining the entry point file, related files, and README.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.react_agent import PhantomWikiReActAgent

class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.agent = PhantomWikiReActAgent()

    def forward(self, question):
        result = self.agent(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/**/*.py"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
src/program/react_agent/__init__.py
src/program/react_agent/PhantomWikiReActAgent.py
src/program/phantomwiki_pipeline/__init__.py
src/program/tools/__init__.py
src/program/tools/retrieve.py
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/react_agent/PhantomWikiReActAgent.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.tools.retrieve import retrieve

class AnswerQuestion(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

class PhantomWikiReActAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(AnswerQuestion, tools=[retrieve])

    def forward(self, question):
        result = self.agent(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/tools/retrieve.py"}}
</tool_call>
<tool_response>
import dspy

from src.retriever.retriever import PhantomWikiRetriever

retriever = PhantomWikiRetriever()

def retrieve(query: str) -> str:
    """Retrieve passages from a knowledge base given a query."""
    results = retriever(query)
    return results
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import dspy

def phantomwiki_f1_feedback(example, pred, trace=None):
    answer = example.answer
    prediction = pred.answer

    f1 = dspy.evaluate.answer_exact_match(example, pred)

    return f1
</tool_call>
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/retriever/**/*.py"}}
</tool_call>
<tool_response>
src/retriever/retriever.py
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/retriever/retriever.py"}}
</tool_call>
<tool_response>
import dspy

class PhantomWikiRetriever(dspy.Module):
    def __init__(self):
        self.retriever = dspy.Retrieve(k=3)

    def forward(self, query: str) -> str:
        results = self.retriever(query_or_queries=query)
        passages = "\n\n".join(results.passages)
        return passages
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Pipeline

This is a DSPy program for answering questions from the PhantomWiki dataset using a ReAct agent with retrieval capabilities.

## Overview

The pipeline uses a ReAct (Reasoning + Acting) agent to answer factoid questions by iteratively retrieving relevant passages from a knowledge base.

## Architecture

- **Pipeline**: `PhantomWikiReActPipeline` - Top-level DSPy module that wraps the agent
- **Agent**: `PhantomWikiReActAgent` - ReAct agent with retrieval tool
- **Tools**: `retrieve` - Retrieval function using `PhantomWikiRetriever`
- **Retriever**: `PhantomWikiRetriever` - DSPy retriever fetching top-3 passages

## Metric

The pipeline is evaluated using `phantomwiki_f1_feedback`, which currently computes exact match (via `dspy.evaluate.answer_exact_match`) between the predicted and gold answers.

## Dataset

The pipeline is designed for the PhantomWiki dataset, which contains factoid questions requiring multi-hop reasoning over a synthetic knowledge base.

</tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: PhantomWiki ReAct Pipeline with DSPy Retrieval-Augmented QA

## ARCHITECTURE SUMMARY:
The system is a DSPy-based Retrieval-Augmented Generation (RAG) pipeline designed to answer factoid questions from the PhantomWiki dataset. It follows a thin pipeline-wrapper pattern: `PhantomWikiReActPipeline` (in `src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`) delegates entirely to `PhantomWikiReActAgent` (in `src/program/react_agent/PhantomWikiReActAgent.py`), which runs a DSPy `ReAct` loop equipped with a retrieval tool.

The retrieval tool (`src/program/tools/retrieve.py`) calls `PhantomWikiRetriever` (`src/retriever/retriever.py`), a DSPy module that fetches the top-3 passages from a configured knowledge base and joins them into a single string. The metric (`src/metric/metric.py`) evaluates predictions using exact match scoring.

## ARCHITECTURE DESCRIPTION:
**Entry point:** `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`) is a minimal `dspy.Module` whose `forward(question)` method instantiates and calls `PhantomWikiReActAgent`.

**Agent layer:** `PhantomWikiReActAgent` (`src/program/react_agent/PhantomWikiReActAgent.py`) wraps `dspy.ReAct` with the `AnswerQuestion` signature (input: `question: str`; output: `answer: str`, expected 1–5 words). The ReAct loop iteratively reasons and decides whether to call the `retrieve` tool or produce a final answer.

**Tool layer:** The `retrieve` function (`src/program/tools/retrieve.py`) is a plain Python callable exposed to the ReAct agent. It instantiates a module-level `PhantomWikiRetriever` singleton and delegates queries to it, returning a concatenated string of passages.

**Retriever layer:** `PhantomWikiRetriever` (`src/retriever/retriever.py`) is a `dspy.Module` wrapping `dspy.Retrieve(k=3)`. Given a query string, it retrieves the top-3 passages from the configured DSPy retrieval backend and joins them with double newlines.

**Data flow:** A raw `question` string enters `PhantomWikiReActPipeline.forward` → passed to `PhantomWikiReActAgent.forward` → consumed by the `dspy.ReAct` loop, which may call `retrieve(query)` one or more times → each call fetches 3 passages via `PhantomWikiRetriever` → passages are returned to the ReAct loop as context → the loop produces a final short `answer` string.

**Metric:** `phantomwiki_f1_feedback` (`src/metric/metric.py`) compares `pred.answer` to `example.answer` using `dspy.evaluate.answer_exact_match`, returning a binary 0/1 score. Despite the "F1" name, the current implementation performs exact match evaluation. This metric drives optimization of the DSPy program (e.g., prompt tuning or few-shot example selection via a DSPy optimizer).
```
