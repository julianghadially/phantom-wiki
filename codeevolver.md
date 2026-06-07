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

## ARCHITECTURE TITLE: 4-Step Anchor-Exhaustion ReAct with Notes Tools (k=10)

## ARCHITECTURE SUMMARY:
The system is a DSPy-based Retrieval-Augmented Generation pipeline for answering multi-hop factoid questions from the PhantomWiki dataset. `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`) wraps `PhantomWikiReAct` (`src/program/phantomwiki_module.py`), which implements a structured 4-step anchor-exhaustion reasoning protocol via a `dspy.ReAct` loop with three tools: `search_wiki`, `take_notes`, and `read_notes`.

The `AnswerQuestion` signature instructs the agent to (1) classify the question type (COUNT/ENTITY/ATTRIBUTE), (2) exhaust all anchor entities matching the question's condition, (3) process each anchor independently, and (4) verify completeness via notes before finishing. Thread-local storage ensures safe concurrent evaluation.

## ARCHITECTURE DESCRIPTION:
**Entry point:** `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`) is a `dspy.Module` that configures a ColBERTv2 retrieval model and GPT-4.1-mini LM, then delegates `forward(question)` to `PhantomWikiReAct` within a `dspy.context`.

**Module layer:** `PhantomWikiReAct` (`src/program/phantomwiki_module.py`) owns a `dspy.Retrieve(k=10)` retriever and a `dspy.ReAct` loop (max 50 iterations) bound to the `AnswerQuestion` signature and three tools. Notes state is stored in `threading.local()` for thread-safety during parallel evaluation. Each call to `forward` resets the notes workspace.

**Signature:** `AnswerQuestion` contains a 4-step structured prompt: classify question type → exhaust all anchor entities via repeated searches → process each anchor independently (COUNT returns distinct numeric counts, ENTITY/ATTRIBUTE returns union of results) → verify via notes before calling finish. Output is `list[str]` covering all correct answers.

**Tools:** `search_wiki(query)` retrieves k=10 passages via ColBERTv2 and returns them joined. `take_notes(key, note)` saves findings to a thread-local dict. `read_notes(key)` reads one or all notes, enabling the agent to review its accumulated findings before finalizing the answer.

**Data flow:** question → `PhantomWikiReActPipeline.forward` → `PhantomWikiReAct.forward` (reset notes) → `dspy.ReAct` loop (classify → search anchors → take_notes → process each anchor → read_notes → finish) → `dspy.Prediction(answer=list[str])`.

**Metric:** `phantomwiki_f1_feedback` uses `dspy.evaluate.answer_exact_match` for binary scoring of `pred.answer` vs `example.answer`.
```
